'''
High-Resolution Representations for Labeling Pixels and Regions
arXiv:1904.04514
Reference: https://github.com/HRNet/HRNet-Semantic-Segmentation/
'''
from typing import Union, Sequence, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv


class Conv2d(nn.Module):
    '''
    Conv2d + BatchNorm2
    '''
    def __init__(self, in_planes:int, out_planes:int, kernel_size:Union[int, Tuple[int, int]]=3, 
                    stride:int=1, padding:Union[int, None]=None, bias:bool=False):
        super(Conv2d, self).__init__()
        if padding is None:
            if kernel_size == 3 or kernel_size == (3, 3):
                padding = 1
            elif kernel_size == 1 or kernel_size == (1, 1):
                padding = 0
            else:
                padding = 0
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size, 
                                stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(num_features=out_planes, momentum=0.10)
        # self.activation = nn.LeakyReLU(negative_slope=0.01)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # x = self.activation(x)
        return x

    def init_weights(self):
        nn.init.normal_(self.conv.weight, mean=0, std=0.001)
        nn.init.constant_(self.bn.weight, 1)
        nn.init.constant_(self.bn.bias, 0)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes:int, planes:int, stride:int=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = Conv2d(in_planes=inplanes, planes=planes, stride=stride)
        self.conv2 = Conv2d(in_planes=planes, out_planes=planes)
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=False)
        self.stride = stride
    
    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.relu(output)
        output = self.conv2(output)
        if self.downsample is not None:
            residual = self.downsample(residual)
        output = output + residual
        output = self.relu(output)
        return output

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes:int, planes:int, stride:int=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = Conv2d(in_planes=inplanes, out_planes=planes, kernel_size=1)
        self.conv2 = Conv2d(in_planes=planes, out_planes=planes, stride=stride)
        self.conv3 = Conv2d(in_planes=planes, out_planes=planes*self.expansion, kernel_size=1)
        self.relu = nn.ReLU(inplace=False)
        self.downsample = downsample
        self.stride = stride
    
    def forward(self, x):
        residual = x
        output = self.conv1(x)
        output = self.relu(output)
        output = self.conv2(output)
        output = self.relu(output)
        output = self.conv3(output)
        if self.downsample is not None:
            residual = self.downsample(residual)
        output = output + residual
        output = self.relu(output)
        return output


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches:int, block:Union[BasicBlock, Bottleneck], num_blocks:Sequence[int], 
                    num_inchannels:Sequence[int], num_channels:Sequence[int], fuse_method:str, multi_scale_output:bool=True):
        super(HighResolutionModule, self).__init__()
        assert(num_branches == len(num_blocks) == len(num_inchannels) == len(num_channels))
        self.num_branches = num_branches
        self.block = block
        self.fuse_method = fuse_method
        self.fuse_func = None
        self.multi_scale_output = multi_scale_output
        self.branches = self.__make_branches(num_branches=num_branches, num_blocks=num_blocks, 
                                                num_inchannels=num_inchannels, num_channels=num_channels)
        num_inchannels_ = [x*self.block.expansion for x in num_inchannels]
        self.fuse_layers = self.__make_fuse_layers(num_inchannels_)
        self.num_inchannels = num_inchannels_
        self.relu = nn.ReLU(inplace=True)
    
    @staticmethod
    def _make_branch(block:Union[BasicBlock, Bottleneck], num_block:int, num_inchannel:int, num_channel:int, stride:int=1, **kwargs):
        downsample = None
        if stride != 1 or num_inchannel != num_channel * block.expansion:
            downsample = Conv2d(in_planes=num_inchannel, out_planes=num_channel*block.expansion, kernel_size=1, stride=stride)
        layers = []
        layers.append(block(inplanes=num_inchannel, planes=num_channel, stride=stride, downsample=downsample))
        num_inchannel_ = num_inchannel * block.expansion
        for i in range(1, num_block):
            layers.append(block(inplanes=num_inchannel_, planes=num_channel))
        return nn.Sequential(*layers)

    def __make_branches(self, num_branches:int, num_blocks:Sequence[int], num_inchannels, num_channels:Sequence[int])
        branches = []
        for i in range(num_branches):
            branches.append(self._make_branch(block=self.block, num_block=num_blocks[i], num_inchannels=num_inchannels[i], num_channel=num_channels[i]))
        return nn.ModuleList(branches)
    
    def __make_fuse_layers(self, num_inchannels_:Sequence[int]):
        if self.num_branches == 1:
            return None
        num_branches_fused = self.num_branches if self.multi_scale_output else 1
        fuse_layers = []
        for i in range(num_branches_fused):
            fuse_layer = []
            for j in range(self.num_branches):
                if j > i:
                    fuse_layer.append(nn.Sequential(
                            Conv2d(in_planes=num_inchannels_[j], out_planes=num_inchannels_[i], kernel_size=1)
                            nn.Upsample(scale_factor=2**(j-i), mode='bilinear')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    downsample_unit = nn.Sequential(
                                            Conv2d(in_planes=num_inchannels_[j], out_planes=num_inchannels_[j], stride=2),
                                            nn.ReLU(inplace=True)
                                        )
                    downsample = [downsample_unit] * max(i - j - 1, 0) + 
                                    [Conv2d(in_planes=num_inchannels_[j], out_planes=num_inchannels_[i], stride=2)]
                    fuse_layer.append(nn.Sequential(*downsample))
            fuse_layers.append(nn.ModuleList(fuse_layer))
        return nn.ModuleList(fuse_layers)

    def fuse(self, y, fuse_ji):
        if self.fuse_method == "sum":
            return y + fuse_ji
        elif self.fuse_method == "cat_conv":
            if self.fuse_func is None:
                out_planes = y.shape[-3]
                self.fuse_func = Conv2d(in_planes=2*out_planes, out_planes=out_planes, kernel_size=1)
            return self.fuse_func(torch.cat([y, fuse_ji], dim=-3))

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]
        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])
        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = x[i]
            for j in range(self.num_branches):
                if j == i:
                    continue
                y = self.fuse(y, self.fuse_layers[i][j](x[j]))
            x_fuse.append(self.relu(y))
        return x_fuse


class HighResolutionNet(nn.Module):
    def __init__(self, num_classes:int, W:int=32, fuse_method:str="sum", **kwargs):
        super(HighResolutionNet, self).__init__()
        self.fuse_method = fuse_method

        # Stem
        self.conv1 = Conv2d(3, 64, stride=2)
        self.conv2 = Conv2d(64, 64, stride=2)
        self.relu = nn.ReLU(inplace=False)

        #stage1
        self.stage1_config = {
            "num_modules": 1,
            "num_branches": 1,
            "block": Bottleneck,
            "num_block": 4,
            "num_inchannel": 64,
            "num_channel": 32,
            "fuse_method": self.fuse_method
        }
        # self.layer1 = self.__make_layer(block=Bottleneck, num_block=num_blocks, num_inchannel=64, num_channel=num_channels)
        self.layer1 = HighResolutionModule._make_branch(**self.stage1_config)
        pre_stage_channels = [self.stage1_config["num_channel"] * self.stage1_config["block"].expansion] #32*4=128

        #stage2
        num_channels = [W, 2*W]
        num_inchannels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.stage2_config = {
            "in_channels": pre_stage_channels,
            "out_channels": num_inchannels,
            "num_modules": 1,
            "num_branches": 2,
            "block": BasicBlock,
            "num_blocks": [4, 4],
            "num_inchannels": num_inchannels,
            "num_channels": num_channels,
            "fuse_method": self.fuse_method
        }
        self.transition1 = self.__make_transition_layer(**self.stage2_config)
        self.stage2, pre_stage_channels = self.__make_stage(**self.stage2_config)

        #stage3
        num_channels = [W, 2*W, 4*W]
        num_inchannels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.stage3_config = {
            "in_channels": pre_stage_channels,
            "out_channels": num_inchannels,
            "num_modules": 1,
            "num_branches": 3,
            "block": BasicBlock,
            "num_blocks": [4, 4, 4],
            "num_inchannels": num_inchannels,
            "num_channels": num_channels,
            "fuse_method": self.fuse_method
        }
        self.transition2 = self.__make_transition_layer(**self.stage3_config)
        self.stage3, pre_stage_channels = self.__make_stage(**self.stage3_config)

        #stage4
        num_channels = [W, 2*W, 4*W, 8*W]
        num_inchannels = [num_channels[i] * block.expansion for i in range(len(num_channels))]
        self.stage4_config = {
            "in_channels": pre_stage_channels,
            "out_channels": num_inchannels,
            "num_modules": 1,
            "num_branches": 4,
            "block": BasicBlock,
            "num_blocks": [4, 4, 4, 4],
            "num_inchannels": num_inchannels,
            "num_channels": num_channels,
            "fuse_method": self.fuse_method
        }
        self.transition3 = self.__make_transition_layer(**self.stage4_config)
        self.stage4, pre_stage_channels = self.__make_stage(**self.stage4_config)

        #last layer
        last_inp_channels = np.int(np.sum(pre_stage_channels))
        self.last_layer = nn.Sequential(
            Conv2d(in_planes=last_inp_channels, out_planes=last_inp_channels, kernel_size=1),
            nn.ReLU(inplace=False),
            Conv2d(in_planes=last_inp_channels, out_planes=num_classes, kernel_size=1)
        )

    def __make_transition_layer(self, in_channels:Sequence[int], out_channels:Sequence[int], **kwargs):
        num_in_channels = len(in_channels)
        num_out_channels = len(out_channels)
        transition_layers = []
        for i in range(num_out_channels):
            if i < num_in_channels:
                if out_channels[i] != in_channels[i]:
                    transition_layers.append(nn.Sequential(
                        Conv2d(in_planes=in_channels[i], out_planes=out_channels[i], kernel_size=3),
                        nn.ReLU(inplace=False)))
                else:
                    transition_layers.append(None)
            else:
                downsample_unit = nn.Sequential(
                                            Conv2d(in_planes=num_inchannels_[j], out_planes=num_inchannels_[j], stride=2),
                                            nn.ReLU(inplace=True)
                                        )
                downsample = [downsample_unit] * max(i - num_in_channels - 1, 0) + 
                                [Conv2d(in_planes=num_inchannels_[j], out_planes=num_inchannels_[i], stride=2)]
                transition_layers.append(nn.Sequential(*downsample))
        return nn.ModuleList(transition_layers)

    def __make_stage(self, num_modules:int, num_branches:int, block:Union[BasicBlock, Bottleneck], 
                        num_blocks:Sequence[int], num_inchannels:Sequence[int], num_channels:Sequence[int], 
                        fuse_method:str="sum", multi_scale_output:bool=True, **kwargs):
        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True
            modules.append(
                HighResolutionModule(num_branches=num_branches, block=block, num_blocks=num_blocks,
                                        num_channels=num_inchannels, num_channels=num_channels, 
                                        fuse_method=fuse_method, multi_scale_output=reset_multi_scale_output)
            )
            num_inchannels = modules[-1].num_inchannels
        return nn.Sequential(*modules), num_inchannels

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)

        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_config['num_branches']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_config['num_branches']):
            if self.transition2[i] is not None:
                if i < self.stage2_config['num_branches']:
                    x_list.append(self.transition2[i](y_list[i]))
                else:
                    x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_config['num_branches']):
            if self.transition3[i] is not None:
                if i < self.stage3_config['num_branches']:
                    x_list.append(self.transition3[i](y_list[i]))
                else:
                    x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        x = self.stage4(x_list)

        # Upsampling
        x0_h, x0_w = x[0].size(2), x[0].size(3)
        x1 = F.upsample(x[1], size=(x0_h, x0_w), mode='bilinear')
        x2 = F.upsample(x[2], size=(x0_h, x0_w), mode='bilinear')
        x3 = F.upsample(x[3], size=(x0_h, x0_w), mode='bilinear')
        x = torch.cat([x[0], x1, x2, x3], 1)

        x = self.last_layer(x)

        return x

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, Conv2d):
                module.init_weights()


def get_HRNet_model(num_classes, W=32, **kwargs):
    model = HighResolutionNet(num_classes, W=32, **kwargs)
    model.init_weights()
    return model
