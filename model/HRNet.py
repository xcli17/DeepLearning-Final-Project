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
        self.conv1 = Conv2d(in_planes=inplanes, out_planes=planes, stride=stride)
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
        num_inchannels_ = [x*self.block.expansion for x in num_channels]
        self.fuse_layers = self.__make_fuse_layers(num_inchannels_)
        self.num_inchannels = num_inchannels_
        self.relu = nn.ReLU(inplace=False)
    
    @staticmethod
    def _make_branch(block:Union[BasicBlock, Bottleneck], num_block:int, num_inchannel:int, num_channel:int, stride:int=1, **kwargs):
        downsample = None
        if stride != 1 or num_inchannel != num_channel * block.expansion:
            downsample = Conv2d(in_planes=num_inchannel, out_planes=num_channel*block.expansion, kernel_size=1, stride=stride)
        layers = []
        layers.append(block(inplanes=num_inchannel, planes=num_channel, stride=stride, downsample=downsample))
        num_inchannel_ = num_channel * block.expansion
        for _ in range(1, num_block):
            layers.append(block(inplanes=num_inchannel_, planes=num_channel))
        return nn.Sequential(*layers)

    def __make_branches(self, num_branches:int, num_blocks:Sequence[int], num_inchannels:Sequence[int], num_channels:Sequence[int]):
        branches = []
        for i in range(num_branches):
            branches.append(self._make_branch(block=self.block, num_block=num_blocks[i], num_inchannel=num_inchannels[i], num_channel=num_channels[i]))
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
                            Conv2d(in_planes=num_inchannels_[j], out_planes=num_inchannels_[i], kernel_size=1),
                            nn.Upsample(scale_factor=2**(j-i), mode='bilinear', align_corners=True)
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    downsample_unit = nn.Sequential(
                                            Conv2d(in_planes=num_inchannels_[j], out_planes=num_inchannels_[j], stride=2),
                                            nn.ReLU(inplace=False)
                                        )
                    downsample = [downsample_unit] * max(i - j - 1, 0) + \
                                [nn.Sequential(
                                    Conv2d(in_planes=num_inchannels_[j], out_planes=num_inchannels_[i], stride=2),
                                    nn.ReLU(inplace=False)
                                )]
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
        block = BasicBlock
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
        block = BasicBlock
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
        block = BasicBlock
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
                                            Conv2d(in_planes=in_channels[-1], out_planes=in_channels[-1], stride=2),
                                            nn.ReLU(inplace=False)
                                        )
                downsample = [downsample_unit] * max(i - num_in_channels - 1, 0) + \
                            [nn.Sequential(
                                Conv2d(in_planes=in_channels[-1], out_planes=out_channels[i], stride=2),
                                nn.ReLU(inplace=False)
                            )]
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
                                        num_inchannels=num_inchannels, num_channels=num_channels, 
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


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial 
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, num_classes:int=0, scale:int=1):
        super(SpatialGather_Module, self).__init__()
        self.num_classes = num_classes
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.size(1), -1)
        feats = feats.permute(0, 2, 1) # batch x hw x c 
        probs = F.softmax(self.scale * probs, dim=2)# batch x k x hw
        # ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch x k x c
        ocr_context = torch.bmm(probs, feats).permute(0,2,1).unsqueeze(3) # b x c x k x 1
        return ocr_context

class ObjectAttentionBlock2D(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
    Return:
        N X C X H X W
    '''
    def __init__(self, in_channels, key_channels, scale=1):
        super(ObjectAttentionBlock2D, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            Conv2d(in_planes=self.in_channels, out_planes=self.key_channels, kernel_size=1),
            nn.ReLU(),
            Conv2d(in_planes=self.key_channels, out_planes=self.key_channels, kernel_size=1),
            nn.ReLU()
        )
        self.f_object = nn.Sequential(
            Conv2d(in_planes=self.in_channels, out_planes=self.key_channels, kernel_size=1),
            nn.ReLU(),
            Conv2d(in_planes=self.key_channels, out_planes=self.key_channels, kernel_size=1),
            nn.ReLU()
        )
        self.f_down = nn.Sequential(
            Conv2d(in_planes=self.in_channels, out_planes=self.key_channels, kernel_size=1),
            nn.ReLU()
        )
        self.f_up = nn.Sequential(
            Conv2d(in_planes=self.key_channels, out_planes=self.in_channels, kernel_size=1),
            nn.ReLU()
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-0.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)   

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context

class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self, in_channels, key_channels, out_channels, scale=1, dropout=0.1, ):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels=in_channels, key_channels=key_channels, scale=scale)
        _in_channels = 2 * in_channels
        self.conv_bn_dropout = nn.Sequential(
            Conv2d(in_planes=_in_channels, out_planes=out_channels, kernel_size=1),
            nn.ReLU(),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)
        output = self.conv_bn_dropout(torch.cat([context, feats], 1))
        return output


class HighResolutionNet_OCR(nn.Module):
    def __init__(self, num_classes:int, W:int=32, fuse_method:str="sum", **kwargs):
        super(HighResolutionNet_OCR, self).__init__()
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
        block = BasicBlock
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
        block = BasicBlock
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
        block = BasicBlock
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
        
        #OCR
        ocr_mid_channels = 512
        ocr_key_channels = 256

        self.connection_to_ocr = nn.Sequential(
            Conv2d(in_planes=last_inp_channels, out_planes=ocr_mid_channels),
            nn.ReLU()
        )
        self.ocr_gather_head = SpatialGather_Module(num_classes=num_classes)
        self.ocr_distri_head = SpatialOCR_Module(in_channels=ocr_mid_channels, key_channels=ocr_key_channels, out_channels=ocr_mid_channels, scale=1, dropout=0.05)
        self.cls_head = nn.Conv2d(in_channels=ocr_mid_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        self.aux_head = nn.Sequential(
            Conv2d(in_planes=last_inp_channels, out_planes=last_inp_channels, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0, bias=True)
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
                try:
                    downsample_unit = nn.Sequential(
                                            Conv2d(in_planes=in_channels[-1], out_planes=in_channels[-1], stride=2),
                                            nn.ReLU(inplace=False)
                                        )
                except:
                    print(in_channels)
                downsample = [downsample_unit] * max(i - num_in_channels - 1, 0) + \
                            [nn.Sequential(
                                Conv2d(in_planes=in_channels[-1], out_planes=out_channels[i], stride=2),
                                nn.ReLU(inplace=False)
                            )]
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
                                        num_inchannels=num_inchannels, num_channels=num_channels, 
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

        feats = x
        out_aux_seg = []
        out_aux = self.aux_head(feats)
        feats = self.connection_to_ocr(feats)
        context = self.ocr_gather_head(feats, out_aux)
        feats = self.ocr_distri_head(feats, context)
        out = self.cls_head(feats)
        out_aux_seg = [out_aux, out]
        return out_aux_seg

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, Conv2d):
                module.init_weights()


def get_HRNet_OCR_model(num_classes, W=32, **kwargs):
    model = HighResolutionNet_OCR(num_classes, W=32, **kwargs)
    model.init_weights()
    return model