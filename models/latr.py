import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.utils import *
from .latr_head import LATRHead
from .ms2one import build_ms2one
from .networks.feature_extractor import *

class LATR(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.no_cuda = args.no_cuda
        self.batch_size = args.batch_size
        self.num_lane_type = 1  # no centerline
        self.num_y_steps = args.num_y_steps
        self.max_lanes = args.max_lanes
        self.num_category = args.num_category
        _dim_ = args._dim_
        num_query = args.num_query
        num_group = args.num_group
        sparse_num_group = args.sparse_num_group
        top_view_region = args.top_view_region
        enlarge_length = 20
        position_range = [
            top_view_region[0][0] - enlarge_length,
            top_view_region[2][1] - enlarge_length,
            -5,
            top_view_region[1][0] + enlarge_length,
            top_view_region[0][1] + enlarge_length,
            5.]

        self.encoder = self.get_encoder(args) # ResNet101
        self.ms2one = build_ms2one(args.ms2one_type, args._dim_)
        self.neck = FPN(in_channels=[512, 1024, 2048], out_channels = _dim_, 
                             start_level=0, add_extra_convs='on_output', num_outs=4)
        self.head = LATRHead(
            args=args,
            dim=_dim_,
            num_group=num_group,
            num_convs=4,
            in_channels=_dim_,
            kernel_dim=_dim_,
            top_view_region=args.top_view_region,
            num_query=num_query,
            pred_dim=self.num_y_steps,
            num_classes=args.num_category,
            embed_dims=_dim_,
            trans_params=dict(init_z=0, bev_h=150, bev_w=70)
        )

    def forward(self, image, _M_inv=None, is_training=True, extra_dict=None):
        out_featList = self.encoder(image)
        out_featList = out_featList[2:]
        neck_out = self.neck(out_featList)
        neck_out = self.ms2one(neck_out)
        output = self.head(
            dict(
                x=neck_out,
                lane_idx=extra_dict['seg_idx_label'],
                seg=extra_dict['seg_label'],
                lidar2img=extra_dict['lidar2img'],
                pad_shape=extra_dict['pad_shape'],
                ground_lanes=extra_dict['ground_lanes'] if is_training else None,
                ground_lanes_dense=extra_dict['ground_lanes_dense'] if is_training else None,
                image=image,
            ),
            is_training=is_training,
        )
        return output

    def get_encoder(self, args):
        if args.encoder == 'ResNext101':
            return deepFeatureExtractor_ResNext101(lv6=False)
        elif args.encoder == 'VGG19':
            return deepFeatureExtractor_VGG19(lv6=False)
        elif args.encoder == 'DenseNet161':
            return deepFeatureExtractor_DenseNet161(lv6=False)
        elif args.encoder == 'InceptionV3':
            return deepFeatureExtractor_InceptionV3(lv6=False)
        elif args.encoder == 'MobileNetV2':
            return deepFeatureExtractor_MobileNetV2(lv6=False)
        elif args.encoder == 'ResNet101':
            return deepFeatureExtractor_ResNet101(lv6=True)
        # elif 'EfficientNet' in args.encoder:
        #     return deepFeatureExtractor_EfficientNet(args.encoder, lv6=False, lv5=False, lv4=False, lv3=False)
        else:
            raise Exception("encoder model in args is not supported")
def make_layers(cfg, in_channels=3, batch_norm=False):
    layers= []
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def make_one_layer(in_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=False, inplace=True):
    conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride)
    if batch_norm:
        layers = [conv2d, nn.BatchNorm2d(out_channels), nn.ReLU(inplace)]
    else:
        layers = [conv2d, nn.ReLU(inplace)]
    return layers

class FPN(nn.Module):
    def __init__(self,
                 in_channels,             # RetinaNet为例 [256, 512, 1024, 2048]
                 out_channels,            # 256
                 num_outs,                # 5
                 start_level=0,           # 1
                 end_level=-1,
                 add_extra_convs=False,   # 'on_input'
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False):
                #  conv_cfg=None, #conv2d
                #  norm_cfg=None, #BN
                #  act_cfg=None,  #relu
                #  upsample_cfg=dict(mode='nearest'),# deconv
                #  init_cfg=dict(
                #  type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FPN, self).__init__()
        assert isinstance(in_channels, list)
        self.in_channels = in_channels                              # self.in_channels = [256, 512, 1024, 2048]
        self.out_channels = out_channels                            # self.out_channels = 256    对应图中M3-M5的channel数为256
        self.num_ins = len(in_channels)                             # self.num_ins = 4
        self.num_outs = num_outs                                    # self.num_outs = 5     对应图中P3-P7
        # 下面4个参数对于结构理解关系不大
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        # self.fp16_enabled = False
        # self.upsample_cfg = upsample_cfg.copy() # 上采样参数

        if end_level == -1 or end_level == self.num_ins - 1:
            self.backbone_end_level = self.num_ins                  # self.backbone_end_level = 4
            assert num_outs >= self.num_ins - start_level
        else:
            # if end_level is not the last level, no extra level is allowed
            self.backbone_end_level = end_level + 1
            assert end_level < self.num_ins
            assert num_outs == end_level - start_level + 1
        self.start_level = start_level                              # self.start_level = 1
        self.end_level = end_level                                  # self.end_level = -1
        self.add_extra_convs = add_extra_convs                      # self.add_extra_convs = 'on_input'
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            # Extra_convs_source choices: 'on_input', 'on_lateral', 'on_output'
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:  # True
            self.add_extra_convs = 'on_input'

        
        self.lateral_convs = nn.ModuleList()        # 对应图中橙色虚线框
        self.fpn_convs = nn.ModuleList()            # 对应图中绿色虚线框

        for i in range(self.start_level, self.backbone_end_level):    # start_level = 1, backbone_end_level = 4，整体数量为3
            # 构造conv 1x1，对应图中3个橙色矩阵
            l_conv = nn.Sequential(
                nn.Conv2d(in_channels[i], out_channels, 1, stride=1, padding=0),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
                ),
            fpn_conv = nn.Sequential(
                nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True)
                ),
            # 构造conv 3x3，对应图中3个绿色矩阵

            self.lateral_convs.append(*l_conv)
            self.fpn_convs.append(*fpn_conv)

        # 添加额外的conv level (e.g., RetinaNet)
        extra_levels = num_outs - self.backbone_end_level + self.start_level    # extra_levels = 5 - 4 + 1 = 2  
        # 其实不论怎么样这个extra_levels都会>=1（当前理解的也就是，在默认情况下图中的Output中的绿色矩形始终存在）
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):    # 2
                if i == 0 and self.add_extra_convs == 'on_input':                # 当i == 0时，满足条件
                    in_channels = self.in_channels[self.backbone_end_level - 1]  # 当i == 0时，in_channels = in_channels[3] 也即2048，此时构造的对应图中紫色的矩阵
                else:                                                            # 当i == 0时，in_channels = 256
                    in_channels = out_channels
                # 构造conv 3x3, stride=2
                extra_fpn_conv = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True)
                )
                self.fpn_convs.append(extra_fpn_conv)
        # 因此RetinaNet最终fpn_convs中有5块Conv块，即对应图中绿色虚线框关联的内容有5块
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # laterals 用来记录每一次计算后的输出值，可以理解成是一个临时变量temp
        laterals = [
            lateral_conv(inputs[i + self.start_level])              # self.start_level = 1，inputs[i + 1]为C3-C5的输入
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]
        # 此时，laterals 已经记录了C3-C5经过conv 1x1之后得到的M3-M5值(还未upsample)
        
        # build top-down path
        used_backbone_levels = len(laterals)                # 3
        for i in range(used_backbone_levels - 1, 0, -1):    # i in [2,1]
            # In some cases, fixing `scale factor` (e.g. 2) is preferred, but
            #  it cannot co-exist with `size` in `F.interpolate`.
            # 这里也就是upsample与相加的操作，可以理解成经过“upsample”与“+”的操作后，才得到真正的M3-M5的值
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='bilinear', align_corners=True)
        # 此时，laterals 记录了经过upsample之后得到的新M3-M5值


        # 建立 outputs
        # part 1: from original levels 此处out对应P3-P5
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)   # used_backbone_levels = 3
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):       # self.num_outs = 5
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:     # self.add_extra_convs = 'on_input'
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':             # 满足条件
                    extra_source = inputs[self.backbone_end_level - 1]  # self.backbone_end_level - 1 = 3 , extra_source 对应图中的C5
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                # 此处outs增加P6
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))   # self.fpn_convs[used_backbone_levels]对应图中紫色的矩阵
                for i in range(used_backbone_levels + 1, self.num_outs): # i in [4]
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        # 此处out增加P7
                        outs.append(self.fpn_convs[i](outs[-1]))  # self.fpn_convs[i]对应con3x3,stride=2     outs[-1]对应P6     这里也对应了之前提到的“在默认情况下图中的Output中的绿色矩形始终存在”
        return tuple(outs)

