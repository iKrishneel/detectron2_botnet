#!/usr/bin/env python

import torch.nn as nn

from detectron2.modeling import (
    BACKBONE_REGISTRY,
    build_resnet_backbone,
    ResNet,
    ShapeSpec,
    FPN,
)
from detectron2.modeling.backbone import BottleneckBlock, BasicStem
from detectron2.modeling.backbone.fpn import LastLevelMaxPool
from detectron2.layers import CNNBlockBase

from bottleneck_transformer_pytorch import BottleStack as Stack


def make_stack(**kwargs: dict):
    block_class = kwargs.pop('block_class')
    return block_class(**kwargs)


def get_stride(blocks):
    stride = 1
    for block in blocks:
        if not isinstance(block, BottleneckBlock):
            continue
        stride *= block.stride
    return stride


class BottleStack(CNNBlockBase):
    def __init__(self, **kwargs):
        super(BottleStack, self).__init__(
            kwargs['dim'], kwargs['dim_out'], stride=2
        )
        self.blocks = Stack(**kwargs)

    def forward(self, x):
        return self.blocks(x)


@BACKBONE_REGISTRY.register()
def build_botnet_backbone(cfg, input_shape: ShapeSpec):
    freeze_at = cfg.MODEL.BACKBONE.FREEZE_AT
    out_features = cfg.MODEL.RESNETS.OUT_FEATURES
    depth = cfg.MODEL.RESNETS.DEPTH
    num_groups = cfg.MODEL.RESNETS.NUM_GROUPS
    width_per_group = cfg.MODEL.RESNETS.WIDTH_PER_GROUP
    bottleneck_channels = num_groups * width_per_group
    in_channels = cfg.MODEL.RESNETS.STEM_OUT_CHANNELS
    out_channels = cfg.MODEL.RESNETS.RES2_OUT_CHANNELS
    stride_in_1x1 = cfg.MODEL.RESNETS.STRIDE_IN_1X1
    res5_dilation = cfg.MODEL.RESNETS.RES5_DILATION
    # deform_on_per_stage = cfg.MODEL.RESNETS.DEFORM_ON_PER_STAGE
    # deform_modulated    = cfg.MODEL.RESNETS.DEFORM_MODULATED
    # deform_num_groups   = cfg.MODEL.RESNETS.DEFORM_NUM_GROUPS
    norm = cfg.MODEL.RESNETS.NORM
    stem = BasicStem(
        in_channels=input_shape.channels,
        out_channels=cfg.MODEL.RESNETS.STEM_OUT_CHANNELS,
        norm=norm,
    )

    num_blocks_per_stage = {
        50: [3, 4, 6, 3],
    }[depth]

    stages = []
    total_stride = stem.stride
    for idx, stage_idx in enumerate(range(2, 6)):
        if idx < len(num_blocks_per_stage) - 1:
            dilation = res5_dilation if stage_idx == 5 else 1
            first_stride = (
                1 if idx == 0 or (stage_idx == 5 and dilation == 2) else 2
            )
            stage_kargs = {
                "num_blocks": num_blocks_per_stage[idx],
                "stride_per_block": [first_stride]
                + [1] * (num_blocks_per_stage[idx] - 1),
                "in_channels": in_channels,
                "out_channels": out_channels,
                "norm": norm,
            }
            stage_kargs["bottleneck_channels"] = bottleneck_channels
            stage_kargs["stride_in_1x1"] = stride_in_1x1
            stage_kargs["dilation"] = dilation
            stage_kargs["num_groups"] = num_groups
            stage_kargs["block_class"] = BottleneckBlock
            blocks = ResNet.make_stage(**stage_kargs)

            total_stride *= get_stride(blocks)
            stages.append(blocks)
        else:
            # fmap_size = input_shape.height // total_stride
            stage_kargs = {}
            stage_kargs["dim"] = out_channels // 2
            # stage_kargs["fmap_size"] = fmap_size
            stage_kargs["fmap_size"] = cfg.MODEL.BOTNET.FMAP_SIZE
            stage_kargs["dim_out"] = cfg.MODEL.BOTNET.DIM_OUT
            stage_kargs["proj_factor"] = cfg.MODEL.BOTNET.PROJ_FACTOR
            stage_kargs["downsample"] = cfg.MODEL.BOTNET.DOWNSAMPLE
            stage_kargs["heads"] = cfg.MODEL.BOTNET.HEADS
            stage_kargs["dim_head"] = cfg.MODEL.BOTNET.DIM_HEAD
            stage_kargs["rel_pos_emb"] = cfg.MODEL.BOTNET.USE_RELATIVE_POS_EMB
            stage_kargs["activation"] = nn.ReLU()
            stage_kargs["block_class"] = BottleStack
            blocks = make_stack(**stage_kargs)
            stages.append([blocks])

        in_channels = out_channels
        out_channels *= 2
        bottleneck_channels *= 2

    return ResNet(stem, stages, out_features=out_features, freeze_at=freeze_at)


@BACKBONE_REGISTRY.register()
def build_botnet_fpn_backbone(cfg, input_shape: ShapeSpec):
    bottom_up = build_botnet_backbone(cfg, input_shape)
    in_features = cfg.MODEL.FPN.IN_FEATURES
    out_channels = cfg.MODEL.FPN.OUT_CHANNELS
    backbone = FPN(
        bottom_up=bottom_up,
        in_features=in_features,
        out_channels=out_channels,
        norm=cfg.MODEL.FPN.NORM,
        top_block=LastLevelMaxPool(),
        fuse_type=cfg.MODEL.FPN.FUSE_TYPE,
    )
    return backbone


if __name__ == '__main__':

    from detectron2_botnet.config import get_cfg

    s = ShapeSpec(3, 1024, 1024)
    c = get_cfg()
    c.MODEL.RESNETS.OUT_FEATURES = ['res5']
    c.MODEL.BOTNET.DIM_OUT = 2048
    m = build_botnet_backbone(c, s)

    import IPython

    IPython.embed()
