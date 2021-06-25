#!/usr/bin/env python

from detectron2.config import CfgNode as CN
from detectron2.config.defaults import _C as _c


_C = _c.clone()

_C.MODEL.BOTNET = CN()
_C.MODEL.BOTNET.FMAP_SIZE = 64
_C.MODEL.BOTNET.DIM_OUT = 2048
_C.MODEL.BOTNET.PROJ_FACTOR = 4
_C.MODEL.BOTNET.DOWNSAMPLE = True
_C.MODEL.BOTNET.HEADS = 4
_C.MODEL.BOTNET.DIM_HEAD = 256
_C.MODEL.BOTNET.USE_RELATIVE_POS_EMB = True


def get_cfg():

    return _C.clone()

