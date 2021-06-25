#!/usr/bin/env python

from dataclasses import dataclass

import torch

from detectron2.config import CfgNode
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.data import DatasetMapper
from detectron2.data.build import build_detection_train_loader
import detectron2.data.transforms as T

from detectron2_botnet.config import get_cfg
from detectron2_botnet.backbone import build_botnet_backbone, build_botnet_fpn_backbone


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    try:
        cfg.OUTPUT_DIR = args.output_dir
        cfg.MODEL.WEIGHTS = args.weights
    except AttributeError as e:
        print(e)

    cfg.freeze()
    default_setup(cfg, args)
    return cfg


@dataclass
class Trainer(DefaultTrainer):

    cfg: CfgNode = None
    
    def __post_init__(self):
        assert self.cfg
        super(Trainer, self).__init__(self.cfg)

    @classmethod
    def build_train_loader(cls, cfg: CfgNode, mapper=None):
        mapper = DatasetMapper(
            cfg=cfg,
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                ),
                T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
                T.RandomFlip(),
                T.Resize([cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN])
            ],
        )
        return build_detection_train_loader(cfg, mapper=mapper)   

    @classmethod
    def build_augmentation(cls, cfg: CfgNode, is_train: bool = True):
        result = detection_utils.build_augmentation(cfg, is_train=is_train)
        return result


def main(args):

    cfg = setup(args)
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()
    

if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--root', required=False, type=str, default=None)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--num_gpus', required=False, type=int, default=1)
    parser.add_argument('--weights', required=False, type=str, default=None)
    parser.add_argument('--reduced_coco', action='store_true', default=False)
    args = parser.parse_args()

    train = True
    if train:
        launch(main, args.num_gpus, args=(args,), dist_url='auto')
    else:
        main(args)
