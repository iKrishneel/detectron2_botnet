#!/usr/bin/env python

import os
from dataclasses import dataclass

import torch

from detectron2.config import CfgNode
from detectron2.engine import (
    DefaultTrainer,
    default_argument_parser,
    default_setup,
    launch,
)
from detectron2.data import (
    detection_utils,
    DatasetCatalog,
    MetadataCatalog,
    DatasetMapper,
)
from detectron2.modeling import build_model
from detectron2.data.build import (
    build_detection_train_loader,
    build_detection_test_loader,
)
import detectron2.data.transforms as T
from detectron2.evaluation import COCOEvaluator

from detectron2_botnet.config import get_cfg
from detectron2_botnet.backbone import (
    build_botnet_backbone,
    build_botnet_fpn_backbone,
)


def setup(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)

    cfg.MODEL.BACKBONE.NAME = "build_botnet_fpn_backbone"
    cfg.MODEL.BACKBONE.FREEZE_AT = 0

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
    def build_train_loader1(cls, cfg: CfgNode):
        mapper = DatasetMapper(
            cfg=cfg,
            is_train=True,
            augmentations=[
                T.ResizeShortestEdge(
                    cfg.INPUT.MIN_SIZE_TRAIN,
                    cfg.INPUT.MAX_SIZE_TRAIN,
                    cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                ),
                # T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE),
                T.RandomFlip(),
                T.Resize([cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN]),
            ],
        )
        return build_detection_train_loader(cfg, mapper=mapper)

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        mapper = DatasetMapper(
            cfg=cfg,
            is_train=False,
            augmentations=[
                T.Resize([cfg.INPUT.MAX_SIZE_TRAIN, cfg.INPUT.MAX_SIZE_TRAIN])
            ],
        )
        return build_detection_test_loader(cfg, dataset_name, mapper=mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == 'coco':
            evaluator = COCOEvaluator(dataset_name=dataset_name)
        else:
            raise ValueError('Evaluator type is unknown!')
        return evaluator

    @classmethod
    def build_augmentation(cls, cfg: CfgNode, is_train: bool = True):
        result = detection_utils.build_augmentation(cfg, is_train=is_train)
        return result


def main(args):

    cfg = setup(args)

    """
    if args.eval_only:
        model = build_model(cfg)
        model.load_state_dict(torch.load(args.weights)['model'])
        return Trainer.test(cfg, model)
    """

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)

    return trainer.train()


if __name__ == '__main__':
    parser = default_argument_parser()
    parser.add_argument('--root', required=False, type=str, default=None)
    parser.add_argument('--output_dir', required=True, type=str)
    parser.add_argument('--num_gpus', required=False, type=int, default=1)
    parser.add_argument('--weights', required=False, type=str, default=None)
    parser.add_argument('--eval_only', action='store_true', default=False)
    args = parser.parse_args()

    if args.eval_only:
        print("eval")
        result = main(args=args)
        print("result \n", result)
    else:
        launch(main, args.num_gpus, args=(args,), dist_url='auto')
