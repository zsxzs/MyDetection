# Copyright (c) OpenMMLab. All rights reserved.
from mydet.registry import MODELS
from mydet.utils import ConfigType, OptConfigType, OptMultiConfig
from .two_stage import TwoStageDetector


@MODELS.register_module()
class MaskScoringRCNN(TwoStageDetector):
    """Mask Scoring RCNN.

    https://arxiv.org/abs/1903.00241
    """

    def __init__(self,
                 backbone: ConfigType,
                 rpn_head: ConfigType,
                 roi_head: ConfigType,
                 train_cfg: ConfigType,
                 test_cfg: ConfigType,
                 neck: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            roi_head=roi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
