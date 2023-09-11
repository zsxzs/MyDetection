import copy
import os.path as osp
from typing import List, Union

from mmengine.fileio import get_local_path

from mydet.registry import DATASETS
from .api_wrappers import COCO
from .base_det_dataset import BaseDetDataset