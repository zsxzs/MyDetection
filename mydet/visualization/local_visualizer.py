import cv2
import torch
import mmcv
import numpy as np

try:
    import seaborn as sns
except ImportError:
    sns = None
    
from mmengine.dist import master_only
from mmengine.structures import InstanceData, PixelData
from mmengine.visualization import Visualizer

from typing import Dict, List, Optional, Tuple, Union

from ..evaluation import INSTANCE_OFFSET
from ..registry import VISUALIZERS
from ..structures import DetDataSample
from ..structures.mask import BitmapMasks, PolygonMasks, bitmap_to_polygon
from .palette import _get_adaptive_scales, get_palette, jitter_color
