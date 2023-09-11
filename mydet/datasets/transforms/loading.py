from typing import Optional, Tuple, Union

import mmcv
import numpy as np
import pycocotools.mask as maskUtils
import torch
from mmcv.transforms import BaseTransform
from mmcv.transforms import LoadAnnotations as MMCV_LoadAnnotations
from mmcv.transforms import LoadImageFromFile
from mmengine.fileio import get
from mmengine.structures import BaseDataElement

from mydet.registry import TRANSFORMS
from mydet.structures.bbox import get_box_type
from mydet.structures.bbox.box_type import autocast_box_type
from mydet.structures.mask import BitmapMasks, PolygonMasks

@TRANSFORMS.register_module()
class LoadAnnotations(MMCV_LoadAnnotations):
    """
        加载和组织标注信息，如 bbox、语义分割图等
        https://mmcv.readthedocs.io/zh_CN/2.x/api/generated/mmcv.transforms.LoadAnnotations.html#mmcv.transforms.LoadAnnotations
    """
    def __init__(self, 
                 with_mask: bool = False,
                 poly2mask: bool = True,
                 box_type: str = 'hbox',
                 # use for semseg
                 reduce_zero_label: bool = False,
                 ignore_index: int = 255,
                 **kwargs) -> None:
        super(LoadAnnotations, self).__init__(**kwargs)
        self.with_mask = with_mask
        self.poly2mask = poly2mask
        self.box_type = box_type
        self.reduce_zero_label = reduce_zero_label
        self.ignore_index = ignore_index
        
    def _load_bboxes(self, results: dict) -> None:
        gt_bboxes = []
        gt_ignore_flags = []
        for instance in results.get('instance', []):
            gt_bboxes.append(instance['bbox'])
            gt_ignore_flags.append(instance['ignore_flag'])
        if self.box_type is None:
            results['gt_bboxes'] = np.array(gt_bboxes, 
                                            dtype=np.float32).reshape((-1, 4))
        else:
            _, box_type_cls = get_box_type(self.box_type)
            results['gt_bboxes'] = box_type_cls(gt_bboxes, dtype=torch.float32)
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        
    def _load_labels(self, results: dict) -> None:
        get_bboxes_labels = []
        for instance in results.get('instances', []):
            get_bboxes_labels.append(instance['bbox_label'])
        # TODO: 待解决的问题（与mmcv不一致）
        results['gt_bboxes_labels'] = np.array(get_bboxes_labels, dtype=np.int64)
        
    def _poly2mask(self, mask_ann: Union[list, dict], 
                   img_h: int, img_w: int) -> np.ndarray:
        """
        额外的方法：将多边形Polygon的坐标转化成bitmap，即像素值的形式。
        """
        if issubclass(mask_ann, list):
            # 一个物体的可能由好几个polygon区域构成
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask
    
    def _process_masks(self, results: dict) -> list:
        # Process gt_masks and filter invalid polygons.
        gt_masks = []
        gt_ignore_flags = []
        for instance in results.get('instances', []):
            gt_mask = instance['mask']
            if isinstance(gt_mask, list):
                gt_mask = [
                    np.array(polygon) for polygon in gt_mask
                    if len(polygon) % 2 == 0 and len(polygon) >= 6 # 判断是否有效
                ]
                if len(gt_mask) == 0:
                    # ignore this instance and set gt_mask to a fake mask
                    instance['ignore_flag'] = 1
                    gt_mask = [np.zeros(6)]
            elif not self.poly2mask:
                # `PolygonMasks` requires a ploygon of format List[np.array],
                # other formats are invalid.
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            elif isinstance(gt_mask, dict) and \
                    not (gt_mask.get('counts') is not None and
                         gt_mask.get('size') is not None and
                         isinstance(gt_mask['counts'], (list, str))):
                # if gt_mask is a dict, it should include `counts` and `size`,
                # so that `BitmapMasks` can uncompressed RLE
                instance['ignore_flag'] = 1
                gt_mask = [np.zeros(6)]
            gt_masks.append(gt_mask)
            # re-process gt_ignore_flags
            gt_ignore_flags.append(instance['ignore_flag'])
        results['gt_ignore_flags'] = np.array(gt_ignore_flags, dtype=bool)
        return gt_masks
    
    def _load_masks(self, results: dict) -> None:
        h, w = results['ori_shape']
        gt_masks = self._process_masks(results)
        if self.poly2mask:
            gt_masks = BitmapMasks([self._poly2mask(mask, h, w) 
                                    for mask in gt_masks], h, w)
        else:
            gt_masks = PolygonMasks([mask for mask in gt_masks], h, w)
        results['gt_masks'] = gt_masks
        
    def _load_seg_map(self, results: dict) -> None:
        if results.get('seg_map_path', None) is None:
            return
        
        img_bytes = get(results['seg_map_path'], backend_args=self.backend_args)
        gt_semantic_seg = mmcv.imfrombytes(img_bytes, flag='unchanged', 
                                           backend=self.imdecode_backend).squeeze()
        if self.reduce_zero_label:
            # avoid using underflow conversion
            gt_semantic_seg[gt_semantic_seg == 0] = self.ignore_index
            gt_semantic_seg = gt_semantic_seg - 1
            gt_semantic_seg[gt_semantic_seg == self.ignore_index -
                            1] = self.ignore_index
            
        # modify if custom classes
        if results.get('label_map', None) is not None:
            # Add deep copy to solve bug of repeatedly
            # replace `gt_semantic_seg`, which is reported in
            # https://github.com/open-mmlab/mmsegmentation/pull/1445/
            gt_semantic_seg_copy = gt_semantic_seg.copy()
            for old_id, new_id in results['label_map'].items():
                gt_semantic_seg[gt_semantic_seg_copy == old_id] = new_id
        results['gt_seg_map'] = gt_semantic_seg
        results['ignore_index'] = self.ignore_index
              
    def transform(self, results: dict) -> dict:
        if self.with_bbox:
            self._load_bboxes(results)
        if self.with_label:
            self._load_labels(results)
        if self.with_mask:
            self._load_masks(results)
        if self.with_seg:
            self._load_seg_map(results)
        return results
    
    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(with_bbox={self.with_bbox}, '
        repr_str += f'with_label={self.with_label}, '
        repr_str += f'with_mask={self.with_mask}, '
        repr_str += f'with_seg={self.with_seg}, '
        repr_str += f'poly2mask={self.poly2mask}, '
        repr_str += f"imdecode_backend='{self.imdecode_backend}', "
        repr_str += f'backend_args={self.backend_args})'
        return repr_str