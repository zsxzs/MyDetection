import os.path as osp
from typing import List, Optional

from mmengine.dataset import BaseDataset
from mmengine.fileio import load
from mmengine.utils import is_abs

from ..registry import DATASETS

@DATASETS.register_module()
class BaseDetDataset(BaseDataset):
    """
    检测任务的数据集基类
    """
    def __init(self, 
               *args,
               seg_map_suffix: str='.png',
               proposal_file: Optional[str]=None,
               file_client_args: dict=None,
               backend_args: dict=None,
               return_classes: bool=False,
               **kwargs) -> None:
        self.seg_map_suffix = seg_map_suffix
        self.proposal_file = proposal_file
        self.backend_args = backend_args
        self.return_classes = return_classes
        if file_client_args is not None:
            raise RuntimeError(
                'The `file_client_args` is deprecated, '
                'please use `backend_args` instead, please refer to'
                'https://github.com/open-mmlab/mmdetection/blob/main/configs/_base_/datasets/coco_detection.py'  # noqa: E501
            )
        super().__init__(*args, **kwargs)
        
    # override full_init()
    def full_init(self) -> None:
        
        if self._fully_initialized:
            return
        
        # 1. load data list
        self.data_list = self.load_data_list()
        # load proposals
        if self.proposal_file is not None:
            self.load_proposals() 
        # 2. filter data (filter illegal data, such as data that has no annotations.)
        self.data_list = self.filter_data()
        # 3. get subset (optional)
        if self._indices is not None:
            self.data_list = self._get_unserialized_subset(self._indices)
        # 4. serialize data (optional)
        # 默认操作为序列化全部样本。
        if self.serialize_data:
            self.data_bytes, self.data_address = self._serialize_data()
        
        self._fully_initialized = True
            
    def load_proposals(self) -> None:
        """
        加载proposals
        
        The `proposals_list` should be a dict[img_path: proposals]
        with the same length as `data_list`. And the `proposals` should be
        a `dict` or :obj:`InstanceData` usually contains following keys.

            - bboxes (np.ndarry): Has a shape (num_instances, 4),
              the last dimension 4 arrange as (x1, y1, x2, y2).
            - scores (np.ndarry): Classification scores, has a shape
              (num_instance, ).
        """
        
        if not is_abs(self.proposal_file):
            self.proposal_file = osp.join(self.data_root, self.proposal_file)
        proposals_list = load(self.proposal_file, backend_args=self.backend_args)
        assert len(self.data_list) == len(proposals_list)
        for data_info in self.data_list:
            img_path = data_info['img_path']
            file_name = osp.join(
                osp.split(osp.split(img_path)[0])[-1],
                osp.split(img_path)[-1])
            proposals = proposals_list[file_name]
            data_info['proposals'] = proposals
    
    def get_cat_ids(self, idx: int) -> List[int]:
        """
        返回指定idx样本中包含的所有实例id
        """
        instances = self.get_data_info(idx)['instances']
        return [instance['bbox_label'] for instance in instances]
        