from typing import Iterator, List, Sequence

from torch.utils.data import BatchSampler, Sampler

# from mydet.datasets.samplers.track_img_sampler import TrackImgSampler
from mydet.registry import DATA_SAMPLERS

@DATA_SAMPLERS.register_module()
class AspectRatioBatchSampler(BatchSampler):
    """
        按照相同的长宽比生成一个batch的数据id
    """
    def __init__(self,
                 sampler: Sampler,
                 batch_size: int,
                 drop_last: bool = False) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError('batch_size should be a positive integer value, '
                             f'but got batch_size={batch_size}')
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last
        # two groups for w < h and w >= h
        self._aspect_ratio_buckets = [[] for _ in range(2)]
        
    def __iter__(self) -> Iterator[List[int]]:
        for idx in self.sampler:
            data_info = self.sampler.dataset.get_data_info(idx)
            width, height = data_info['width'], data_info['height']
            bucket_id = 0 if width < height else 1
            bucket = self._aspect_ratio_buckets[bucket_id]
            bucket.append(idx)
            
            if len(bucket) == self.batch_size:
                # 生成一个batch索引
                yield bucket[:]
                del bucket[:]
        
        # 剩余数据
        left_data = self._aspect_ratio_buckets[0] + \
                                self._aspect_ratio_buckets[1]
        self._aspect_ratio_buckets = [[] for _ in range(2)]
        while len(left_data) > 0:
            if len(left_data) <= self.batch_size:
                if not self.drop_last:
                    yield left_data[:]
                left_data = []
            else:
                yield left_data[:self.batch_size]
                left_data = left_data[self.batch_size:]
        
    def __len__(self) -> int:
        if self.drop_last:
            return len(self.sampler) // self.batch_size
        else:
            return (len(self.sampler) + self.batch_size - 1) // self.batch_size