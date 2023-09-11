import warnings
from typing import Sequence

from mmengine.evaluator import DumpResults
from mmengine.evaluator.metric import _to_cpu

from mydet.registry import METRICS
from mydet.structures.mask import encode_mask_results

@METRICS.register_module()
class DumpDetResults(DumpResults):
    
    def process(self, data_batch: dict, 
                data_samples: Sequence[dict]) -> None:
        data_samples = _to_cpu(data_samples)
        for data_sample in data_samples:
            data_sample.pop('gt_instances', None)
            data_sample.pop('ignored_instances', None)
            data_sample.pop('gt_panoptic_seg', None)
            
            if 'pred_instances' in data_sample:
                pred = data_sample['pred_instances']
                if 'masks' in pred:
                    pred['masks'] = encode_mask_results(pred['masks'].numpy())
            if 'pred_panoptic_seg' in data_sample:
                warnings.warn(
                    'Panoptic segmentation map will not be compressed. '
                    'The dumped file will be extremely large! '
                    'Suggest using `CocoPanopticMetric` to save the coco '
                    'format json and segmentation png files directly.')
        self.results.extend(data_samples)