import argparse
import os
import os.path as osp
import warnings
from copy import deepcopy

from mmengine import ConfigDict
from mmengine.config import Config, DictAction
from mmengine.runner import Runner

from mydet.utils import setup_cache_size_limit_of_dynamo
from mydet.engine.hooks.utils import trigger_visualization_hook
from mydet.registry import RUNNERS
from mydet.evaluation import DumpDetResults


def parse_args():
    parser = argparse.ArgumentParser(description="test (eval) phase")
    parser.add_argument('config', help="test config file path")
    parser.add_argument('checkpoint', help="checkpoint file path")
    parser.add_argument('--work-dir', help='saving log path')
    parser.add_argument('--out', type=str, help='预测文件转存到pickle文件')
    parser.add_argument('--show', action='store_true', help='show prediction results')
    parser.add_argument('--show-dir', action='store_true',
        help='directory where painted images will be saved. '
        'If specified, it will be automatically saved '
        'to the work_dir/timestamp/show_dir')
    parser.add_argument('--wait-time', type=float, default=2, 
                        help='the interval of show (s)')
    parser.add_argument('--cfg-options',
        nargs='+',
        action=DictAction,
        help='重写参数，以键值对的形式传入')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    parser.add_argument('--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    args = parse_args()
    
    # TorchDynamo 
    setup_cache_size_limit_of_dynamo()
    
    # Load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # saving path
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join('./work_dirs', osp.splitext(osp.basename(args.config))[0])
    
    cfg.load_from = args.checkpoint
    
    if args.show or args.show_dir:
        # 设置实时展示可视化结果或者存储该结果
        cfg = trigger_visualization_hook(cfg, args)
    
    # test time augmention
    if args.tta:
        if 'tta_model' not in cfg:
            warnings.warn('Cannot find ``tta_model`` in config, '
                          'we will set it as default.')
            cfg.tta_model = dict(
                type='DetTTAModel',
                tta_cfg=dict(
                    nms=dict(type='nms', iou_threshold=0.5), max_per_img=100))
        if 'tta_pipeline' not in cfg:
            warnings.warn('Cannot find ``tta_pipeline`` in config, '
                          'we will set it as default.')
            test_data_cfg = cfg.test_dataloader.dataset
            while 'dataset' in test_data_cfg:
                test_data_cfg = test_data_cfg['dataset']
            cfg.tta_pipeline = deepcopy(test_data_cfg.pipeline)
            flip_tta = dict(
                type='TestTimeAug',
                transforms=[
                    [
                        dict(type='RandomFlip', prob=1.),
                        dict(type='RandomFlip', prob=0.)
                    ],
                    [
                        dict(
                            type='PackDetInputs',
                            meta_keys=('img_id', 'img_path', 'ori_shape',
                                       'img_shape', 'scale_factor', 'flip',
                                       'flip_direction'))
                    ],
                ])
            cfg.tta_pipeline[-1] = flip_tta
        cfg.model = ConfigDict(**cfg.tta_model, module=cfg.model)
        cfg.test_dataloader.dataset.pipeline = cfg.tta_pipeline
        
    # runner
    if 'runner_type' not in cfg:
        # 默认设置
        runner = Runner.from_cfg(cfg)
    else:
        # build customized runner from the registry
        # if 'runner_type' is set in the cfg
        runner = RUNNERS.build(cfg)
        
    if args.out is not None:
        assert args.out.endswith(('.pkl', '.pickle')), \
            'The dump file must be a pkl file.'
        runner.test_evaluator.metrics.append(
            DumpDetResults(out_file_path=args.out))
        
    runner.test()
    
    
if __name__=="__main__":
    main()