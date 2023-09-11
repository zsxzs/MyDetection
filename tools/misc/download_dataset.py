import argparse
import tarfile
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from tarfile import TarFile
from zipfile import ZipFile

import torch
# from mmengine.utils.path import mkdir_or_exist

def parse_args():
    parser = argparse.ArgumentParser(description="下载训练数据集")
    parser.add_argument('--dataset-name', type=str, default='coco2017', help='dataset name')
    parser.add_argument('--save-dir', type=str, default='data/coco', help='saving dirname')
    parser.add_argument('--unzip', action='store_true', 
                            help='whether unzip dataset or not, zipped files will be saved')
    parser.add_argument('--delete', action='store_true', help='delete the download zipped files')
    parser.add_argument('--threads', type=int, default=4, help='number of threading')
    args = parser.parse_args()
    return args

def download(url, dir, unzip=True, delete=False, threads=1):
    def download_one(url, dir):
        f = dir / Path(url).name
        if Path(url).is_file():
            Path(url).rename(f)
        elif not f.exists():
            print(f'Downloading {url} to {f}')
            torch.hub.download_url_to_file(url, f, progress=True)
        if unzip and f.suffix in ('.zip', '.tar'):
            print(f'Unzipping {f.name}')
            if f.suffix == '.zip':
                ZipFile(f).extractall(path=dir)
            elif f.suffix(f) == '.tar':
                TarFile(f).extractall(path=dir)
            if delete:
                f.unlink()
                print(f'Delete {f}')
    
    dir = Path(dir)
    if threads > 1:
        pool = ThreadPool(threads)
        pool.imap(lambda x: download_one(*x), zip(url, repeat(dir)))
        pool.close()
        pool.join()
    else:
        for u in [url] if isinstance(url, (str, Path)) else url:
            download_one(u, dir)
       
            
# TODO: download_objects365
def download_objects365v2(url, dir, unzip=True, delete=False, threads=1):
    pass


def main():
    args = parse_args()
    path = Path(args.save_dir)
    if not path.exists():
        path.mkdir(parents=True, exist_ok=True)
    data2url = dict(
        coco2017=[
            'http://images.cocodataset.org/zips/train2017.zip',
            'http://images.cocodataset.org/zips/val2017.zip',
            'http://images.cocodataset.org/zips/test2017.zip',
            'http://images.cocodataset.org/zips/unlabeled2017.zip',
            'http://images.cocodataset.org/annotations/annotations_trainval2017.zip',  # noqa
            'http://images.cocodataset.org/annotations/stuff_annotations_trainval2017.zip',  # noqa
            'http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip',  # noqa
            'http://images.cocodataset.org/annotations/image_info_test2017.zip',  # noqa
            'http://images.cocodataset.org/annotations/image_info_unlabeled2017.zip',  # noqa
        ],
        coco2014=[
            'http://images.cocodataset.org/zips/train2014.zip',
            'http://images.cocodataset.org/zips/val2014.zip',
            'http://images.cocodataset.org/zips/test2014.zip',
            'http://images.cocodataset.org/annotations/annotations_trainval2014.zip',  # noqa
            'http://images.cocodataset.org/annotations/image_info_test2014.zip'  # noqa
        ],
        lvis=[
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip',  # noqa
            'https://s3-us-west-2.amazonaws.com/dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip',  # noqa
        ],
        voc2007=[
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtrainval_06-Nov-2007.tar',  # noqa
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar',  # noqa
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCdevkit_08-Jun-2007.tar',  # noqa
        ],
        voc2012=[
            'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar',  # noqa
        ],
        balloon=[
            # src link: https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip # noqa
            'https://download.openmmlab.com/mmyolo/data/balloon_dataset.zip'
        ],
        objects365v2=[
            # training annotations
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/zhiyuan_objv2_train.tar.gz',  # noqa
            # validation annotations
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/zhiyuan_objv2_val.json',  # noqa
            # training url root
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/train/',  # noqa
            # validation url root_1
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v1/',  # noqa
            # validation url root_2
            'https://dorc.ks3-cn-beijing.ksyun.com/data-set/2020Objects365%E6%95%B0%E6%8D%AE%E9%9B%86/val/images/v2/'  # noqa
        ],
        ade20k_2016=[
            # training images and semantic segmentation annotations
            'http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip',  # noqa
            # instance segmentation annotations
            'http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar',  # noqa
            # img categories ids
            'https://raw.githubusercontent.com/CSAILVision/placeschallenge/master/instancesegmentation/imgCatIds.json',  # noqa
            # category mapping
            'https://raw.githubusercontent.com/CSAILVision/placeschallenge/master/instancesegmentation/categoryMapping.txt'  # noqa
        ],
        refcoco=[
            # images
            'http://images.cocodataset.org/zips/train2014.zip',
            # refcoco annotations
            'https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip',
            # refcoco+ annotations
            'https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco+.zip',
            # refcocog annotations
            'https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcocog.zip'
        ])
    url = data2url.get(args.dataset_name, None)
    if url is None:
        print('Only support ADE20K, COCO, RefCOCO, VOC, LVIS, '
              'balloon, and Objects365v2 now!')
        return
    if args.dataset_name == 'objects365v2':
        download_objects365v2(url, dir=path, unzip=args.unzip,
                              delete=args.delete, threads=args.threads)
    else:
        download(url, dir=path, unzip=args.unzip,
                 delete=args.delete, threads=args.threads)
        
    
if __name__ == '__main__':
    main()