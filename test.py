if __name__ == '__main__':
    from mmengine.runner import Runner
    
    runner = Runner(
        train_dataloader=dict(
        batch_size=32,
        sampler=dict(
            type='DefaultSampler',
            shuffle=True),
            dataset=dict(
                type='VOCDataset',
                root='/home/jykj/zs/mmdetection/data/VOCdevkit',
                ann_file='VOC2007/ImageSets/Main/trainval.txt',
                data_prefix=dict(sub_data_root='VOC2007/'),
                filter_cfg=dict(filter_empty_gt=True, min_size=32)))
        
    )