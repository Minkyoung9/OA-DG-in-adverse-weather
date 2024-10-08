_base_ = '/home/intern/minkyoung/OA-DG/configs/_base_/default_runtime.py'

# YOLOv5 model settings
model = dict(
    type='YOLOV5',
    model=dict(
        type='YOLOV5',
        depth=0.67,  # adjust as needed
        width=0.75,  # adjust as needed
        pretrained='./yolov5m.pt',  # pretrained YOLOv5-m model
        num_classes=8,  # adjust the number of classes based on your dataset
    )
)

# Dataset settings
dataset_type = 'CocoDataset'
data_root = 'dataset/COCO/'
img_norm_cfg = dict(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0], to_rgb=True)

train_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize',
        img_scale=[(640, 640), (1280, 1280)],  # YOLOv5 default scales
        keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(640, 640),  # Test scale
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_train2017.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline))

# Optimizer settings
optimizer = dict(type='SGD', lr=0.01, momentum=0.937, weight_decay=0.0005)

# Learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=1000,  # number of iterations for warmup
    warmup_ratio=0.1,
    step=[160, 240])  # adjust the step values based on your training schedule

# Runtime settings
runner = dict(type='EpochBasedRunner', max_epochs=300)  # Set your max epochs
evaluation = dict(interval=1, metric=['bbox'])
