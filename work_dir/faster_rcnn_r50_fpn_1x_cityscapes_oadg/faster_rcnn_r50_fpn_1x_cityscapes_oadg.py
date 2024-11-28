model = dict(
    type='FasterRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=None),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='RPNHead',
        in_channels=256,
        feat_channels=256,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0]),
        loss_cls=dict(
            type='CrossEntropyLossPlus',
            use_sigmoid=True,
            loss_weight=1.0,
            num_views=2,
            additional_loss='jsdv1_3_2aug',
            lambda_weight=0.1,
            wandb_name='rpn_cls'),
        loss_bbox=dict(
            type='L1LossPlus',
            loss_weight=1.0,
            num_views=2,
            additional_loss='None',
            lambda_weight=0.0,
            wandb_name='rpn_bbox')),
    roi_head=dict(
        type='ContrastiveRoIHead',
        bbox_roi_extractor=dict(
            type='SingleRoIExtractor',
            roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='Shared2FCContrastiveHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=6,
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[0.0, 0.0, 0.0, 0.0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            reg_class_agnostic=False,
            loss_cls=dict(
                type='CrossEntropyLossPlus',
                use_sigmoid=False,
                loss_weight=1.0,
                num_views=2,
                additional_loss='jsdv1_3_2aug',
                lambda_weight=10,
                wandb_name='roi_cls',
                log_pos_ratio=True),
            loss_bbox=dict(
                type='SmoothL1LossPlus',
                loss_weight=1.0,
                beta=1.0,
                num_views=2,
                additional_loss='None',
                lambda_weight=0.0,
                wandb_name='roi_bbox'),
            with_cont=True,
            cont_predictor_cfg=dict(
                num_linear=2, feat_channels=256, return_relu=True),
            out_dim_cont=256,
            loss_cont=dict(
                type='ContrastiveLossPlus',
                loss_weight=0.01,
                num_views=2,
                temperature=0.06))),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=-1,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False,
            dropout=False),
        wandb=dict(
            layer_list=[], log=dict(features_list=[], vars=['log_vars'])),
        random_proposal_cfg=dict(
            bbox_from='oagrb',
            num_bboxes=10,
            scales=(0.01, 0.3),
            ratios=(0.3, 3.3333333333333335),
            iou_max=0.7,
            iou_min=0.0)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=1000,
            max_per_img=1000,
            nms=dict(type='nms', iou_threshold=0.7),
            min_bbox_size=0),
        rcnn=dict(
            score_thr=0.05,
            nms=dict(type='nms', iou_threshold=0.5),
            max_per_img=100)))
dataset_type = 'CityscapesDataset'
data_root = '/home/intern/minkyoung/dataset/cityscapes/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(
        type='OAMix',
        version='augmix',
        num_views=2,
        keep_orig=True,
        severity=10,
        random_box_ratio=(3, 0.3333333333333333),
        random_box_scale=(0.01, 0.1),
        oa_random_box_scale=(0.005, 0.1),
        oa_random_box_ratio=(3, 0.3333333333333333),
        spatial_ratio=4,
        sigma_ratio=0.3),
    dict(
        type='Normalize',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        to_rgb=True),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(
        type='Collect',
        keys=[
            'img', 'img2', 'gt_bboxes', 'gt_bboxes2', 'gt_labels',
            'multilevel_boxes', 'oamix_boxes'
        ])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(2048, 1024),
        flip=False,
        transforms=[
            dict(type='Resize', img_scale=(1024, 512), keep_ratio=True),
            dict(type='RandomFlip'),
            dict(
                type='Normalize',
                mean=[123.675, 116.28, 103.53],
                std=[58.395, 57.12, 57.375],
                to_rgb=True),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img'])
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=8,
        dataset=dict(
            type='CityscapesDataset',
            ann_file=
            '/home/intern/minkyoung/dataset/cityscapes/annotations/instancesonly_filtered_gtFine_train.json',
            img_prefix=
            '/home/intern/minkyoung/dataset/cityscapes/leftImg8bit/train/',
            pipeline=[
                dict(type='LoadImageFromFile'),
                dict(type='LoadAnnotations', with_bbox=True),
                dict(
                    type='Resize',
                    img_scale=[(2048, 800), (2048, 1024)],
                    keep_ratio=True),
                dict(type='RandomFlip', flip_ratio=0.5),
                dict(
                    type='OAMix',
                    version='augmix',
                    num_views=2,
                    keep_orig=True,
                    severity=10,
                    random_box_ratio=(3, 0.3333333333333333),
                    random_box_scale=(0.01, 0.1),
                    oa_random_box_scale=(0.005, 0.1),
                    oa_random_box_ratio=(3, 0.3333333333333333),
                    spatial_ratio=4,
                    sigma_ratio=0.3),
                dict(
                    type='Normalize',
                    mean=[123.675, 116.28, 103.53],
                    std=[58.395, 57.12, 57.375],
                    to_rgb=True),
                dict(type='Pad', size_divisor=32),
                dict(type='DefaultFormatBundle'),
                dict(
                    type='Collect',
                    keys=[
                        'img', 'img2', 'gt_bboxes', 'gt_bboxes2', 'gt_labels',
                        'multilevel_boxes', 'oamix_boxes'
                    ])
            ])),
    val=dict(
        type='CityscapesDataset',
        ann_file=
        '/home/intern/minkyoung/dataset/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix='/home/intern/minkyoung/dataset/cityscapes/leftImg8bit/val/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1024, 512), keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]),
    test=dict(
        type='CityscapesDataset',
        ann_file=
        '/home/intern/minkyoung/dataset/cityscapes/annotations/instancesonly_filtered_gtFine_val.json',
        img_prefix='/home/intern/minkyoung/dataset/cityscapes-c/',
        pipeline=[
            dict(type='LoadImageFromFile'),
            #dict(type='LoadAnnotations', with_bbox=True),
            dict(
                type='MultiScaleFlipAug',
                img_scale=(2048, 1024),
                flip=False,
                transforms=[
                    dict(
                        type='Resize', img_scale=(1024, 512), keep_ratio=True),
                    dict(type='RandomFlip'),
                    dict(
                        type='Normalize',
                        mean=[123.675, 116.28, 103.53],
                        std=[58.395, 57.12, 57.375],
                        to_rgb=True),
                    dict(type='Pad', size_divisor=32),
                    dict(type='DefaultFormatBundle'),
                    dict(type='Collect', keys=['img'])
                ])
        ]))
evaluation = dict(interval=1, metric='bbox')
checkpoint_config = dict(interval=1)
log_config = dict(interval=100, hooks=[dict(type='TextLoggerHook')])
custom_hooks = []
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/faster_rcnn_r50_fpn_1x_coco/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
resume_from = '/home/intern/minkyoung/OA-DG/work_dir/faster_rcnn_r50_fpn_1x_cityscapes_oadg/latest.pth'
workflow = [('train', 1)]
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(grad_clip=None)
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=0.001,
    step=[1])
runner = dict(type='EpochBasedRunner', max_epochs=2)
num_views = 2
lw_jsd_rpn = 0.1
lw_jsd_roi = 10
lw_cont = 0.01
temperature = 0.06
random_proposal_cfg = dict(
    bbox_from='oagrb',
    num_bboxes=10,
    scales=(0.01, 0.3),
    ratios=(0.3, 3.3333333333333335),
    iou_max=0.7,
    iou_min=0.0)
oamix_config = dict(
    type='OAMix',
    version='augmix',
    num_views=2,
    keep_orig=True,
    severity=10,
    random_box_ratio=(3, 0.3333333333333333),
    random_box_scale=(0.01, 0.1),
    oa_random_box_scale=(0.005, 0.1),
    oa_random_box_ratio=(3, 0.3333333333333333),
    spatial_ratio=4,
    sigma_ratio=0.3)
custom_imports = dict(
    imports=['mmdet.datasets.pipelines.oa_mix'], allow_failed_imports=False)
work_dir = '/home/intern/minkyoung/OA-DG/work_dir/faster_rcnn_r50_fpn_1x_cityscapes_oadg'
auto_resume = False
gpu_ids = [1]
