%%writefile configs/custom_config.py

custom_imports = dict(imports=['mmpretrain.models'], allow_failed_imports=False)

# =========================================================
# 1. BASE CONFIG
# =========================================================

_base_ = [
    'convnext/cascade-mask-rcnn_convnext-t-p4-w7_fpn_4conv1fc-giou_amp-ms-crop-3x_coco.py'
]

load_from = 'https://download.openmmlab.com/mmdetection/v2.0/convnext/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco/cascade_mask_rcnn_convnext-t_p4_w7_fpn_giou_4conv1f_fp16_ms-crop_3x_coco_20220509_204200-8f07c40b.pth'

checkpoint_file = 'https://download.openmmlab.com/mmclassification/v0/convnext/downstream/convnext-tiny_3rdparty_32xb128-noema_in1k_20220301-795e9634.pth'

# =========================================================
# 2. DATASET & PIPELINES
# =========================================================
image_size = (800, 800)
dataset_type = 'CocoDataset'
data_root = '/kaggle/input/parasite-eggs-2-instance-segmentation/'

classes = (
    "Ascaris lumbricoides", "Capillaria philippinensis", "Enterobius vermicularis",
    "Fasciolopsis buski", "Hookworm egg", "Hymenolepis diminuta",
    "Hymenolepis nana", "Opisthorchis viverrine", "Paragonimus spp",
    "Taenia spp- egg", "Trichuris trichiura"
)
num_classes = len(classes) 

albu_train_transforms = [
    dict(
        type='CLAHE', 
        clip_limit=4.0,
        tile_grid_size=(8, 8),
        p=0.5),
    dict(
        type='RandomRotate90', 
        p=0.5),
    dict(
        type='Rotate',         
        limit=15,
        p=0.5)
]

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=None, to_float32=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_label=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='Resize',
        scale=(800, 800),
        keep_ratio=True  
    ),
    dict(
        type='Albu',
        transforms=albu_train_transforms,
        bbox_params=dict(
            type='BboxParams',
            format='pascal_voc',
            label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
            min_visibility=0.1,
            filter_lost_elements=True),
        keymap={
            'img': 'image',
            'gt_masks': 'masks',
            'gt_bboxes': 'bboxes'
        },
        skip_img_without_anno=True),
    dict(type='PhotoMetricDistortion', brightness_delta=32, contrast_range=(0.8, 1.2), saturation_range=(0.8, 1.2), hue_delta=5),
    dict(type='PackDetInputs'),
]

test_pipeline = [
    dict(type='LoadImageFromFile', to_float32=True, backend_args=None),
    dict(type='Resize', scale=image_size, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True, with_label=True),
    dict(type='PackDetInputs', meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
]

train_dataloader = dict(
    batch_size=4,
    num_workers=2,
    dataset=dict(
        _delete_=True,
        type=dataset_type,
        data_root=data_root,
        ann_file='train/_annotations.coco.json',
        data_prefix=dict(img='train/'),
        metainfo=dict(classes=classes),
        pipeline=train_pipeline,
        filter_cfg=dict(filter_empty_gt=True)
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        data_root=data_root,
        ann_file='valid/_annotations.coco.json',
        data_prefix=dict(img='valid/'),
        metainfo=dict(classes=classes),
        pipeline=test_pipeline,
        filter_cfg=dict(filter_empty_gt=False) 
    )
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    dataset=dict(
        type=dataset_type,
        test_mode=True,
        data_root=data_root,
        ann_file='test/_annotations.coco.json',
        data_prefix=dict(img='test/'),
        metainfo=dict(classes=classes),
        pipeline=test_pipeline,
        filter_cfg=dict(filter_empty_gt=False) 
    )
)

val_evaluator = dict(
    type='CocoMetric', 
    ann_file=data_root + 'valid/_annotations.coco.json', 
    metric=['bbox', 'segm'], 
    classwise=True,
    format_only=False,
    backend_args=None
)

test_evaluator = dict(
    type='CocoMetric', 
    ann_file=data_root + 'test/_annotations.coco.json', 
    metric=['bbox', 'segm'], 
    classwise=True,
    format_only=False,
    backend_args=None
)

# =========================================================
# 3. MODEL SETTINGS 
# =========================================================


model = dict(

    # Convnext
    backbone=dict(
        _delete_=True,
        type='mmpretrain.ConvNeXt',
        arch='tiny',
        # arch='small',
        out_indices=[0, 1, 2, 3],
        # drop_path_rate=0.6,
        drop_path_rate=0.4,

        # ConvNeXt
        layer_scale_init_value=1.0,

        # ConvNeXt-V2
        # layer_scale_init_value=0., 
        # use_grn=True,
        gap_before_final_norm=False,
        init_cfg=dict(
            type='Pretrained', checkpoint=checkpoint_file,
            prefix='backbone.')),
    
    neck=dict(in_channels=[96, 192, 384, 768]),
    
    # Mask R-CNN
    # roi_head=dict(
    #     bbox_head=dict(num_classes=num_classes),
    #     mask_head=dict(num_classes=num_classes)
    # )

    # Cascade Mask R-CNN
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes, 
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes, 
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.05, 0.05, 0.1, 0.1]), 
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes, 
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]), 
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0)),
        ],
        mask_head=dict(
            type='FCNMaskHead',
            num_convs=4,
            in_channels=256,
            conv_out_channels=256,
            num_classes=num_classes,
            loss_mask=dict(
                type='CrossEntropyLoss', use_mask=True, loss_weight=1.0)
        )
    )
)

# =========================================================
# 4. TRAINING LOOP & OPTIMIZER
# =========================================================
max_iters = 13200

train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    max_iters=max_iters,
    val_interval=440,
    dynamic_intervals=[]
)

val_cfg = dict(type='ValLoop')
test_cfg = dict(type='TestLoop')

optim_wrapper = dict(
    type='OptimWrapper',

    constructor='LearningRateDecayOptimizerConstructor',
    optimizer=dict(
        _delete_=True,
        type='AdamW',
        lr=0.0001, 
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)),

    paramwise_cfg={
        'decay_rate': 0.9,
        'decay_type': 'layer_wise',
        'num_layers': 6
    },
    clip_grad=dict(max_norm=1.0, norm_type=2) 
)

param_scheduler = [
    dict(
        type='LinearLR',
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=880
    ),
    dict(
        type='CosineAnnealingLR',
        T_max=max_iters,
        by_epoch=False,
        begin=880,
        end=max_iters,
        eta_min=0,
    )
]

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=440,               
        max_keep_ckpts=3,            
        save_best='coco/segm_mAP',   
        rule='greater'              
    ),
    
    logger=dict(type='LoggerHook', interval=55),
    
    timer=dict(type='IterTimerHook'),
    param_scheduler=dict(type='ParamSchedulerHook'),
    sampler_seed=dict(type='DistSamplerSeedHook'),
    visualization=dict(type='DetVisualizationHook')
)

# Visualization
vis_backends = [
    dict(type='LocalVisBackend'),
    dict(type='WandbVisBackend',
         init_kwargs={
            'project': 'cascade_mask_rcnn_mmdetection',
            'group': 'cascade_mask_rcnn/mmdetection'
         })
]

visualizer = dict(
    type='DetLocalVisualizer',
    vis_backends=vis_backends,
    name='visualizer')

work_dir='/kaggle/working/work_dir'