"""
PTv3 + PPT
"""

_base_ = ["../_base_/default_runtime.py"]

# hook
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

# path settings
my_data_root = "data/qh_inference"
weight = 'exp/heritage/qh_inference_v1/model/model_best.pth'

# misc custom setting
epoch = 500
eval_epoch = 50
batch_size = 1 # bs: total bs in all gpus
num_worker = 2 # worker processes for data loading
mix_prob = 0.8 # mix3D augmentation probability (https://arxiv.org/pdf/2110.02210)
empty_cache = True # clear GPU cache after each iteration
enable_amp = False # automatic mixed precision (float16/32)
find_unused_parameters = True # pytorch DDP param, find params not used in forward pass
grid_size = 0.1
sphere_point_max = 100000

# copied from base config
resume = True
evaluate = True
test_only = True
seed = 44350923
sync_bn = True

HERITAGE_CATEGORIES = [
    "wall",
    "floor",
    "roof",
    "ceiling",
    "footpath",
    "grass",
    "column",
    "door",
    "window",
    "stair",
    "railing",
    "rainwater pipe",
    "other",
]

# trainer
train = dict(
    type="LoRATrainer",
)

# model settings
model = dict(
    type="PPT-LoRA",
    rank=64,
    lora_alpha=48,
    lora_dropout_p=0.1,
    new_conditions = ["Heritage"],
    condition_mapping = {"Heritage": "ScanNet"},
    device = "cuda",
)

optimizer=dict(
    type="AdamW",
    weight_decay=0.1,
    lr=0.002,
    betas=(0.9, 0.999)
)

# scheduler settings
scheduler = dict(
    type="OneCycleLR",
    max_lr=0.003,
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)
param_dicts = [dict(keyword="block", lr=0.0005)]

# dataset settings
data = dict(
    num_classes=len(HERITAGE_CATEGORIES),
    ignore_index=-1,
    names=HERITAGE_CATEGORIES,
    train=dict(
        type="LibraryDataset",
        split="train",
        data_root=my_data_root,  # Optional, will use default if not specified
        glob_pattern="combined*.pth",
        transform=[
            dict(type='CenterShift', apply_z=True),
            dict(
                type='RandomRotate',
                angle=[-1, 1],
                axis='z',
                center=[0, 0, 0],
                p=0.25),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='x',
                p=0.25),
            dict(
                type='RandomRotate',
                angle=[-0.015625, 0.015625],
                axis='y',
                p=0.25),
            dict(type='RandomScale', scale=[0.9, 1.1]),
            dict(type='RandomFlip', p=0.5),
            dict(type='RandomJitter', sigma=0.005, clip=0.02),
            # dict(
            #     type='ElasticDistortion',
            #     distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type='ChromaticJitter', p=0.95, std=0.05),
            dict(
                type='GridSample',
                grid_size=grid_size,
                hash_type='fnv',
                mode='train',
                return_grid_coord=True
            ),
            dict(type='SphereCrop', point_max=sphere_point_max, mode='random'),
            dict(type='CenterShift', apply_z=False),
            dict(type='NormalizeColor'),
            dict(type='ShufflePoint'),
            dict(type='ToTensor'),
            dict(type='Collect', keys=('coord', 'grid_coord', 'segment', 'condition'), feat_keys=('color', 'normal'))
        ],
        test_mode=False,
    ),
    val=dict(
        type="LibraryDataset",
        split="val",
        data_root=my_data_root,
        glob_pattern="combined*.pth",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", grid_size=grid_size, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="SphereCrop", point_max=sphere_point_max, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment", "condition"), feat_keys=("color", "normal"))
        ],
        test_mode=False,
        loop=8,
    ),
    test=dict(
        type="LibraryDataset",
        split="test",
        data_root=my_data_root,
        glob_pattern="combined*.pth",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(
                type="GridSample",
                grid_size=grid_size,
                hash_type="fnv",
                mode="test",
                keys=("coord", "color", "normal"),
                return_grid_coord=True
            ),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index", "condition"), feat_keys=("color", "normal"))
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis='z', center=[0, 0, 0], p=1)],
                # [dict(type="RandomRotateTargetAngle", angle=[1/2], axis='z', center=[0, 0, 0], p=1)],
                # [dict(type="RandomRotateTargetAngle", angle=[1], axis='z', center=[0, 0, 0], p=1)],
                # [dict(type="RandomRotateTargetAngle", angle=[3/2], axis='z', center=[0, 0, 0], p=1)],
                # [dict(type="RandomFlip", p=1)]
            ]
        )
    )
)
