

"""
PTv3 + PPT
"""

_base_ = ["../_base_/default_runtime.py"]

# misc custom setting
epoch = 100
eval_epoch = 100
# batch_size = 16  # bs: total bs in all gpus
batch_size = 1
num_worker = 6 # worker processes for data loading
mix_prob = 0.8 # mix3D augmentation probability (https://arxiv.org/pdf/2110.02210)
empty_cache = False # clear GPU cache after each iteration
enable_amp = True # automatic mixed precision (float16/32)
find_unused_parameters = True # pytorch DDP param, find params not used in forward pass

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
    rank=10,
    lora_alpha=20,
    lora_dropout_p=0.0,
    new_conditions = ["Heritage"],
    condition_mapping = {"Heritage": "ScanNet"},
    device = "cuda",
)

optimizer=dict(
    type="AdamW",
    weight_decay=0.05,
    lr=0.005,
    betas=(0.9, 0.999)
)

# scheduler settings
epoch = 100
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.005, 0.0005],
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
        data_root="data/clouds/res0.02_pr0.019/library",  # Optional, will use default if not specified
        glob_pattern="combined*.pth",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="RandomDropout", dropout_ratio=0.2, dropout_application_ratio=0.2),
            dict(type="RandomRotate", angle=[-1, 1], axis='z', center=[0, 0, 0], p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis='x', p=0.5),
            dict(type="RandomRotate", angle=[-1/64, 1/64], axis='y', p=0.5),
            dict(type="RandomScale", scale=[0.9, 1.1]),
            dict(type="RandomFlip", p=0.5),
            dict(type="RandomJitter", sigma=0.005, clip=0.02),
            dict(type="ElasticDistortion", distortion_params=[[0.2, 0.4], [0.8, 1.6]]),
            dict(type="ChromaticAutoContrast", p=0.2, blend_factor=None),
            dict(type="ChromaticTranslation", p=0.95, ratio=0.05),
            dict(type="ChromaticJitter", p=0.95, std=0.05),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="SphereCrop", point_max=100000, mode="random"),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ShufflePoint"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("color", "normal"))
        ],
        test_mode=False,
        loop=1  # Adjust if needed
    ),
    val=dict(
        type="LibraryDataset",
        split="val",
        data_root="data/clouds/res0.02_pr0.019/library",
        glob_pattern="combined*.pth",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="train", return_grid_coord=True),
            dict(type="CenterShift", apply_z=False),
            dict(type="NormalizeColor"),
            dict(type="ToTensor"),
            dict(type="Collect", keys=("coord", "grid_coord", "segment"), feat_keys=("color", "normal"))
        ],
        test_mode=False,
        loop=1
    ),
    test=dict(
        type="LibraryDataset",
        split="test",
        data_root="data/clouds/res0.02_pr0.019/library",
        glob_pattern="combined*.pth",
        transform=[
            dict(type="CenterShift", apply_z=True),
            dict(type="NormalizeColor"),
        ],
        test_mode=True,
        test_cfg=dict(
            voxelize=dict(type="GridSample", grid_size=0.02, hash_type="fnv", mode="test", keys=("coord", "color", "normal"), return_grid_coord=True),
            crop=None,
            post_transform=[
                dict(type="CenterShift", apply_z=False),
                dict(type="ToTensor"),
                dict(type="Collect", keys=("coord", "grid_coord", "index"), feat_keys=("color", "normal"))
            ],
            aug_transform=[
                [dict(type="RandomRotateTargetAngle", angle=[0], axis='z', center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1/2], axis='z', center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[1], axis='z', center=[0, 0, 0], p=1)],
                [dict(type="RandomRotateTargetAngle", angle=[3/2], axis='z', center=[0, 0, 0], p=1)],
                [dict(type="RandomFlip", p=1)]
            ]
        )
    )
)
