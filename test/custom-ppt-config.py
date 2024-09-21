###############################################################################################
# Data directory config.
###############################################################################################
my_data_root = "./data/qh/training_5cm_voxels_2x2x2"
# my_data_root = "./data/qh/training_slice_voxels_4x4x3"
save_path = "exp/qh/training_zeroshot_curveball/semseg-pt-v3m1-1-ppt-extreme"  # WARNING: if this dir already exists, Pointcept will fail with a very esoteric error!

my_categories = [
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
    "rain water pipe",
    "OTHER",
]

###############################################################################################
# Job config
###############################################################################################
resume = False
evaluate = False
test_only = True
seed = 44350923
num_worker = 48
batch_size = 24
batch_size_val = None
batch_size_test = None
sync_bn = False
enable_amp = True
empty_cache = False
find_unused_parameters = True
mix_prob = 0.8
param_dicts = [dict(keyword="block", lr=0.0005)]
hooks = [
    dict(type="CheckpointLoader"),
    dict(type="IterationTimer", warmup_iter=2),
    dict(type="InformationWriter"),
    dict(type="SemSegEvaluator"),
    dict(type="CheckpointSaver", save_freq=None),
    dict(type="PreciseEvaluator", test_last=False),
]

###############################################################################################
# Model config
###############################################################################################
weight = "models/PointTransformerV3/scannet-semseg-pt-v3m1-1-ppt-extreme/model/model_best.pth"

# Point Transformer
pdnorm_config = dict(
    use_pdnorm=True,
    bn=True,
    ln=True,
    decouple=True,
    adaptive=False,
    affine=True,
    eps=1e-3,
    momentum=0.01,
)

backbone_config = dict(
    type="PT-v3m2",
    in_channels=6,
    order=("z", "z-trans", "hilbert", "hilbert-trans"),
    stride=(2, 2, 2, 2),
    enc_depths=(3, 3, 3, 6, 3),
    enc_channels=(48, 96, 192, 384, 512),
    enc_num_head=(3, 6, 12, 24, 32),
    enc_patch_size=(1024, 1024, 1024, 1024, 1024),
    dec_depths=(3, 3, 3, 3),
    dec_channels=(64, 96, 192, 384),
    dec_num_head=(4, 6, 12, 24),
    dec_patch_size=(1024, 1024, 1024, 1024),
    mlp_ratio=4,
    qkv_bias=True,
    qk_scale=None,
    attn_drop=0.0,
    proj_drop=0.0,
    drop_path=0.3,
    shuffle_orders=True,
    pre_norm=True,
    enable_rpe=False,
    enable_flash=True,
    upcast_attention=False,
    upcast_softmax=False,
    cls_mode=False,
)

# Point prompt training
model = dict(
    type="PPT-v1m3",
    backbone=backbone_config,
    pdnorm=pdnorm_config,
    criteria=[
        dict(type="CrossEntropyLoss", loss_weight=1.0, ignore_index=-1),
        dict(type="LovaszLoss", mode="multiclass", loss_weight=1.0, ignore_index=-1),
    ],
    backbone_out_channels=64,
    context_channels=256,
    template="[x]",
    categories=my_categories,
    clip_model="ViT-B/16",
    backbone_mode=False,
)

###############################################################################################
# Optimiser and Scheduler config
###############################################################################################
optimizer = dict(type="AdamW", lr=0.005, weight_decay=0.05)
scheduler = dict(
    type="OneCycleLR",
    max_lr=[0.005, 0.0005],
    pct_start=0.05,
    anneal_strategy="cos",
    div_factor=10.0,
    final_div_factor=1000.0,
)

