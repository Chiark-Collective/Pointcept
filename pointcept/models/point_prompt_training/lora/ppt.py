import logging
import typing as ty
import os
from collections import OrderedDict
from functools import partial, wraps
from pathlib import Path
from ordered_set import OrderedSet

import torch
import torch.optim
import torch.nn as nn
import minlora
import torch
import gc
from pointcept.models.point_prompt_training import PDNorm
from minlora import LoRAParametrization
from spconv.pytorch.conv import SubMConv3d

from pointcept.engines.defaults import (
    create_ddp_model,
    default_argument_parser,
    default_config_parser,
)
import pointcept.utils.comm as comm
# from pointcept.models import build_model
from pointcept.models.builder import MODELS
from pointcept.utils.registry import build_from_cfg
from pointcept.utils.env import get_random_seed
from pointcept.utils.optimizer import OPTIMIZERS
from pointcept.utils.config import Config
from .utils import (
    WeightFreezer,
    patch_cfg,
    configure_adamw_lora,
    count_trainable_parameters,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
# TRAINED_PPT_BASE_CONFIG = REPO_ROOT / "test/custom-ppt-config.py" 
TRAINED_PPT_BASE_CONFIG = Path("./test/custom-ppt-config.py") 


def build_model(cfg):
    model = build_from_cfg(cfg.model, registry=MODELS)
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Num params: {n_parameters}")
    model = create_ddp_model(
        model.cuda(),
        broadcast_buffers=False,
        find_unused_parameters=cfg.find_unused_parameters,
    )
    if os.path.isfile(cfg.weight):
        logger.info(f"Loading weight at: {cfg.weight}")
        checkpoint = torch.load(cfg.weight)
        weight = OrderedDict()
        for key, value in checkpoint["state_dict"].items():
            if key.startswith("module."):
                if comm.get_world_size() == 1:
                    key = key[7:]  # module.xxx.xxx -> xxx.xxx
            else:
                if comm.get_world_size() > 1:
                    key = "module." + key  # xxx.xxx -> module.xxx.xxx
            weight[key] = value
        # for k in weight:
        #     print(k)
        # print("DITCHING CLASS EMBEDDING")
        weight.pop("class_embedding")
        model.load_state_dict(weight, strict=False)
        logger.info(
            "=> Loaded weight '{}' (epoch {})".format(
                cfg.weight, checkpoint["epoch"]
            )
        )
    else:
        raise RuntimeError("=> No checkpoint found at '{}'".format(cfg.weight))
    return model


def load_base_model(
    cfg_file: Path = TRAINED_PPT_BASE_CONFIG,
    repo_root: Path = REPO_ROOT,
    device: str = "cuda"
) -> nn.Module:
    """load trained PPT model weights from config for application of LoRA / pdnorm expansion"""
    assert cfg_file.exists
    args = default_argument_parser().parse_args(args=["--config-file", f"{cfg_file}"])
    # this patching thing not stricty necessary, doesn't matter though because we throw everything away except
    # the model 
    # print(TRAINED_PPT_BASE_CONFIG)
    # raise Exception
    cfg = Config.fromfile(args.config_file, args.options)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    if cfg.seed is None:
        cfg.seed = get_random_seed()

    # cfg.data.train.loop = cfg.epoch // cfg.eval_epoch

    os.makedirs(os.path.join(cfg.save_path, "model"), exist_ok=True)
    if not cfg.resume:
        cfg.dump(os.path.join(cfg.save_path, "config.py"))
    # cfg = patch_cfg(cfg, repo_root=repo_root)
    # print(cfg)
    # raise Exception
    # cfg = default_config_parser(args.config_file, args.options); cfg = patch_cfg(cfg, repo_root=repo_root)
    # tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    # model = tester.model
    try:
        model = build_model(cfg)
    except Exception as e:
        logger.error(e)
        logger.warning(cfg.keys())
        raise RuntimeError
    model.to(device)
    return model


def expand_ppt_model_conditions(
    model: nn.Module,
    new_conditions: list[str],
    condition_mapping: dict[str, str | None] | None = None
) -> nn.Module:
    """
    Expands a trained PPT model to handle new conditions (datasets). The appropriate 
    normalisation layers are either copied from the trained norm layers corresponding 
    to existing datasets or are initialised randomly (as specified by condition_mapping).
    
    Args:
    - model: The trained PPT model
    - new_conditions: List of new condition names to add
    - condition_mapping: dict mapping new conditions to existing ones for weight initialisation
    
    Returns:
    - Updated model with expanded normalisation layers
    """
    if condition_mapping is None:
        condition_mapping = {}
    for condition in new_conditions:
        if condition not in condition_mapping:
            condition_mapping[condition] = None

    original_conditions = model.conditions
    model.conditions = tuple(OrderedSet(list(original_conditions) + new_conditions))

    def expand_pdnorm(pdnorm):
        if isinstance(pdnorm, PDNorm) and pdnorm.decouple:
            first_norm = pdnorm.norm[0]
            device = first_norm.weight.device  # Get the device of the original norm layer
            if isinstance(first_norm, nn.BatchNorm1d):
                new_norm_func = lambda: type(first_norm)(
                    first_norm.num_features,
                    eps=first_norm.eps,
                    momentum=first_norm.momentum,
                    affine=first_norm.affine,
                    track_running_stats=first_norm.track_running_stats
                ).to(device)
            elif isinstance(first_norm, nn.LayerNorm):
                new_norm_func = lambda: type(first_norm)(
                    first_norm.normalized_shape,
                    eps=first_norm.eps,
                    elementwise_affine=first_norm.elementwise_affine
                ).to(device)
            else:
                raise ValueError(f"Unsupported normalization type: {type(first_norm)}")

            new_norms = [new_norm_func() for _ in new_conditions]
            
            for i, condition in enumerate(new_conditions):
                if condition_mapping[condition] in original_conditions:
                    source_idx = original_conditions.index(condition_mapping[condition])
                    new_norms[i].weight.data.copy_(pdnorm.norm[source_idx].weight.data)
                    new_norms[i].bias.data.copy_(pdnorm.norm[source_idx].bias.data)
                    if isinstance(new_norms[i], nn.BatchNorm1d):
                        new_norms[i].running_mean.copy_(pdnorm.norm[source_idx].running_mean)
                        new_norms[i].running_var.copy_(pdnorm.norm[source_idx].running_var)
                else:
                    # Initialize with random values
                    nn.init.normal_(new_norms[i].weight, mean=1.0, std=0.02)
                    nn.init.zeros_(new_norms[i].bias)
            
            pdnorm.norm.extend(new_norms)
            pdnorm.conditions = model.conditions
        return pdnorm

    def update_norm_layers(module):
        for name, child in module.named_children():
            if isinstance(child, PDNorm):
                setattr(module, name, expand_pdnorm(child))
            else:
                update_norm_layers(child)

    update_norm_layers(model)

    old_embed = model.embedding_table
    device = old_embed.weight.device
    new_embed = nn.Embedding(len(model.conditions), old_embed.embedding_dim).to(device)
    nn.init.normal_(new_embed.weight, mean=0.0, std=0.02)
    new_embed.weight.data[:len(original_conditions)] = old_embed.weight.data
    
    for i, condition in enumerate(new_conditions):
        new_idx = len(original_conditions) + i
        if condition_mapping.get(condition) in original_conditions:
            source_idx = original_conditions.index(condition_mapping[condition])
            new_embed.weight.data[new_idx] = old_embed.weight.data[source_idx]
    
    model.embedding_table = new_embed

    return model


@MODELS.register_module("PPT-LoRA")
class PointPromptTrainingLoRA(nn.Module):
    """Point Prompt Training for multi-dataset 3D scene understanding (LoRA variant)."""

    def __init__(
        self,
        rank: int = 10,
        lora_alpha: int = 20,
        lora_dropout_p: float = 0.,
        base_model_config: Path = TRAINED_PPT_BASE_CONFIG,
        new_conditions: list[str] = ["Heritage"],
        condition_mapping: dict[str, str | None] | None = {"Heritage": "ScanNet"},
        device: str = "cuda",
    ):
        """Initialize the PointPromptTraining model."""
        super().__init__()
        self.base_model_config = base_model_config
        self.lora_config = ppt_lora_config(rank, lora_alpha, lora_dropout_p)
        self.new_conditions = new_conditions
        self.condition_mapping = condition_mapping
        self.device = device

        self._load_base_model()
        self._inject_trainable_parameters()

    def _load_base_model(self):
        self.model = load_base_model(self.base_model_config, device=self.device) 

    def load_state_dict(
        self,
        state_dict: ty.Mapping[str, ty.Any],
        strict: bool = True,
        assign: bool = False
    ):
        # print("*" * 88)
        # print(list(state_dict.keys())[:10])
        # print("*" * 88)
        prefix = 'model.'
        n_clip = len(prefix)
        adapted_dict = {k[n_clip:]: v for k, v in state_dict.items()
                        if k.startswith(prefix)}
        return self.model.load_state_dict(adapted_dict, strict, assign)

    def _inject_trainable_parameters(self):
        assert self.model is not None
        # track parameters 
        n_param_trainable_base = count_trainable_parameters(self.model)
        logger.info(f"{n_param_trainable_base} trainable parameters on base model")
        # freeze weights in original model
        self.weight_freezer = WeightFreezer(self.model)
        self.weight_freezer.freeze_all()
        # insert new parameters in normalisation layers to accommodate new datasets
        if self.new_conditions is not None:
            self.model = expand_ppt_model_conditions(self.model, self.new_conditions, self.condition_mapping)
        # insert LoRA adapters to model
        minlora.add_lora(self.model, lora_config=self.lora_config)
        # track parameters
        lora_trainable_params = count_trainable_parameters(self.model)
        logger.info(f"{lora_trainable_params} params after LoRA")

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)


def ppt_lora_config(
    rank: int = 10,
    lora_alpha: int = 20,
    lora_dropout_p: float = 0.,
    **lora_parametrization_extra_kwargs
) -> dict:
    return {
        torch.nn.Embedding: {
            "weight": partial(
                LoRAParametrization.from_embedding,
                rank=rank,
                lora_alpha=lora_alpha,
                lora_dropout_p=lora_dropout_p,
                **lora_parametrization_extra_kwargs
            ),
        },
        torch.nn.Linear: {
            "weight": partial(LoRAParametrization.from_linear,
                rank=rank,
                lora_alpha=lora_alpha,
                lora_dropout_p=lora_dropout_p,
                **lora_parametrization_extra_kwargs
            ),
        },
        SubMConv3d: {
            "weight": partial(LoRAParametrization.from_sparseconv3d,
                rank=rank,
                lora_alpha=lora_alpha,
                lora_dropout_p=lora_dropout_p,
                **lora_parametrization_extra_kwargs
            ),
        }
    }



            
