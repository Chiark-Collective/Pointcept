import logging
import typing as ty
from functools import partial, wraps
from pathlib import Path

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
    default_argument_parser,
    default_config_parser,
)
from pointcept.engines.test import TESTERS
from pointcept.models.builder import MODELS
from .utils import (
    WeightFreezer,
    patch_cfg,
    configure_adamw_lora,
    count_trainable_parameters,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
TRAINED_PPT_BASE_CONFIG = REPO_ROOT / "test/custom-ppt-config.py" 


def load_base_model(cfg_file: Path = TRAINED_PPT_BASE_CONFIG, repo_root: Path = REPO_ROOT, device: str = "cuda") -> nn.Module:
    """load trained PPT model weights from config for application of LoRA / pdnorm expansion"""
    assert cfg_file.exists
    args = default_argument_parser().parse_args(args=["--config-file", f"{cfg_file}"])
    # this patching thing not stricty necessary, doesn't matter though because we throw everything away except
    # the model 
    cfg = default_config_parser(args.config_file, args.options); cfg = patch_cfg(cfg, repo_root=repo_root)
    tester = TESTERS.build(dict(type=cfg.test.type, cfg=cfg))
    model = tester.model
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
    model.conditions = tuple(set(list(original_conditions) + new_conditions))

    def expand_pdnorm(pdnorm):
        if isinstance(pdnorm, PDNorm) and pdnorm.decouple:
            first_norm = pdnorm.norm[0]
            if isinstance(first_norm, nn.BatchNorm1d):
                new_norm_func = lambda: type(first_norm)(
                    first_norm.num_features,
                    eps=first_norm.eps,
                    momentum=first_norm.momentum,
                    affine=first_norm.affine,
                    track_running_stats=first_norm.track_running_stats
                )
            elif isinstance(first_norm, nn.LayerNorm):
                new_norm_func = lambda: type(first_norm)(
                    first_norm.normalized_shape,
                    eps=first_norm.eps,
                    elementwise_affine=first_norm.elementwise_affine
                )
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
    new_embed = nn.Embedding(len(model.conditions), old_embed.embedding_dim)
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
        base_model_config: Path,
        lora_config: dict,
        new_conditions: list[str] = ["Heritage"],
        condition_mapping: dict[str, str | None] | None = {"Heritage": "ScanNet"},
        device: str = "cuda",
    ):
        """Initialize the PointPromptTraining model."""
        super().__init__()
        self.base_model_config = base_model_config
        self.lora_config = lora_config
        self.new_conditions = new_conditions
        self.condition_mapping = condition_mapping
        self.device = device
        self.model = load_base_model(self.base_model_config, device=self.device) 
        self._inject_trainable_parameters()

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


# create AdamW optimizer (for LoRA weights only)
# optimizer = configure_adamw_lora(
#     model,
#     weight_decay,
#     learning_rate,
#     (beta1, beta2),
#     device_type
# )


            