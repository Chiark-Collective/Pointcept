
import inspect
import logging
import typing as ty
from functools import partial
from pathlib import Path

import torch
import torch.optim
import torch.nn as nn
import minlora
import torch
from minlora.utils import name_is_lora


logger = logging.getLogger(__name__)



class WeightFreezer:
    """
    Utility class for conditional, invertible freezing/unfreezing of model 
    weights with state tracking
    """
    def __init__(self, model: nn.Module) -> None:
        self.model = model
        self.original_states = {}
        self._store_initial_states()
        
    def _store_initial_states(self) -> None:
        for name, param in self.model.named_parameters():
            self.original_states[name] = param.requires_grad

    def freeze_if(self, filter_fn: ty.Callable[[str, nn.Parameter], bool] | None) -> None:
        filter_fn = filter_fn or (lambda n, p: True)
        for name, param in self.model.named_parameters():
            if filter_fn(name, param):
                param.requires_grad = False
    
    def freeze_all(self) -> None:
        return self.freeze_if(filter_fn=None)

    def unfreeze_if(
        self,
        filter_fn: ty.Callable[[str, nn.Parameter], bool] | None, 
        hard: bool = False
    ) -> None:
        """
        Defaults to restoring to original state if the filter_fn returns True,
        meaning if the initial model had certain parameters frozen, these will 
        faithfully still be frozen. Setting hard=True overrides this and unfreezes
        irrespective of the initial state.
        """
        filter_fn = filter_fn or (lambda n, p: True)
        for name, param in self.model.named_parameters():
            if filter_fn(name, param):
                if hard:
                    param.requires_grad = True
                else:
                    param.requires_grad = self.original_states.get(name, True)

    def unfreeze_all(self, hard: bool = False) -> None:
        return self.unfreeze_if(filter_fn=None, hard=hard)

    def reset(self) -> None:
        for name, param in self.model.named_parameters():
            param.requires_grad = self.original_states.get(name, True)

    def print_frozen_status(self, print_unfrozen: bool = False) -> None:
        for name, param in self.model.named_parameters():
            state = "unfrozen" if param.requires_grad else "frozen"
            if state == "unfrozen" and not print_unfrozen:
                continue
            # print(f"{name}: {state}")


def count_trainable_parameters(model):
    return dict(
        trainable=sum(p.numel() for p in model.parameters() if p.requires_grad),
        frozen=sum(p.numel() for p in model.parameters() if not p.requires_grad)
    )


def named_trainable_parameters(model):
    return dict(
        trainable=[n for n, p in model.named_parameters() if p.requires_grad],
        frozen=[n for n, p in model.named_parameters() if not p.requires_grad]
    )


def is_lora(name: str, value: nn.Parameter) -> bool:
    return name_is_lora(name)


def filter_named_params(
    model: nn.Module,
    filter_fn: ty.Callable[[str, nn.Parameter], bool] | None
) -> ty.Generator[tuple[str, nn.Parameter], ty.Any, ty.Any]:
    """
    generator which returns (parameter_name, weight tensor)
    for all tensors whose names match the filter function
    """
    for n, p in model.named_parameters():
        if filter_fn is None or filter_fn(n, p):
            yield n, p


get_named_lora_params = partial(filter_named_params, filter_fn=is_lora)
get_named_non_lora_params = partial(filter_named_params, filter_fn=(lambda x: not is_lora(x)))


def count_lora_parameters(model):
    """use minlora directly"""
    return sum(p.numel() for p in minlora.get_lora_params(model))


def count_lora_params_manual(model):
    """just looking at weight tensor names manually as a cross check"""
    return sum(p.numel() for n, p in get_named_lora_params(model))


def assert_lora_trainable(model):
    for param in minlora.get_lora_params(model):
        assert param.requires_grad


def total_optimized_params(optimizer: torch.optim.Optimizer) -> int:
    tot = 0
    for param_group in optimizer.param_groups:
        for param in param_group["params"]:
            tot += param.numel()
    return tot

    
def create_spoofed_input(batch_size=2, num_points=1000, n_classes=5, num_features=6, device='cpu'):
    return {
        'coord': torch.rand(num_points * batch_size, num_features, device=device),
        'feat': torch.rand(num_points * batch_size, num_features, device=device),
        'grid_coord': torch.randint(0, 100, (num_points * batch_size, 3), device=device),
        'batch': torch.arange(batch_size, device=device).repeat_interleave(num_points),
        'offset': torch.tensor([num_points * i for i in range(1, batch_size + 1)], device=device),
        'condition': ['ScanNet'] * batch_size,
        'grid_size': torch.tensor([0.01], device=device),
        'segment': torch.randint(low=0, high=n_classes-1, size=(num_points * batch_size,), device=device)
    }


def patch_cfg(cfg: dict, repo_root: Path) -> dict:
    cfg = cfg.copy()
    cfg["my_data_root"] = repo_root / cfg["my_data_root"]
    cfg["weight"] = repo_root / cfg["weight"]
    cfg["batch_size_test_per_gpu"] = 1
    return cfg


def configure_adamw_lora(
    model,
    weight_decay: float = 0.05,
    lr: float = 0.005,
    betas: tuple[float, float] = (0.9, 0.999),
    device: str = "cuda"
) -> torch.optim.AdamW:
    """
    Create an AdamW optimiser which targets only LoRA parameters during
    gradient descent
    """
    # apply weight decay to all lora params
    optim_groups = [
        {"params": list(minlora.get_lora_params(model)) , "weight_decay": weight_decay},
        # could also add biases for fine-tuning,
        # {"params": minlora.get_bias_params(model), "weight_decay": 0.0}, # bias params don't get weight decay
    ]

    def parameter_count(optim_groups):
        n = sum(p.numel() for group in optim_groups for p in group["params"])
        if n < 1e6:
            return f"{n/1e3:.1f}k"
        else:
            return f"{n/1e6:.1f}M"

    logger.info(f"Optimizing {parameter_count(optim_groups)} parameters")

    # new PyTorch nightly has a new 'fused' option for AdamW that is much faster
    use_fused = (device == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
    logger.info(f"Using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    return torch.optim.AdamW(
        optim_groups,
        lr=lr,
        betas=betas,
        **extra_args
    )



