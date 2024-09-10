import logging
import typing as ty
from functools import partial
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
    create_spoofed_input,
    configure_adamw_lora,
    assert_lora_trainable,
    total_optimized_params,
    count_trainable_parameters,
    count_lora_parameters
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


REPO_ROOT = Path(__file__).parent.parent.parent.parent.parent
TRAINED_PPT_BASE_CONFIG = REPO_ROOT / "test/custom-ppt-config.py" 


def load_base_model(cfg_file: Path = TRAINED_PPT_BASE_CONFIG, device: str = "cuda") -> nn.Module:
    """load trained PPT model weights from config for application of LoRA / pdnorm expansion"""
    assert cfg_file.exists
    args = default_argument_parser().parse_args(args=["--config-file", f"{cfg_file}"])
    cfg = default_config_parser(args.config_file, args.options); cfg = patch_cfg(cfg)
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


def test_ppt_model_expansion(
    model,
    new_conditions: list[str] = ["NewDataset1", "NewDataset2"],
    condition_mapping: dict[str, str | None] | None = {
        "NewDataset1": "ScanNet",  # Copy from ScanNet
        "NewDataset2": None  # Random initialization
    },
    device: str = "cuda"
):
    """
    Test function to verify the correctness of PDNorm expansion in a PPT model.
    
    Args:
    - model: The original PPT model
    - new_conditions: List of new conditions to add (default: ["NewDataset1", "NewDataset2"])
    - device: The device to run the test on (default: "cuda")
    
    Returns:
    - None, but raises AssertionError if any test fails
    """
    # Ensure the model is on the specified device
    model = model.to(device)
    
    # Setup
    original_conditions = model.conditions
    condition_mapping = {
        "NewDataset1": "ScanNet",  # Copy from ScanNet
        "NewDataset2": None  # Random initialization
    }
    
    # Store original embedding weights
    original_embedding_weights = model.embedding_table.weight.clone()
    
    # Expand the model
    expanded_model = expand_ppt_model_conditions(model, new_conditions, condition_mapping)
    expanded_model = expanded_model.to(device)
    
    # Helper function to check if tensors are close
    def tensors_close(a, b, rtol=1e-5, atol=1e-8):
        return torch.allclose(a.to(device), b.to(device), rtol=rtol, atol=atol)
    
    # Test embedding table
    assert expanded_model.embedding_table.weight.shape[0] == len(original_conditions) + len(new_conditions), "Embedding table size mismatch"
    assert tensors_close(expanded_model.embedding_table.weight[:len(original_conditions)], original_embedding_weights), "Original embeddings changed"
    assert tensors_close(
        expanded_model.embedding_table.weight[len(original_conditions)], 
        original_embedding_weights[original_conditions.index("ScanNet")]
    ), "NewDataset1 embedding not copied correctly"
    
    def check_pdnorm_layers(module, prefix=''):
        for name, child in module.named_children():
            full_name = f"{prefix}.{name}" if prefix else name
            if isinstance(child, PDNorm):
                assert len(child.norm) == len(original_conditions) + len(new_conditions), f"PDNorm {full_name} size mismatch"
                
                # Get the corresponding PDNorm from the original model
                original_pdnorm = model
                for part in full_name.split('.'):
                    original_pdnorm = getattr(original_pdnorm, part)
                
                # Check parameters of original conditions
                for i, condition in enumerate(original_conditions):
                    assert tensors_close(child.norm[i].weight, original_pdnorm.norm[i].weight), f"Weight mismatch for {condition} in {full_name}"
                    assert tensors_close(child.norm[i].bias, original_pdnorm.norm[i].bias), f"Bias mismatch for {condition} in {full_name}"
                    assert child.norm[i].eps == original_pdnorm.norm[i].eps, f"Eps mismatch for {condition} in {full_name}"
                
                # Check parameters of new conditions
                scannet_idx = original_conditions.index("ScanNet")
                new_dataset1_idx = len(original_conditions)
                new_dataset2_idx = len(original_conditions) + 1
                
                # NewDataset1 should be copied from ScanNet
                if not tensors_close(child.norm[new_dataset1_idx].weight, child.norm[scannet_idx].weight):
                    logger.info(f"NewDataset1 weight: {child.norm[new_dataset1_idx].weight}")
                    logger.info(f"ScanNet weight: {child.norm[scannet_idx].weight}")
                    raise AssertionError(f"NewDataset1 weight not copied correctly in {full_name}")
                
                if not tensors_close(child.norm[new_dataset1_idx].bias, child.norm[scannet_idx].bias):
                    logger.info(f"NewDataset1 bias: {child.norm[new_dataset1_idx].bias}")
                    logger.info(f"ScanNet bias: {child.norm[scannet_idx].bias}")
                    raise AssertionError(f"NewDataset1 bias not copied correctly in {full_name}")
                
                # NewDataset2 should be randomly initialized
                assert not tensors_close(child.norm[new_dataset2_idx].weight, child.norm[scannet_idx].weight, rtol=1e-3, atol=1e-3), f"NewDataset2 weight should not match ScanNet in {full_name}"
                
                # Check that NewDataset2 is properly initialized
                assert torch.allclose(child.norm[new_dataset2_idx].weight.mean(), torch.tensor(1.0, device=device), rtol=1e-1), f"NewDataset2 weight not properly initialized in {full_name}"
                assert torch.allclose(child.norm[new_dataset2_idx].bias.mean(), torch.tensor(0.0, device=device), rtol=1e-1), f"NewDataset2 bias not properly initialized in {full_name}"
                
                # Check eps and other parameters
                for i in range(len(child.norm)):
                    assert child.norm[i].eps == child.norm[0].eps, f"Eps mismatch in {full_name} for layer {i}"
                    if isinstance(child.norm[i], nn.BatchNorm1d):
                        assert child.norm[i].momentum == child.norm[0].momentum, f"Momentum mismatch in {full_name} for layer {i}"
                        assert child.norm[i].affine == child.norm[0].affine, f"Affine mismatch in {full_name} for layer {i}"
                        assert child.norm[i].track_running_stats == child.norm[0].track_running_stats, f"Track_running_stats mismatch in {full_name} for layer {i}"
                    elif isinstance(child.norm[i], nn.LayerNorm):
                        assert child.norm[i].elementwise_affine == child.norm[0].elementwise_affine, f"Elementwise_affine mismatch in {full_name} for layer {i}"
            else:
                check_pdnorm_layers(child, full_name)
    
    # Run the recursive check
    check_pdnorm_layers(expanded_model)
    logger.info("All tests passed successfully!")


@MODELS.register_module("PPT-LoRA")
class PointPromptTrainingLoRA(nn.Module):
    """Point Prompt Training for multi-dataset 3D scene understanding (LoRA variant)."""

    def __init__(
        self,
        base_model_config: Path,
        lora_config: dict,
        freeze_config: dict,
        new_conditions: list[str] = ["Heritage"],
        condition_mapping: dict[str, str | None] | None = {"Heritage": "ScanNet"},
        device: str = "cuda",
    ):
        """Initialize the PointPromptTraining model."""
        super().__init__()
        self.base_model_config = base_model_config
        self.lora_config = lora_config
        self.freeze_config = freeze_config
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

    def test_ppt_model_expansion(self):
        logger.info(f"Testing expansion of normalisation layers to new conditions {self.new_conditions}")
        logger.warning("Purging model from memory temporarily to conduct PDNorm layer injection test (modifies inplace)")
        # purge model
        self.model = None; gc.collect()
        # reload model from config
        self.model = load_base_model(self.base_model_config, device=self.device) 
        test_ppt_model_expansion(
            self.model,
            new_conditions=self.new_conditions,
            condition_mapping=self.condition_mapping,
            device=self.device
        )
        # purge after test
        self.model = None; gc.collect()
        self.model = load_base_model(self.base_model_config, device=self.device) 
        self._inject_trainable_parameters()

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)


# lora adapter hyperparameters
lora_hparams = dict(
    lora_dropout_p = 0.0,
    rank = 10,
    lora_alpha = 20
)

# optimizer hyperparameters
weight_decay = 0.05
learning_rate = 0.005
beta1, beta2 = 0.9, 0.999#0.95
device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast

lora_config = {
    torch.nn.Embedding: {
        "weight": partial(LoRAParametrization.from_embedding, **lora_hparams),
    },
    torch.nn.Linear: {
        "weight": partial(LoRAParametrization.from_linear, **lora_hparams),
    },
    SubMConv3d: {
        "weight": partial(LoRAParametrization.from_sparseconv3d, **lora_hparams),
    }
}


# create AdamW optimizer (for LoRA weights only)
optimizer = configure_adamw_lora(
    model,
    weight_decay,
    learning_rate,
    (beta1, beta2),
    device_type
)

print("performing cross checks")
# check all lora parameters trainable
assert_lora_trainable(model)
# check manual lora parameter counting against minlora to check that it matches
assert count_lora_parameters(model) == count_lora_params_manual(model)
# cross check with lora params with gradient enabled
assert total_optimized_params(optimizer) == lora_trainable_params

print("restoring initial model state")
# remove adapters and unfreeze weights to original state
minlora.remove_lora(model)
wf.unfreeze_all()

print("# trainable params after removing lora:", count_trainable_parameters(model))
wf.print_frozen_status()

wf.freeze_all()
minlora.add_lora(model, lora_config=lora_config)

def inspect_lora_gradients(model, x, num_steps=5):
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    def check_grads():
        a_no_grad, b_no_grad = [], []
        a_with_grad, b_with_grad = 0, 0
        total_a, total_b = 0, 0
        trainable_params_with_grad = 0
        frozen_params = 0
        total_params = 0

        for name, param in model.named_parameters():
            total_params += param.numel()
            if not param.requires_grad:
                frozen_params += param.numel()
            elif param.grad is not None and torch.any(param.grad != 0):
                trainable_params_with_grad += param.numel()

            if 'lora_A' in name:
                total_a += 1
                if param.grad is None or torch.all(param.grad == 0):
                    a_no_grad.append(name)
                else:
                    a_with_grad += 1
            elif 'lora_B' in name:
                total_b += 1
                if param.grad is None or torch.all(param.grad == 0):
                    b_no_grad.append(name)
                else:
                    b_with_grad += 1

        return (a_with_grad, b_with_grad, total_a, total_b, a_no_grad, b_no_grad, 
                trainable_params_with_grad, frozen_params, total_params)

    # Initial forward and backward pass
    y = model(x)
    loss = y["loss"].sum()
    loss.backward()
    
    results = check_grads()
    (
        a_grad,
        b_grad,
        total_a,
        total_b,
        a_no_grad,
        b_no_grad,
        trainable_grad,
        frozen,
        total
    ) = results

    logger.info("*** First Pass ***")
    logger.info
    logger.info(f"Initial gradients: A: {a_grad}/{total_a}, B: {b_grad}/{total_b}")
    logger.info(f"Trainable parameters with gradients: {trainable_grad:,}")
    logger.info(f"Frozen parameters: {frozen:,}")
    logger.info(f"Total parameters: {total:,}")
    if a_no_grad:
        logger.info(f"Total A matrices without gradients: {len(a_no_grad)}")
    if b_no_grad:
        logger.info(f"Total B matrices without gradients: {len(b_no_grad)}")

    # Perform several optimization steps
    for i in range(num_steps):
        optimizer.step()
        optimizer.zero_grad()
        
        y = model(x)
        loss = y["loss"].sum()
        loss.backward()
        
        results = check_grads()
        a_grad, b_grad, total_a, total_b, a_no_grad, b_no_grad, trainable_grad, frozen, total = results

        logger.info(f"\nGradients after step {i+1}:")
        logger.info(f"A: {a_grad}/{total_a}, B: {b_grad}/{total_b}")
        logger.info(f"Trainable parameters with gradients: {trainable_grad:,}")
        logger.info(f"Frozen parameters: {frozen:,}")
        logger.info(f"Total parameters: {total:,}")
        if a_no_grad:
            logger.info(f"A matrices without gradients: {a_no_grad}")
        if b_no_grad:
            logger.info(f"B matrices without gradients: {b_no_grad}")
            
X = create_spoofed_input(device="cuda", batch_size=16)
inspect_lora_gradients(model, X)
