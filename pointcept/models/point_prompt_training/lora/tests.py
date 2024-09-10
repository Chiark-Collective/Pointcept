import logging
from functools import wraps

import torch
import torch.optim
import torch.nn as nn
import torch
import gc
from pointcept.models.point_prompt_training import PDNorm

from .ppt import expand_ppt_model_conditions, load_base_model, PointPromptTrainingLoRA
from .utils import (
    create_spoofed_input,
)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


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


def reload_model_for_test():
    def decorator(test_method):
        @wraps(test_method)
        def wrapper(self, *args, **kwargs):
            logger.warning("Purging model from memory temporarily to conduct test (modifies inplace)")
            # Purge and reload model
            self.model = None; gc.collect()
            self.model = load_base_model(self.base_model_config, device=self.device)
            try:
                # Run the test
                result = test_method(self, *args, **kwargs)
            finally:
                # Purge test model
                self.model = None; gc.collect()
                # Restore original model
                self.model = load_base_model(self.base_model_config, device=self.device)
                self._inject_trainable_parameters()
            return result
        return wrapper
    return decorator


class PointPromptTrainingLoRATester:
    def __init__(self, model: PointPromptTrainingLoRA):
        self.model = model
        self.base_model_config = model.base_model_config
        self.device = model.device
        self.new_conditions = model.new_conditions
        self.condition_mapping = model.condition_mapping

    @reload_model_for_test()
    def test_ppt_model_expansion(self):
        logger.info(f"Testing expansion of normalisation layers to new conditions {self.new_conditions}")
        test_ppt_model_expansion(
            self.model,
            new_conditions=self.new_conditions,
            condition_mapping=self.condition_mapping,
            device=self.device
        )

    @reload_model_for_test()
    def test_lora_gradients(self):
        X = create_spoofed_input(device=self.device, batch_size=16)
        inspect_lora_gradients(self.model, X)

    def test(self):
        self.test_ppt_model_expansion()
        self.test_lora_gradients()

