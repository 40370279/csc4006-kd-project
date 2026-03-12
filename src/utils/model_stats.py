import os
import time
import torch


def count_trainable_params(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: torch.nn.Module) -> int:
    """
    Count the total number of parameters in a model, including frozen ones.
    """
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: torch.nn.Module) -> float:
    """
    Estimate in-memory model size in MB from parameters and buffers.
    """
    param_size = 0
    for p in model.parameters():
        param_size += p.numel() * p.element_size()

    buffer_size = 0
    for b in model.buffers():
        buffer_size += b.numel() * b.element_size()

    total_size_bytes = param_size + buffer_size
    return total_size_bytes / (1024 ** 2)


def checkpoint_size_mb(path: str) -> float:
    """
    Return checkpoint file size in MB.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return os.path.getsize(path) / (1024 ** 2)


def measure_latency(
    model: torch.nn.Module,
    input_tensor: torch.Tensor,
    device: torch.device,
    warmup: int = 20,
    runs: int = 100,
) -> float:
    """
    Measure average inference latency (seconds per forward pass).

    Parameters
    ----------
    model : torch.nn.Module
        Model to evaluate.
    input_tensor : torch.Tensor
        Example input tensor, e.g. shape (1, 12, 5000).
    device : torch.device
        Device on which to run the measurement.
    warmup : int
        Number of warmup iterations before timing.
    runs : int
        Number of timed forward passes.

    Returns
    -------
    float
        Average latency in seconds per forward pass.
    """
    model.eval()
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    start = time.time()

    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)

    if device.type == "cuda":
        torch.cuda.synchronize()

    end = time.time()
    return (end - start) / runs