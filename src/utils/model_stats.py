import os
import time
import torch


def count_trainable_params(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    """
    # Sum the number of elements (parameters) for all tensors
    # that require gradients (i.e., are trainable)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_all_params(model: torch.nn.Module) -> int:
    """
    Count the total number of parameters in a model, including frozen ones.
    """
    # Sum the number of elements for all parameters regardless of trainability
    return sum(p.numel() for p in model.parameters())


def model_size_mb(model: torch.nn.Module) -> float:
    """
    Estimate in-memory model size in MB from parameters and buffers.
    """
    # Calculate total size of parameters (weights, biases)
    param_size = 0
    for p in model.parameters():
        # numel() = number of elements, element_size() = bytes per element
        param_size += p.numel() * p.element_size()

    # Calculate total size of buffers (e.g., BatchNorm running stats)
    buffer_size = 0
    for b in model.buffers():
        buffer_size += b.numel() * b.element_size()

    # Total size in bytes
    total_size_bytes = param_size + buffer_size

    # Convert bytes → megabytes
    return total_size_bytes / (1024 ** 2)


def checkpoint_size_mb(path: str) -> float:
    """
    Return checkpoint file size in MB.
    """
    # Check that checkpoint file exists
    if not os.path.exists(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Get file size in bytes and convert to MB
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

    # Set model to evaluation mode (disable dropout, etc.)
    model.eval()

    # Move model and input to specified device (CPU or GPU)
    model = model.to(device)
    input_tensor = input_tensor.to(device)

    # Warmup phase: run forward passes without timing
    # This stabilises GPU performance (avoids startup overhead affecting results)
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(input_tensor)

    # Ensure all CUDA operations are complete before timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Start timing
    start = time.time()

    # Run multiple forward passes for accurate average latency
    with torch.no_grad():
        for _ in range(runs):
            _ = model(input_tensor)

    # Ensure all CUDA operations are complete after timing
    if device.type == "cuda":
        torch.cuda.synchronize()

    # End timing
    end = time.time()

    # Return average time per forward pass
    return (end - start) / runs