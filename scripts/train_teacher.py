import os
import argparse
import math
import time
import copy
import random
import numpy as np
from typing import Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim

# Dataset wrapper for preprocessed ECG arrays
from src.data.dataset import ECGDataset
# Train-time ECG augmentation pipeline
from src.data.augmentations import ECGAugment
# Teacher model architecture
from src.models.teacher_cnn import TeacherCNN
# Evaluation helper returning accuracy, macro-AUC, macro-F1, etc.
from src.utils.metrics import evaluate_classification
# Utility helpers for model size, parameter counts, checkpoint size, and latency
from src.utils.model_stats import (
    count_trainable_params,
    count_all_params,
    model_size_mb,
    checkpoint_size_mb,
    measure_latency,
)

# Optional EMA support for stabilising teacher weights during training
try:
    from torch_ema import ExponentialMovingAverage
    HAS_EMA = True
except ImportError:
    ExponentialMovingAverage = None
    HAS_EMA = False


# Path to preprocessed train/validation/test splits
DATA_PATH = os.path.join("processed", "ptbxl_500hz_10s.npz")
# Directory for checkpoint storage
CHECKPOINT_DIR = "checkpoints"
# Default output checkpoint path for the best teacher model
DEFAULT_TEACHER_CKPT = os.path.join(CHECKPOINT_DIR, "teacher_cnn_best.pt")


def set_seed(seed: int, deterministic: bool = True, warn_only: bool = True):
    """
    Set Python, NumPy, and PyTorch random seeds for reproducibility.

    Args:
        seed: seed value
        deterministic: whether to force deterministic PyTorch behaviour
        warn_only: whether unsupported deterministic operations should warn instead of fail
    """
    # Python RNG seed
    random.seed(seed)
    # NumPy RNG seed
    np.random.seed(seed)
    # Make Python hash-based operations reproducible
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch CPU seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # PyTorch CUDA seed for current GPU
        torch.cuda.manual_seed(seed)
        # PyTorch CUDA seed for all visible GPUs
        torch.cuda.manual_seed_all(seed)

    # Configure cuDNN reproducibility settings
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

    try:
        if deterministic:
            # Enforce deterministic algorithms where available
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        else:
            # Allow non-deterministic kernels
            torch.use_deterministic_algorithms(False)
    except Exception:
        # Some PyTorch builds / ops may not fully support this
        pass


def seed_worker(worker_id: int):
    """
    Seed DataLoader worker processes deterministically.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_splits(path: str = DATA_PATH) -> Tuple[np.ndarray, ...]:
    """
    Load preprocessed PTB-XL train/validation/test splits.

    Args:
        path: path to compressed .npz file

    Returns:
        Tuple containing:
            X_train, y_train, X_val, y_val, X_test, y_test, classes
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed .npz not found at {path}. Run scripts/preprocess_ptbxl.py first."
        )

    # Use memory mapping for more efficient loading of large arrays
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
        data["classes"],
    )


def compute_class_weights(labels: np.ndarray, gamma: float = 0.20) -> torch.Tensor:
    """
    Compute smoothed inverse-frequency class weights.

    Args:
        labels: integer class labels
        gamma: exponent controlling strength of inverse-frequency weighting

    Returns:
        Torch tensor of class weights
    """
    classes, counts = np.unique(labels, return_counts=True)

    # Check labels are contiguous integers 0..C-1
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(
            f"Labels must be contiguous integers from 0..C-1, but got classes={classes}"
        )

    # Relative class frequencies
    freq = counts.astype(np.float32) / counts.sum()
    # Inverse frequency weighting
    inv = 1.0 / (freq + 1e-6)
    # Smooth weighting strength using gamma
    weights = np.power(inv, gamma)
    # Normalise so mean weight is 1
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(labels: np.ndarray, power: float = 0.35) -> WeightedRandomSampler:
    """
    Build a weighted sampler for oversampling minority classes.

    Args:
        labels: integer class labels
        power: exponent controlling sampling aggressiveness

    Returns:
        WeightedRandomSampler instance
    """
    # Number of samples per class
    class_counts = np.bincount(labels)
    # Inverse class frequency weights
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
    # Smooth the weighting using power
    class_weights = np.power(class_weights, power)
    # Assign each sample the weight of its class
    sample_weights = class_weights[labels]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_cosine_with_warmup_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_factor: float,
):
    """
    Build a learning-rate scheduler with linear warmup followed by cosine decay.

    Args:
        optimizer: optimiser to schedule
        total_epochs: total number of epochs
        warmup_epochs: number of warmup epochs
        min_lr_factor: minimum LR as a fraction of initial LR

    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(epoch: int):
        # Linear warmup phase
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        # Cosine decay phase
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    scaler: torch.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
    ema=None,
):
    """
    Train the teacher model for one epoch.

    Args:
        model: teacher model
        loader: training DataLoader
        criterion: classification loss
        optimizer: optimiser
        scaler: AMP GradScaler
        device: cpu or cuda
        grad_clip: gradient clipping threshold
        ema: optional exponential moving average tracker

    Returns:
        Tuple of (average training loss, training accuracy)
    """
    # Set model to training mode
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Use AMP only on CUDA
    use_amp = device.type == "cuda"

    for X, y in loader:
        # Move batch to device
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Clear previous gradients
        optimizer.zero_grad(set_to_none=True)

        # Mixed precision forward pass
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = model(X)
            loss = criterion(logits, y)

        # Backpropagate scaled loss
        scaler.scale(loss).backward()

        # Optional gradient clipping
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Optimiser step and scaler update
        scaler.step(optimizer)
        scaler.update()

        # Update EMA weights if enabled
        if ema is not None:
            ema.update()

        # Accumulate metrics
        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / float(total), float(correct) / float(total)


def parse_args():
    """
    Parse command-line arguments for teacher training.
    """
    parser = argparse.ArgumentParser(description="Train teacher model on PTB-XL")

    # Core optimisation settings
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)

    # Training duration / early stopping
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--patience", type=int, default=18)

    # LR scheduling settings
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr_factor", type=float, default=0.02)

    # Regularisation / imbalance handling
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--class_weight_gamma", type=float, default=0.20)
    parser.add_argument("--label_smoothing", type=float, default=0.03)
    parser.add_argument("--sampler_power", type=float, default=0.35)

    # Data loading / reproducibility
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)

    # Teacher capacity choice
    parser.add_argument(
        "--teacher_size",
        type=str,
        default="large",
        choices=["medium", "large", "xlarge"],
        help="Teacher capacity to use",
    )

    # Output checkpoint path
    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        default=DEFAULT_TEACHER_CKPT,
        help="Path to save the best teacher checkpoint",
    )

    # Optional weighted sampling
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="Enable weighted sampling.",
    )

    # Optional disabling of loss class weights
    parser.add_argument(
        "--no_class_weights",
        action="store_true",
        help="Disable class-weighted CE.",
    )

    # Optional disabling of training augmentation
    parser.add_argument(
        "--disable_augmentation",
        action="store_true",
        help="Disable train-time augmentation.",
    )

    # EMA decay hyperparameter
    parser.add_argument(
        "--ema_decay",
        type=float,
        default=0.999,
        help="EMA decay. Ignored if torch-ema is unavailable.",
    )

    # Determinism controls
    parser.add_argument("--no_deterministic", action="store_true")
    parser.add_argument("--strict_deterministic", action="store_true")

    return parser.parse_args()


def save_checkpoint(
    path: str,
    model: nn.Module,
    classes,
    n_leads: int,
    best_val_macro_auc: float,
    epoch: int,
    args,
):
    """
    Save teacher checkpoint atomically.

    Args:
        path: final checkpoint path
        model: teacher model
        classes: class names
        n_leads: number of ECG leads
        best_val_macro_auc: best validation macro-AUC achieved
        epoch: epoch at which checkpoint was saved
        args: parsed training arguments
    """
    payload = {
        # Deep copy model weights to ensure clean checkpoint state
        "model_state_dict": copy.deepcopy(model.state_dict()),
        # Metadata needed for later reuse
        "classes": classes,
        "n_leads": n_leads,
        "teacher_arch": "TeacherCNN",
        "teacher_size": args.teacher_size,
        "best_val_macro_auc": best_val_macro_auc,
        "epoch": epoch,
        "seed": args.seed,
        "train_args": vars(args),
    }

    # Save to temporary file first, then atomically replace
    tmp_path = path + ".tmp"
    torch.save(payload, tmp_path)
    os.replace(tmp_path, path)

    # Sanity check that file exists
    if not os.path.exists(path):
        raise RuntimeError(f"Checkpoint save failed: {path}")


def main():
    """
    Main pipeline for training and evaluating the teacher model.
    """
    args = parse_args()

    # Configure reproducibility settings
    deterministic = not args.no_deterministic
    warn_only = not args.strict_deterministic

    set_seed(args.seed, deterministic=deterministic, warn_only=warn_only)

    # Select device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)
    print("Seed:", args.seed, flush=True)
    print("Teacher size:", args.teacher_size, flush=True)
    print("Teacher checkpoint path:", args.teacher_ckpt, flush=True)
    print("Deterministic mode:", deterministic, flush=True)
    print("Deterministic warn_only:", warn_only, flush=True)
    print(
        (
            "Training: "
            f"batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}, "
            f"patience={args.patience}, warmup_epochs={args.warmup_epochs}, "
            f"min_lr_factor={args.min_lr_factor}, weight_decay={args.weight_decay}, "
            f"class_weight_gamma={args.class_weight_gamma}, "
            f"label_smoothing={args.label_smoothing}, sampler_power={args.sampler_power}, "
            f"grad_clip={args.grad_clip}, num_workers={args.num_workers}, "
            f"teacher_size={args.teacher_size}, use_weighted_sampler={args.use_weighted_sampler}, "
            f"class_weights={not args.no_class_weights}, disable_augmentation={args.disable_augmentation}, "
            f"ema_decay={args.ema_decay}"
        ),
        flush=True,
    )

    # Load processed train / validation / test splits
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_splits()
    n_leads = X_train.shape[1]
    n_classes = len(classes)

    # Print dataset summary
    print("Training set size:", X_train.shape[0], flush=True)
    print("Validation set size:", X_val.shape[0], flush=True)
    print("Test set size:", X_test.shape[0], flush=True)
    print("Number of leads:", n_leads, "classes:", n_classes, flush=True)
    print("Classes:", classes, flush=True)

    # Use augmentation on training set unless disabled
    train_transform = None if args.disable_augmentation else ECGAugment()

    # Wrap arrays in dataset objects
    train_dataset = ECGDataset(X_train, y_train, transform=train_transform)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)

    # Shared DataLoader parameters
    common_loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
    }

    # Build training loader with either weighted sampler or standard shuffle
    if args.use_weighted_sampler:
        train_sampler = build_weighted_sampler(y_train, power=args.sampler_power)
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            sampler=train_sampler,
            **common_loader_kwargs,
        )
        print("Training sampler: weighted random sampler enabled", flush=True)
    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            **common_loader_kwargs,
        )
        print("Training sampler: standard shuffled loader", flush=True)

    # Validation and test loaders should not shuffle
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **common_loader_kwargs,
    )

    # Build teacher model
    model = TeacherCNN(
        n_leads=n_leads,
        n_classes=n_classes,
        size=args.teacher_size,
    ).to(device)

    # Print model statistics
    print("Trainable parameters:", count_trainable_params(model), flush=True)
    print("Total parameters:", count_all_params(model), flush=True)
    print("Estimated model size (MB): {:.3f}".format(model_size_mb(model)), flush=True)

    # Optionally disable class weights
    if args.no_class_weights:
        class_weights = None
        print("Class weights: DISABLED", flush=True)
    else:
        class_weights = compute_class_weights(
            y_train,
            gamma=args.class_weight_gamma,
        ).to(device)
        print("Class weights:", class_weights.cpu().numpy(), flush=True)

    # Cross-entropy loss with optional class weights and label smoothing
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # AdamW optimiser
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    # Warmup + cosine decay scheduler
    scheduler = build_cosine_with_warmup_scheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr_factor=args.min_lr_factor,
    )

    # AMP gradient scaler for mixed precision
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    # Optional exponential moving average of weights
    ema = None
    if HAS_EMA:
        ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
        print("EMA enabled", flush=True)
    else:
        print("EMA not available (torch-ema not installed)", flush=True)

    # Ensure checkpoint directory exists
    ckpt_dir = os.path.dirname(args.teacher_ckpt)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    # Tracking variables for best validation model
    best_val_macro_auc = -1.0
    best_epoch = 0
    best_time_sec = 0.0
    epochs_without_improvement = 0
    training_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Train one epoch
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
            ema=ema,
        )

        # Evaluate with EMA weights after warmup if available, otherwise raw weights
        if ema is not None and epoch >= max(3, args.warmup_epochs):
            with ema.average_parameters():
                val_metrics = evaluate_classification(model, val_loader, device)
        else:
            val_metrics = evaluate_classification(model, val_loader, device)

        val_acc = val_metrics["acc"]
        val_macro_auc = val_metrics["macro_auc"]
        val_macro_f1 = val_metrics["macro_f1"]
        val_weighted_f1 = val_metrics["weighted_f1"]

        # Step scheduler after epoch
        scheduler.step()
        epoch_time = time.time() - epoch_start_time

        # Print epoch summary
        print(
            "[Epoch {:03d}] lr={:.6f}, train_loss={:.4f}, train_acc={:.4f}, "
            "val_acc={:.4f}, val_macro_auc={:.4f}, val_macro_f1={:.4f}, "
            "val_weighted_f1={:.4f}, epoch_time={:.2f}s".format(
                epoch,
                optimizer.param_groups[0]["lr"],
                train_loss,
                train_acc,
                val_acc,
                val_macro_auc,
                val_macro_f1,
                val_weighted_f1,
                epoch_time,
            ),
            flush=True,
        )

        # Save checkpoint only if validation macro-AUC improves
        if not np.isnan(val_macro_auc) and val_macro_auc > best_val_macro_auc:
            best_val_macro_auc = val_macro_auc
            best_epoch = epoch
            best_time_sec = time.time() - training_start_time
            epochs_without_improvement = 0

            # Save EMA-averaged weights if EMA is active and sufficiently warmed up
            if ema is not None and epoch >= max(3, args.warmup_epochs):
                with ema.average_parameters():
                    save_checkpoint(
                        path=args.teacher_ckpt,
                        model=model,
                        classes=classes,
                        n_leads=n_leads,
                        best_val_macro_auc=best_val_macro_auc,
                        epoch=epoch,
                        args=args,
                    )
            else:
                save_checkpoint(
                    path=args.teacher_ckpt,
                    model=model,
                    classes=classes,
                    n_leads=n_leads,
                    best_val_macro_auc=best_val_macro_auc,
                    epoch=epoch,
                    args=args,
                )

            print("  -> Saved best teacher model to", args.teacher_ckpt, flush=True)
        else:
            # Early stopping patience counter
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping", flush=True)
                break

    # Ensure best checkpoint exists
    if not os.path.exists(args.teacher_ckpt):
        raise FileNotFoundError(f"Best teacher checkpoint not found: {args.teacher_ckpt}")

    # Reload best checkpoint for final evaluation
    ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\nLoaded best checkpoint for final evaluation.", flush=True)
    print("Best validation macro-AUC: {:.4f}".format(best_val_macro_auc), flush=True)
    print("Best epoch:", best_epoch, flush=True)
    print("Time to best model: {:.2f} seconds".format(best_time_sec), flush=True)
    print("Checkpoint size (MB): {:.3f}".format(checkpoint_size_mb(args.teacher_ckpt)), flush=True)
    print("Checkpoint seed:", ckpt.get("seed", "UNKNOWN"), flush=True)
    print("Checkpoint teacher size:", ckpt.get("teacher_size", "UNKNOWN"), flush=True)
    print("Checkpoint teacher arch:", ckpt.get("teacher_arch", "UNKNOWN"), flush=True)

    # Final test-set evaluation
    test_metrics = evaluate_classification(model, test_loader, device)

    print("Test Accuracy: {:.4f}".format(test_metrics["acc"]), flush=True)
    print("Test Macro-AUC: {:.4f}".format(test_metrics["macro_auc"]), flush=True)
    print("Test Macro-F1: {:.4f}".format(test_metrics["macro_f1"]), flush=True)
    print("Test Weighted-F1: {:.4f}".format(test_metrics["weighted_f1"]), flush=True)

    # Print full classification report if available
    if "report" in test_metrics:
        print("Classification Report:", flush=True)
        print(test_metrics["report"], flush=True)

    # Print confusion matrix if available
    if "confusion_matrix" in test_metrics:
        print("Confusion Matrix:", flush=True)
        print(test_metrics["confusion_matrix"], flush=True)

    # Reset seed before latency measurement for consistency
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Dummy input matching expected ECG tensor shape for latency benchmarking
    dummy_input = torch.randn(1, n_leads, X_train.shape[2], device=device)
    latency_sec = measure_latency(model, dummy_input, device)
    print("Inference latency: {:.4f} ms/sample".format(latency_sec * 1000.0), flush=True)


if __name__ == "__main__":
    main()