import os
import argparse
import time
import random
import numpy as np
from typing import Tuple

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import torch.nn as nn
import torch.optim as optim

# Dataset wrapper for PTB-XL ECG arrays
from src.data.dataset import ECGDataset
# On-the-fly ECG augmentation for training
from src.data.augmentations import ECGAugment
# Lightweight student CNN architecture
from src.models.student_cnn import StudentCNN
# Evaluation utility returning accuracy, AUC, F1, report, confusion matrix, etc.
from src.utils.metrics import evaluate_classification
# Utility functions for model statistics and latency measurement
from src.utils.model_stats import (
    count_trainable_params,
    count_all_params,
    model_size_mb,
    checkpoint_size_mb,
    measure_latency,
)

# Path to preprocessed PTB-XL splits
DATA_PATH = os.path.join("processed", "ptbxl_500hz_10s.npz")
# Directory where checkpoints will be saved
CHECKPOINT_DIR = "checkpoints"
# Default checkpoint filename for the best baseline student
DEFAULT_STUDENT_CKPT = os.path.join(CHECKPOINT_DIR, "student_baseline_best.pt")


def set_seed(seed: int, deterministic: bool = True, warn_only: bool = True):
    """
    Set seeds for Python, NumPy, and PyTorch to improve reproducibility.

    Args:
        seed: random seed value
        deterministic: whether to force deterministic PyTorch behaviour where possible
        warn_only: if True, PyTorch will warn instead of failing when deterministic
                   implementations are unavailable for some operations
    """
    # Python RNG seed
    random.seed(seed)
    # NumPy RNG seed
    np.random.seed(seed)
    # Hash seed for reproducible Python hashing behaviour
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch CPU seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # PyTorch GPU seed for current GPU
        torch.cuda.manual_seed(seed)
        # PyTorch GPU seed for all visible GPUs
        torch.cuda.manual_seed_all(seed)

    # cuDNN deterministic mode for reproducibility
    torch.backends.cudnn.deterministic = deterministic
    # Disable benchmark mode so cuDNN does not choose non-deterministic fast kernels
    torch.backends.cudnn.benchmark = False

    try:
        if deterministic:
            # Force deterministic algorithms where possible
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        else:
            # Allow non-deterministic algorithms
            torch.use_deterministic_algorithms(False)
    except Exception:
        # Some PyTorch versions / backends may not support this fully
        pass


def seed_worker(worker_id: int):
    """
    Seed each DataLoader worker separately but deterministically.

    This helps keep multi-worker loading reproducible.
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

    # Use memory mapping to reduce RAM usage when possible
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
    Compute smoothed inverse-frequency class weights for imbalanced classification.

    Args:
        labels: integer class labels
        gamma: exponent controlling weighting strength
               0.0 -> no reweighting
               1.0 -> full inverse-frequency weighting

    Returns:
        Torch tensor of normalised class weights
    """
    classes, counts = np.unique(labels, return_counts=True)

    # Ensure labels are in contiguous 0..C-1 form
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(
            f"Labels must be contiguous integers from 0..C-1, but got classes={classes}"
        )

    # Convert counts to relative frequencies
    freq = counts.astype(np.float32) / counts.sum()
    # Inverse frequency weighting
    inv = 1.0 / (freq + 1e-6)
    # Smooth the effect of inverse weighting
    weights = np.power(inv, gamma)
    # Normalise so average weight is 1
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(labels: np.ndarray, power: float = 0.35) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to oversample underrepresented classes.

    Args:
        labels: integer class labels for training set
        power: exponent controlling sampling strength

    Returns:
        WeightedRandomSampler for balanced training batches
    """
    # Count samples per class
    class_counts = np.bincount(labels)
    # Inverse frequency class weights
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
    # Smooth sampling strength
    class_weights = np.power(class_weights, power)
    # Map each sample to the weight of its class
    sample_weights = class_weights[labels]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def train_one_epoch(
    student: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    grad_clip: float,
):
    """
    Train the baseline student for one epoch.

    Args:
        student: model to train
        loader: training DataLoader
        criterion: loss function
        optimizer: optimiser
        device: cpu or cuda
        grad_clip: max gradient norm (if > 0)

    Returns:
        Tuple of (average training loss, training accuracy)
    """
    # Set model to training mode
    student.train()

    running_loss = 0.0
    correct = 0
    total = 0

    # Use AMP only when running on CUDA
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for X, y in loader:
        # Move batch to device
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Clear old gradients
        optimizer.zero_grad(set_to_none=True)

        # Mixed-precision forward pass on CUDA
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            logits = student(X)
            loss = criterion(logits, y)

        # Backpropagate scaled loss
        scaler.scale(loss).backward()

        # Optional gradient clipping for stability
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)

        # Optimiser step through GradScaler
        scaler.step(optimizer)
        scaler.update()

        # Accumulate weighted loss
        running_loss += loss.item() * X.size(0)
        # Predicted classes
        preds = logits.argmax(dim=1)
        # Count correct predictions
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / float(total)
    acc = float(correct) / float(total)

    return avg_loss, acc


def parse_args():
    """
    Parse command-line arguments for baseline student training.
    """
    parser = argparse.ArgumentParser(description="Train baseline student on PTB-XL")

    # Core optimisation hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)

    # Class imbalance / regularisation settings
    parser.add_argument("--class_weight_gamma", type=float, default=0.20)
    parser.add_argument("--sampler_power", type=float, default=0.35)
    parser.add_argument("--label_smoothing", type=float, default=0.03)

    # Student model capacity
    parser.add_argument(
        "--student_size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
    )

    # Checkpoint output path
    parser.add_argument(
        "--student_ckpt",
        type=str,
        default=DEFAULT_STUDENT_CKPT,
    )

    # Optional weighted sampler instead of normal shuffle
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="Enable weighted sampling.",
    )

    # Optional flag to disable augmentation
    parser.add_argument(
        "--disable_augmentation",
        action="store_true",
    )

    # Determinism controls
    parser.add_argument("--no_deterministic", action="store_true")
    parser.add_argument("--strict_deterministic", action="store_true")

    return parser.parse_args()


def main():
    """
    Main training and evaluation pipeline for the baseline student model.
    """
    args = parse_args()

    # Configure reproducibility mode
    deterministic = not args.no_deterministic
    warn_only = not args.strict_deterministic
    set_seed(args.seed, deterministic=deterministic, warn_only=warn_only)

    # Select GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)
    print("Seed:", args.seed, flush=True)
    print("Deterministic mode:", deterministic, flush=True)
    print("Deterministic warn_only:", warn_only, flush=True)
    print(
        (
            "Baseline hyperparameters: "
            f"class_weight_gamma={args.class_weight_gamma:.2f}, "
            f"sampler_power={args.sampler_power:.2f}, "
            f"label_smoothing={args.label_smoothing:.3f}, "
            f"student_size={args.student_size}"
        ),
        flush=True,
    )
    print(
        (
            "Training: "
            f"batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}, "
            f"patience={args.patience}, weight_decay={args.weight_decay}, "
            f"grad_clip={args.grad_clip}, num_workers={args.num_workers}, "
            f"disable_augmentation={args.disable_augmentation}, "
            f"use_weighted_sampler={args.use_weighted_sampler}"
        ),
        flush=True,
    )
    print("Student checkpoint path:", args.student_ckpt, flush=True)

    # Load processed train / val / test arrays and class names
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_splits()
    n_leads = X_train.shape[1]
    n_classes = len(classes)

    # Print dataset summary
    print("Training set size:", X_train.shape[0], flush=True)
    print("Validation set size:", X_val.shape[0], flush=True)
    print("Test set size:", X_test.shape[0], flush=True)
    print("Number of leads:", n_leads, "classes:", n_classes, flush=True)
    print("Classes:", classes, flush=True)

    # Apply augmentation only to training data unless explicitly disabled
    train_transform = None if args.disable_augmentation else ECGAugment()

    # Wrap arrays in dataset objects
    train_dataset = ECGDataset(X_train, y_train, transform=train_transform)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)

    # Shared DataLoader options
    common_loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
    }

    # Initialise student model
    student = StudentCNN(
        n_leads=n_leads,
        n_classes=n_classes,
        size=args.student_size,
    ).to(device)

    # Print model size statistics
    print("Trainable parameters:", count_trainable_params(student), flush=True)
    print("Total parameters:", count_all_params(student), flush=True)
    print("Estimated model size (MB): {:.3f}".format(model_size_mb(student)), flush=True)

    # Compute class weights for loss reweighting
    class_weights = compute_class_weights(
        np.array(y_train),
        gamma=args.class_weight_gamma,
    ).to(device)
    print("Class weights:", class_weights.cpu().numpy(), flush=True)

    # Build training DataLoader with optional weighted sampling
    if args.use_weighted_sampler:
        train_sampler = build_weighted_sampler(
            np.array(y_train),
            power=args.sampler_power,
        )
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

    # Validation and test loaders should never shuffle
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

    # Cross-entropy loss with class weighting and label smoothing
    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # AdamW optimiser for stable baseline training
    optimizer = optim.AdamW(
        student.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    # Cosine annealing learning-rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )

    # Create checkpoint directory if needed
    ckpt_dir = os.path.dirname(args.student_ckpt)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    # Track best validation performance based on macro-AUC
    best_val_metric = -1.0
    best_epoch = 0
    best_time_sec = 0.0
    epochs_without_improvement = 0
    training_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Train for one full epoch
        train_loss, train_acc = train_one_epoch(
            student=student,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            grad_clip=args.grad_clip,
        )

        # Evaluate on validation set
        val_metrics = evaluate_classification(student, val_loader, device)
        val_acc = val_metrics["acc"]
        val_macro_auc = val_metrics["macro_auc"]
        val_macro_f1 = val_metrics["macro_f1"]
        val_weighted_f1 = val_metrics["weighted_f1"]

        # Step the scheduler once per epoch
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Print epoch summary
        print(
            "[Epoch {:03d}] train_loss={:.4f}, train_acc={:.4f}, "
            "val_acc={:.4f}, val_macro_auc={:.4f}, val_macro_f1={:.4f}, val_weighted_f1={:.4f}, "
            "lr={:.6f}, epoch_time={:.2f}s".format(
                epoch,
                train_loss,
                train_acc,
                val_acc,
                val_macro_auc,
                val_macro_f1,
                val_weighted_f1,
                current_lr,
                epoch_time,
            ),
            flush=True,
        )

        # Save checkpoint only if validation macro-AUC improves
        if not np.isnan(val_macro_auc) and val_macro_auc > best_val_metric:
            best_val_metric = val_macro_auc
            best_epoch = epoch
            best_time_sec = time.time() - training_start_time
            epochs_without_improvement = 0

            torch.save(
                {
                    # Learned model weights
                    "model_state_dict": student.state_dict(),
                    # Metadata needed for traceability / reproducibility
                    "classes": classes,
                    "n_leads": n_leads,
                    "student_size": args.student_size,
                    "seed": args.seed,
                    "lr": args.lr,
                    "epochs": args.epochs,
                    "patience": args.patience,
                    "weight_decay": args.weight_decay,
                    "grad_clip": args.grad_clip,
                    "class_weight_gamma": args.class_weight_gamma,
                    "sampler_power": args.sampler_power,
                    "label_smoothing": args.label_smoothing,
                    "disable_augmentation": args.disable_augmentation,
                    "use_weighted_sampler": args.use_weighted_sampler,
                    "best_val_macro_auc": best_val_metric,
                    "best_epoch": best_epoch,
                },
                args.student_ckpt,
            )
            print("  -> New best baseline student saved to", args.student_ckpt, flush=True)
        else:
            # Increase patience counter if validation did not improve
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.", flush=True)
                break

    # Ensure at least one best checkpoint exists before testing
    if not os.path.exists(args.student_ckpt):
        raise FileNotFoundError(
            f"Best baseline student checkpoint not found: {args.student_ckpt}"
        )

    # Reload best checkpoint for final test evaluation
    ckpt = torch.load(args.student_ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])

    print("Loaded best baseline student checkpoint for final test evaluation.", flush=True)
    print("Best validation macro-AUC: {:.4f}".format(best_val_metric), flush=True)
    print("Best epoch:", best_epoch, flush=True)
    print("Time to best model: {:.2f} seconds".format(best_time_sec), flush=True)
    print("Checkpoint size (MB): {:.3f}".format(checkpoint_size_mb(args.student_ckpt)), flush=True)
    print("Student size:", ckpt.get("student_size", "UNKNOWN"), flush=True)
    print("Checkpoint seed:", ckpt.get("seed", "UNKNOWN"), flush=True)

    # Final evaluation on held-out test set
    test_metrics = evaluate_classification(student, test_loader, device)
    print("Test accuracy (baseline student): {:.4f}".format(test_metrics["acc"]), flush=True)
    print("Test macro-AUC (baseline student): {:.4f}".format(test_metrics["macro_auc"]), flush=True)
    print("Test macro-F1 (baseline student): {:.4f}".format(test_metrics["macro_f1"]), flush=True)
    print("Test weighted-F1 (baseline student): {:.4f}".format(test_metrics["weighted_f1"]), flush=True)

    # Print full classification report if returned
    if "report" in test_metrics:
        print("Test classification report:", flush=True)
        print(test_metrics["report"], flush=True)

    # Print confusion matrix if returned
    if "confusion_matrix" in test_metrics:
        print("Test confusion matrix:", flush=True)
        print(test_metrics["confusion_matrix"], flush=True)

    # Reset seed before latency measurement for consistency
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Dummy input matching the model's expected input shape
    dummy_input = torch.randn(1, n_leads, X_train.shape[2], device=device)
    latency_sec = measure_latency(student, dummy_input, device)
    print("Inference latency: {:.4f} ms/sample".format(latency_sec * 1000.0), flush=True)

    # Return key test metrics for use by wrappers / experiment runners
    return {
        "acc": test_metrics["acc"],
        "macro_auc": test_metrics["macro_auc"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
    }


if __name__ == "__main__":
    results = main()