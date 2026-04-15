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
import torch.nn.functional as F

# Dataset wrapper for PTB-XL ECG arrays
from src.data.dataset import ECGDataset
# On-the-fly ECG augmentation used during training
from src.data.augmentations import ECGAugment
# High-capacity teacher architecture
from src.models.teacher_cnn import TeacherCNN
# Lightweight student architecture
from src.models.student_cnn import StudentCNN
# Evaluation helper returning accuracy, AUC, F1, report, confusion matrix, etc.
from src.utils.metrics import evaluate_classification
# Utility functions for model statistics and latency measurement
from src.utils.model_stats import (
    count_trainable_params,
    count_all_params,
    model_size_mb,
    checkpoint_size_mb,
    measure_latency,
)

# Path to preprocessed dataset splits
DATA_PATH = os.path.join("processed", "ptbxl_500hz_10s.npz")
# Directory for model checkpoints
CHECKPOINT_DIR = "checkpoints"

# Default teacher checkpoint path
DEFAULT_TEACHER_CKPT = os.path.join(CHECKPOINT_DIR, "teacher_cnn_best.pt")
# Default KD student checkpoint path
DEFAULT_STUDENT_CKPT = os.path.join(CHECKPOINT_DIR, "student_kd_best.pt")


def set_seed(seed: int, deterministic: bool = True, warn_only: bool = True):
    """
    Set Python, NumPy, and PyTorch seeds for reproducibility.

    Args:
        seed: random seed value
        deterministic: whether to force deterministic PyTorch behaviour
        warn_only: whether unsupported deterministic operations should warn instead of error
    """
    # Python random seed
    random.seed(seed)
    # NumPy random seed
    np.random.seed(seed)
    # Python hash seed for reproducible hashing
    os.environ["PYTHONHASHSEED"] = str(seed)

    # PyTorch CPU seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        # PyTorch CUDA seed for current GPU
        torch.cuda.manual_seed(seed)
        # PyTorch CUDA seed for all visible GPUs
        torch.cuda.manual_seed_all(seed)

    # Configure cuDNN determinism
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

    try:
        if deterministic:
            # Enforce deterministic algorithms where available
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        else:
            # Allow non-deterministic algorithms
            torch.use_deterministic_algorithms(False)
    except Exception:
        # Some versions / backends may not fully support this
        pass


def seed_worker(worker_id: int):
    """
    Seed each DataLoader worker deterministically.
    """
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_splits(path: str = DATA_PATH) -> Tuple[np.ndarray, ...]:
    """
    Load preprocessed PTB-XL train/validation/test splits.

    Args:
        path: path to processed .npz file

    Returns:
        Tuple containing:
            X_train, y_train, X_val, y_val, X_test, y_test, classes
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed .npz not found at {path}. Run scripts/preprocess_ptbxl.py first."
        )

    # Load arrays with memory mapping to reduce RAM pressure
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
        labels: integer-encoded class labels
        gamma: exponent controlling weighting strength

    Returns:
        Torch tensor of normalised class weights
    """
    classes, counts = np.unique(labels, return_counts=True)

    # Ensure labels are contiguous integers from 0..C-1
    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(
            f"Labels must be contiguous integers from 0..C-1, but got classes={classes}"
        )

    # Relative frequencies per class
    freq = counts.astype(np.float32) / counts.sum()
    # Inverse frequency weighting
    inv = 1.0 / (freq + 1e-6)
    # Smooth inverse weighting with exponent gamma
    weights = np.power(inv, gamma)
    # Normalise so mean weight is 1
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(labels: np.ndarray, power: float = 0.35) -> WeightedRandomSampler:
    """
    Build a WeightedRandomSampler to oversample minority classes.

    Args:
        labels: training labels
        power: exponent controlling sampling strength

    Returns:
        WeightedRandomSampler instance
    """
    # Count examples per class
    class_counts = np.bincount(labels)
    # Inverse frequency class weights
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
    # Smooth the weights
    class_weights = np.power(class_weights, power)
    # Assign each sample the weight of its class
    sample_weights = class_weights[labels]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_alpha(epoch: int, total_epochs: int, alpha_start: float, alpha_end: float) -> float:
    """
    Compute the KD hard-label weighting alpha using cosine interpolation.

    Args:
        epoch: current epoch index (1-based)
        total_epochs: total number of training epochs
        alpha_start: initial alpha value
        alpha_end: final alpha value

    Returns:
        Interpolated alpha for the current epoch
    """
    if total_epochs <= 1:
        return alpha_end

    # Linear progress through training
    progress = (epoch - 1) / float(total_epochs - 1)
    # Cosine-smoothed progress
    cosine_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
    # Interpolate from alpha_start to alpha_end
    return alpha_start + (alpha_end - alpha_start) * cosine_progress


class FeatureProjector(nn.Module):
    """
    Projection head mapping teacher pooled features into the student feature space.
    """
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        # Linear projection followed by batch normalisation
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Apply projection head
        return self.proj(x)


def kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    student_pooled: torch.Tensor,
    teacher_pooled: torch.Tensor,
    teacher_projector: nn.Module,
    targets: torch.Tensor,
    hard_criterion: nn.Module,
    alpha: float,
    temperature: float,
    feature_mse_weight: float,
    logit_mse_weight: float,
):
    """
    Compute full knowledge distillation loss.

    Components:
    - hard classification loss
    - soft-target KL divergence loss
    - centred logit MSE matching
    - feature-space MSE after teacher projection

    Returns:
        total_loss, hard_loss, soft_loss, logit_mse, feature_mse
    """
    # Standard supervised loss using true labels
    hard_loss = hard_criterion(student_logits, targets)

    # Temperature-scaled student log-probabilities
    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    # Temperature-scaled teacher probabilities
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)

    # KL divergence between teacher and student softened outputs
    soft_loss = F.kl_div(
        log_p_student,
        p_teacher,
        reduction="batchmean",
    ) * (temperature ** 2)

    # Centre logits before MSE to focus on relative structure rather than absolute offset
    student_logits_centered = student_logits - student_logits.mean(dim=1, keepdim=True)
    teacher_logits_centered = teacher_logits - teacher_logits.mean(dim=1, keepdim=True)
    # Direct logit matching term
    logit_mse = F.mse_loss(student_logits_centered, teacher_logits_centered)

    # Flatten pooled features if needed
    student_pooled = student_pooled.flatten(1)
    teacher_pooled = teacher_pooled.flatten(1)

    # Project teacher features into the student feature dimension
    teacher_proj = teacher_projector(teacher_pooled)

    # L2-normalise projected teacher and student features before MSE comparison
    student_norm = F.normalize(student_pooled, p=2, dim=1)
    teacher_norm = F.normalize(teacher_proj, p=2, dim=1)

    # Ensure projected teacher features and student features have matching shape
    if student_norm.shape != teacher_norm.shape:
        raise ValueError(
            f"Feature shape mismatch: student_norm={student_norm.shape}, "
            f"teacher_norm={teacher_norm.shape}"
        )

    # Feature distillation loss
    feature_mse = F.mse_loss(student_norm, teacher_norm)

    # Combined KD objective
    total_loss = (
        alpha * hard_loss
        + (1.0 - alpha) * soft_loss
        + logit_mse_weight * logit_mse
        + feature_mse_weight * feature_mse
    )

    return (
        total_loss,
        hard_loss.detach(),
        soft_loss.detach(),
        logit_mse.detach(),
        feature_mse.detach(),
    )


def train_one_epoch(
    teacher: nn.Module,
    student: nn.Module,
    teacher_projector: nn.Module,
    loader: DataLoader,
    hard_criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    alpha: float,
    temperature: float,
    grad_clip: float,
    feature_mse_weight: float,
    logit_mse_weight: float,
):
    """
    Train the KD student for one epoch using the frozen teacher.

    Args:
        teacher: pretrained teacher model (frozen)
        student: student model to train
        teacher_projector: projection head for teacher features
        loader: training DataLoader
        hard_criterion: supervised classification loss
        optimizer: optimiser over student + projector parameters
        device: cpu or cuda
        alpha: hard-label weighting
        temperature: distillation temperature
        grad_clip: gradient clipping threshold
        feature_mse_weight: weight for feature distillation loss
        logit_mse_weight: weight for logit matching loss

    Returns:
        Average total loss, average hard loss, average soft loss,
        average logit MSE, average feature MSE, training accuracy
    """
    # Teacher stays in eval mode during KD
    teacher.eval()
    # Student and projector are trainable
    student.train()
    teacher_projector.train()

    running_loss = 0.0
    running_hard = 0.0
    running_soft = 0.0
    running_logit_mse = 0.0
    running_feature_mse = 0.0
    correct = 0
    total = 0

    # Use AMP only on CUDA
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for X, y in loader:
        # Move batch to device
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        # Clear gradients
        optimizer.zero_grad(set_to_none=True)

        # Run teacher forward pass without gradients
        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                teacher_logits, teacher_features = teacher(X, return_features=True)
                teacher_pooled = teacher.pool_head(teacher_features)

        # Run student forward pass and compute KD loss
        with torch.amp.autocast(device_type="cuda", enabled=use_amp):
            student_logits, student_pooled = student(X, return_features=True)

            loss, hard_loss, soft_loss, logit_mse, feature_mse = kd_loss(
                student_logits=student_logits,
                teacher_logits=teacher_logits,
                student_pooled=student_pooled,
                teacher_pooled=teacher_pooled,
                teacher_projector=teacher_projector,
                targets=y,
                hard_criterion=hard_criterion,
                alpha=alpha,
                temperature=temperature,
                feature_mse_weight=feature_mse_weight,
                logit_mse_weight=logit_mse_weight,
            )

        # Backpropagate scaled loss
        scaler.scale(loss).backward()

        # Optional gradient clipping for stability
        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(teacher_projector.parameters(), grad_clip)

        # Optimiser step
        scaler.step(optimizer)
        scaler.update()

        # Accumulate losses and accuracy stats
        running_loss += loss.item() * X.size(0)
        running_hard += hard_loss.item() * X.size(0)
        running_soft += soft_loss.item() * X.size(0)
        running_logit_mse += logit_mse.item() * X.size(0)
        running_feature_mse += feature_mse.item() * X.size(0)

        preds = student_logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / float(total)
    avg_hard = running_hard / float(total)
    avg_soft = running_soft / float(total)
    avg_logit_mse = running_logit_mse / float(total)
    avg_feature_mse = running_feature_mse / float(total)
    acc = float(correct) / float(total)

    return avg_loss, avg_hard, avg_soft, avg_logit_mse, avg_feature_mse, acc


def parse_args():
    """
    Parse command-line arguments for KD training.
    """
    parser = argparse.ArgumentParser(description="Train KD student on PTB-XL")

    # Core optimisation hyperparameters
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)

    # Class imbalance / regularisation hyperparameters
    parser.add_argument("--class_weight_gamma", type=float, default=0.20)
    parser.add_argument("--sampler_power", type=float, default=0.35)
    parser.add_argument("--label_smoothing", type=float, default=0.03)

    # Student size choice
    parser.add_argument(
        "--student_size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
    )

    # KD-specific hyperparameters
    parser.add_argument("--alpha_start", type=float, default=0.20)
    parser.add_argument("--alpha_end", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--logit_mse_weight", type=float, default=0.03)
    parser.add_argument("--feature_mse_weight", type=float, default=0.03)
    parser.add_argument("--projector_lr_scale", type=float, default=1.0)

    # Teacher checkpoint path
    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        default=DEFAULT_TEACHER_CKPT,
    )

    # Student checkpoint output path
    parser.add_argument(
        "--student_ckpt",
        type=str,
        default=DEFAULT_STUDENT_CKPT,
    )

    # Optional weighted sampling
    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="Enable weighted sampling.",
    )

    # Optional augmentation disable switch
    parser.add_argument(
        "--disable_augmentation",
        action="store_true",
    )

    # Determinism control flags
    parser.add_argument("--no_deterministic", action="store_true")
    parser.add_argument("--strict_deterministic", action="store_true")

    return parser.parse_args()


def main():
    """
    Main pipeline for KD student training and final evaluation.
    """
    args = parse_args()

    # Configure deterministic behaviour and set seeds
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
            "KD hyperparameters: "
            f"alpha_start={args.alpha_start:.3f}, "
            f"alpha_end={args.alpha_end:.3f}, "
            f"temperature={args.temperature:.2f}, "
            f"logit_mse_weight={args.logit_mse_weight:.3f}, "
            f"feature_mse_weight={args.feature_mse_weight:.3f}, "
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
    print("Teacher checkpoint path:", args.teacher_ckpt, flush=True)
    print("Student checkpoint path:", args.student_ckpt, flush=True)

    # Load processed train / val / test splits
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

    # Shared DataLoader settings
    common_loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
    }

    # Ensure teacher checkpoint exists before training KD student
    if not os.path.exists(args.teacher_ckpt):
        raise FileNotFoundError(
            f"Teacher checkpoint not found at {args.teacher_ckpt}. Train teacher first."
        )

    # Load teacher checkpoint and metadata
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher_arch = teacher_ckpt.get("teacher_arch", "TeacherCNN")
    teacher_size = teacher_ckpt.get("teacher_size", "large")
    teacher_best_val = teacher_ckpt.get("best_val_macro_auc", None)

    # Validate teacher architecture
    if teacher_arch != "TeacherCNN":
        raise ValueError(
            f"Unsupported teacher_arch '{teacher_arch}'. Expected 'TeacherCNN'."
        )

    # Rebuild teacher model and load trained weights
    teacher = TeacherCNN(
        n_leads=n_leads,
        n_classes=n_classes,
        size=teacher_size,
    ).to(device)
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.eval()

    # Freeze teacher parameters so only student + projector are trained
    for p in teacher.parameters():
        p.requires_grad = False

    # Build student model
    student = StudentCNN(
        n_leads=n_leads,
        n_classes=n_classes,
        size=args.student_size,
    ).to(device)

    # Student pooled feature dimensionality
    student_pooled_dim = student.feature_dim

    # Teacher pooled feature dimensionality after statistics pooling (mean + std)
    teacher_last_channels = teacher.SIZE_CONFIGS[teacher_size]["channels"][-1]
    teacher_pooled_dim = teacher_last_channels * 2

    # Projection head to align teacher features to student feature size
    teacher_projector = FeatureProjector(
        in_dim=teacher_pooled_dim,
        out_dim=student_pooled_dim,
    ).to(device)

    # Print model statistics
    print("Trainable parameters (student):", count_trainable_params(student), flush=True)
    print("Total parameters (student):", count_all_params(student), flush=True)
    print("Estimated model size (MB): {:.3f}".format(model_size_mb(student)), flush=True)
    print("Teacher pooled dim:", teacher_pooled_dim, flush=True)
    print("Student pooled dim:", student_pooled_dim, flush=True)

    # Compute class weights for hard classification loss
    class_weights = compute_class_weights(
        np.array(y_train),
        gamma=args.class_weight_gamma,
    ).to(device)
    print("Class weights:", class_weights.cpu().numpy(), flush=True)

    # Build training loader using either weighted sampling or standard shuffling
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

    # Hard-label supervised loss component
    hard_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    # Optimiser updates both student and teacher projector
    optimizer = optim.AdamW(
        [
            {"params": student.parameters(), "lr": args.lr},
            {"params": teacher_projector.parameters(), "lr": args.lr * args.projector_lr_scale},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    # Cosine annealing learning-rate schedule
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )

    # Create checkpoint directory if required
    ckpt_dir = os.path.dirname(args.student_ckpt)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    # Tracking variables for best validation model
    best_val_metric = -1.0
    best_epoch = 0
    best_time_sec = 0.0
    epochs_without_improvement = 0
    training_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        # Compute epoch-specific alpha using cosine schedule
        alpha = get_alpha(
            epoch=epoch,
            total_epochs=args.epochs,
            alpha_start=args.alpha_start,
            alpha_end=args.alpha_end,
        )

        # Train one epoch of KD
        train_loss, train_hard, train_soft, train_logit_mse, train_feature_mse, train_acc = train_one_epoch(
            teacher=teacher,
            student=student,
            teacher_projector=teacher_projector,
            loader=train_loader,
            hard_criterion=hard_criterion,
            optimizer=optimizer,
            device=device,
            alpha=alpha,
            temperature=args.temperature,
            grad_clip=args.grad_clip,
            feature_mse_weight=args.feature_mse_weight,
            logit_mse_weight=args.logit_mse_weight,
        )

        # Validate student model
        val_metrics = evaluate_classification(student, val_loader, device)
        val_acc = val_metrics["acc"]
        val_macro_auc = val_metrics["macro_auc"]
        val_macro_f1 = val_metrics["macro_f1"]
        val_weighted_f1 = val_metrics["weighted_f1"]

        # Update scheduler after each epoch
        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

        # Print epoch summary
        print(
            "[Epoch {:03d}] train_loss={:.4f}, train_hard={:.4f}, train_soft={:.4f}, "
            "train_logit_mse={:.4f}, train_feature_mse={:.4f}, train_acc={:.4f}, alpha={:.3f}, "
            "val_acc={:.4f}, val_macro_auc={:.4f}, val_macro_f1={:.4f}, val_weighted_f1={:.4f}, "
            "lr={:.6f}, epoch_time={:.2f}s".format(
                epoch,
                train_loss,
                train_hard,
                train_soft,
                train_logit_mse,
                train_feature_mse,
                train_acc,
                alpha,
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
                    # Student weights
                    "model_state_dict": student.state_dict(),
                    # Projector weights required to reproduce training state
                    "teacher_projector_state_dict": teacher_projector.state_dict(),
                    # Metadata for reproducibility / traceability
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
                    "teacher_checkpoint": args.teacher_ckpt,
                    "teacher_arch": teacher_arch,
                    "teacher_size": teacher_size,
                    "teacher_best_val_macro_auc": teacher_best_val,
                    "alpha_start": args.alpha_start,
                    "alpha_end": args.alpha_end,
                    "temperature": args.temperature,
                    "logit_mse_weight": args.logit_mse_weight,
                    "feature_mse_weight": args.feature_mse_weight,
                    "best_val_macro_auc": best_val_metric,
                    "best_epoch": best_epoch,
                },
                args.student_ckpt,
            )
            print("  -> New best KD student saved to", args.student_ckpt, flush=True)
        else:
            # Increase patience counter if validation does not improve
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.", flush=True)
                break

    # Ensure best checkpoint exists before final evaluation
    if not os.path.exists(args.student_ckpt):
        raise FileNotFoundError(f"Best KD student checkpoint not found: {args.student_ckpt}")

    # Load best student checkpoint for held-out test evaluation
    ckpt = torch.load(args.student_ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])

    print("Loaded best KD student checkpoint for final test evaluation.", flush=True)
    print("Best validation macro-AUC: {:.4f}".format(best_val_metric), flush=True)
    print("Best epoch:", best_epoch, flush=True)
    print("Time to best model: {:.2f} seconds".format(best_time_sec), flush=True)
    print("Checkpoint size (MB): {:.3f}".format(checkpoint_size_mb(args.student_ckpt)), flush=True)
    print("Student size:", ckpt.get("student_size", "UNKNOWN"), flush=True)
    print("Teacher used:", ckpt.get("teacher_checkpoint", "UNKNOWN"), flush=True)
    print("Teacher arch:", ckpt.get("teacher_arch", "UNKNOWN"), flush=True)
    print("Teacher size:", ckpt.get("teacher_size", "UNKNOWN"), flush=True)
    print("Checkpoint seed:", ckpt.get("seed", "UNKNOWN"), flush=True)

    # Final evaluation on the test set
    test_metrics = evaluate_classification(student, test_loader, device)
    print("Test accuracy (KD student): {:.4f}".format(test_metrics["acc"]), flush=True)
    print("Test macro-AUC (KD student): {:.4f}".format(test_metrics["macro_auc"]), flush=True)
    print("Test macro-F1 (KD student): {:.4f}".format(test_metrics["macro_f1"]), flush=True)
    print("Test weighted-F1 (KD student): {:.4f}".format(test_metrics["weighted_f1"]), flush=True)

    # Print detailed classification report if available
    if "report" in test_metrics:
        print("Test classification report:", flush=True)
        print(test_metrics["report"], flush=True)

    # Print confusion matrix if available
    if "confusion_matrix" in test_metrics:
        print("Test confusion matrix:", flush=True)
        print(test_metrics["confusion_matrix"], flush=True)

    # Reset seeds before latency measurement for consistency
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Build dummy input matching model input shape for latency benchmarking
    dummy_input = torch.randn(1, n_leads, X_train.shape[2], device=device)
    latency_sec = measure_latency(student, dummy_input, device)
    print("Inference latency: {:.4f} ms/sample".format(latency_sec * 1000.0), flush=True)

    # Return key test metrics for wrappers / experiment runners
    return {
        "acc": test_metrics["acc"],
        "macro_auc": test_metrics["macro_auc"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
    }


if __name__ == "__main__":
    results = main()