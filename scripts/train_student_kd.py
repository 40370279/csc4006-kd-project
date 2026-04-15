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

from src.data.dataset import ECGDataset
from src.data.augmentations import ECGAugment
from src.models.teacher_cnn import TeacherCNN
from src.models.student_cnn import StudentCNN
from src.utils.metrics import evaluate_classification
from src.utils.model_stats import (
    count_trainable_params,
    count_all_params,
    model_size_mb,
    checkpoint_size_mb,
    measure_latency,
)

DATA_PATH = os.path.join("processed", "ptbxl_500hz_10s.npz")
CHECKPOINT_DIR = "checkpoints"

DEFAULT_TEACHER_CKPT = os.path.join(CHECKPOINT_DIR, "teacher_cnn_best.pt")
DEFAULT_STUDENT_CKPT = os.path.join(CHECKPOINT_DIR, "student_kd_best.pt")


def set_seed(seed: int, deterministic: bool = True, warn_only: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = False

    try:
        if deterministic:
            torch.use_deterministic_algorithms(True, warn_only=warn_only)
        else:
            torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_splits(path: str = DATA_PATH) -> Tuple[np.ndarray, ...]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed .npz not found at {path}. Run scripts/preprocess_ptbxl.py first."
        )

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
    classes, counts = np.unique(labels, return_counts=True)

    if not np.array_equal(classes, np.arange(len(classes))):
        raise ValueError(
            f"Labels must be contiguous integers from 0..C-1, but got classes={classes}"
        )

    freq = counts.astype(np.float32) / counts.sum()
    inv = 1.0 / (freq + 1e-6)
    weights = np.power(inv, gamma)
    weights = weights / weights.mean()

    return torch.tensor(weights, dtype=torch.float32)


def build_weighted_sampler(labels: np.ndarray, power: float = 0.35) -> WeightedRandomSampler:
    class_counts = np.bincount(labels)
    class_weights = 1.0 / np.maximum(class_counts, 1).astype(np.float64)
    class_weights = np.power(class_weights, power)
    sample_weights = class_weights[labels]
    sample_weights = torch.as_tensor(sample_weights, dtype=torch.double)

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def get_alpha(epoch: int, total_epochs: int, alpha_start: float, alpha_end: float) -> float:
    if total_epochs <= 1:
        return alpha_end

    progress = (epoch - 1) / float(total_epochs - 1)
    cosine_progress = 0.5 * (1.0 - np.cos(np.pi * progress))
    return alpha_start + (alpha_end - alpha_start) * cosine_progress


class FeatureProjector(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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
    hard_loss = hard_criterion(student_logits, targets)

    log_p_student = F.log_softmax(student_logits / temperature, dim=1)
    p_teacher = F.softmax(teacher_logits / temperature, dim=1)

    soft_loss = F.kl_div(
        log_p_student,
        p_teacher,
        reduction="batchmean",
    ) * (temperature ** 2)

    student_logits_centered = student_logits - student_logits.mean(dim=1, keepdim=True)
    teacher_logits_centered = teacher_logits - teacher_logits.mean(dim=1, keepdim=True)
    logit_mse = F.mse_loss(student_logits_centered, teacher_logits_centered)

    student_pooled = student_pooled.flatten(1)
    teacher_pooled = teacher_pooled.flatten(1)

    teacher_proj = teacher_projector(teacher_pooled)

    student_norm = F.normalize(student_pooled, p=2, dim=1)
    teacher_norm = F.normalize(teacher_proj, p=2, dim=1)

    if student_norm.shape != teacher_norm.shape:
        raise ValueError(
            f"Feature shape mismatch: student_norm={student_norm.shape}, "
            f"teacher_norm={teacher_norm.shape}"
        )

    feature_mse = F.mse_loss(student_norm, teacher_norm)

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
    teacher.eval()
    student.train()
    teacher_projector.train()

    running_loss = 0.0
    running_hard = 0.0
    running_soft = 0.0
    running_logit_mse = 0.0
    running_feature_mse = 0.0
    correct = 0
    total = 0

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad():
            with torch.amp.autocast(device_type="cuda", enabled=use_amp):
                teacher_logits, teacher_features = teacher(X, return_features=True)
                teacher_pooled = teacher.pool_head(teacher_features)

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

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
            torch.nn.utils.clip_grad_norm_(teacher_projector.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

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
    parser = argparse.ArgumentParser(description="Train KD student on PTB-XL")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=8e-4)
    parser.add_argument("--epochs", type=int, default=70)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--class_weight_gamma", type=float, default=0.20)
    parser.add_argument("--sampler_power", type=float, default=0.35)
    parser.add_argument("--label_smoothing", type=float, default=0.03)

    parser.add_argument(
        "--student_size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
    )

    parser.add_argument("--alpha_start", type=float, default=0.20)
    parser.add_argument("--alpha_end", type=float, default=0.50)
    parser.add_argument("--temperature", type=float, default=4.0)
    parser.add_argument("--logit_mse_weight", type=float, default=0.03)
    parser.add_argument("--feature_mse_weight", type=float, default=0.03)
    parser.add_argument("--projector_lr_scale", type=float, default=1.0)

    parser.add_argument(
        "--teacher_ckpt",
        type=str,
        default=DEFAULT_TEACHER_CKPT,
    )

    parser.add_argument(
        "--student_ckpt",
        type=str,
        default=DEFAULT_STUDENT_CKPT,
    )

    parser.add_argument(
        "--use_weighted_sampler",
        action="store_true",
        help="Enable weighted sampling.",
    )

    parser.add_argument(
        "--disable_augmentation",
        action="store_true",
    )

    parser.add_argument("--no_deterministic", action="store_true")
    parser.add_argument("--strict_deterministic", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    deterministic = not args.no_deterministic
    warn_only = not args.strict_deterministic
    set_seed(args.seed, deterministic=deterministic, warn_only=warn_only)

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

    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_splits()
    n_leads = X_train.shape[1]
    n_classes = len(classes)

    print("Training set size:", X_train.shape[0], flush=True)
    print("Validation set size:", X_val.shape[0], flush=True)
    print("Test set size:", X_test.shape[0], flush=True)
    print("Number of leads:", n_leads, "classes:", n_classes, flush=True)
    print("Classes:", classes, flush=True)

    train_transform = None if args.disable_augmentation else ECGAugment()

    train_dataset = ECGDataset(X_train, y_train, transform=train_transform)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)

    common_loader_kwargs = {
        "num_workers": args.num_workers,
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": args.num_workers > 0,
        "worker_init_fn": seed_worker,
    }

    if not os.path.exists(args.teacher_ckpt):
        raise FileNotFoundError(
            f"Teacher checkpoint not found at {args.teacher_ckpt}. Train teacher first."
        )

    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device, weights_only=False)
    teacher_arch = teacher_ckpt.get("teacher_arch", "TeacherCNN")
    teacher_size = teacher_ckpt.get("teacher_size", "large")
    teacher_best_val = teacher_ckpt.get("best_val_macro_auc", None)

    if teacher_arch != "TeacherCNN":
        raise ValueError(
            f"Unsupported teacher_arch '{teacher_arch}'. Expected 'TeacherCNN'."
        )

    teacher = TeacherCNN(
        n_leads=n_leads,
        n_classes=n_classes,
        size=teacher_size,
    ).to(device)
    teacher.load_state_dict(teacher_ckpt["model_state_dict"])
    teacher.eval()

    for p in teacher.parameters():
        p.requires_grad = False

    student = StudentCNN(
        n_leads=n_leads,
        n_classes=n_classes,
        size=args.student_size,
    ).to(device)

    student_pooled_dim = student.feature_dim

    teacher_last_channels = teacher.SIZE_CONFIGS[teacher_size]["channels"][-1]
    teacher_pooled_dim = teacher_last_channels * 2

    teacher_projector = FeatureProjector(
        in_dim=teacher_pooled_dim,
        out_dim=student_pooled_dim,
    ).to(device)

    print("Trainable parameters (student):", count_trainable_params(student), flush=True)
    print("Total parameters (student):", count_all_params(student), flush=True)
    print("Estimated model size (MB): {:.3f}".format(model_size_mb(student)), flush=True)
    print("Teacher pooled dim:", teacher_pooled_dim, flush=True)
    print("Student pooled dim:", student_pooled_dim, flush=True)

    class_weights = compute_class_weights(
        np.array(y_train),
        gamma=args.class_weight_gamma,
    ).to(device)
    print("Class weights:", class_weights.cpu().numpy(), flush=True)

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

    hard_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = optim.AdamW(
        [
            {"params": student.parameters(), "lr": args.lr},
            {"params": teacher_projector.parameters(), "lr": args.lr * args.projector_lr_scale},
        ],
        weight_decay=args.weight_decay,
        betas=(0.9, 0.99),
    )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-5,
    )

    ckpt_dir = os.path.dirname(args.student_ckpt)
    if ckpt_dir:
        os.makedirs(ckpt_dir, exist_ok=True)

    best_val_metric = -1.0
    best_epoch = 0
    best_time_sec = 0.0
    epochs_without_improvement = 0
    training_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start_time = time.time()

        alpha = get_alpha(
            epoch=epoch,
            total_epochs=args.epochs,
            alpha_start=args.alpha_start,
            alpha_end=args.alpha_end,
        )

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

        val_metrics = evaluate_classification(student, val_loader, device)
        val_acc = val_metrics["acc"]
        val_macro_auc = val_metrics["macro_auc"]
        val_macro_f1 = val_metrics["macro_f1"]
        val_weighted_f1 = val_metrics["weighted_f1"]

        scheduler.step()
        epoch_time = time.time() - epoch_start_time
        current_lr = optimizer.param_groups[0]["lr"]

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

        if not np.isnan(val_macro_auc) and val_macro_auc > best_val_metric:
            best_val_metric = val_macro_auc
            best_epoch = epoch
            best_time_sec = time.time() - training_start_time
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "teacher_projector_state_dict": teacher_projector.state_dict(),
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
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.", flush=True)
                break

    if not os.path.exists(args.student_ckpt):
        raise FileNotFoundError(f"Best KD student checkpoint not found: {args.student_ckpt}")

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

    test_metrics = evaluate_classification(student, test_loader, device)
    print("Test accuracy (KD student): {:.4f}".format(test_metrics["acc"]), flush=True)
    print("Test macro-AUC (KD student): {:.4f}".format(test_metrics["macro_auc"]), flush=True)
    print("Test macro-F1 (KD student): {:.4f}".format(test_metrics["macro_f1"]), flush=True)
    print("Test weighted-F1 (KD student): {:.4f}".format(test_metrics["weighted_f1"]), flush=True)

    if "report" in test_metrics:
        print("Test classification report:", flush=True)
        print(test_metrics["report"], flush=True)

    if "confusion_matrix" in test_metrics:
        print("Test confusion matrix:", flush=True)
        print(test_metrics["confusion_matrix"], flush=True)

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dummy_input = torch.randn(1, n_leads, X_train.shape[2], device=device)
    latency_sec = measure_latency(student, dummy_input, device)
    print("Inference latency: {:.4f} ms/sample".format(latency_sec * 1000.0), flush=True)

    return {
        "acc": test_metrics["acc"],
        "macro_auc": test_metrics["macro_auc"],
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
    }


if __name__ == "__main__":
    results = main()