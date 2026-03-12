import os
import argparse
import math
import time
import copy
import numpy as np
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import ECGDataset
from src.data.augmentations import ECGAugment
from src.models.teacher_cnn import TeacherCNN
from src.utils.metrics import evaluate_classification
from src.utils.model_stats import (
    count_trainable_params,
    count_all_params,
    model_size_mb,
    checkpoint_size_mb,
    measure_latency,
)

try:
    from torch_ema import ExponentialMovingAverage
    HAS_EMA = True
except ImportError:
    ExponentialMovingAverage = None
    HAS_EMA = False


DATA_PATH = os.path.join("processed", "ptbxl_500hz_10s.npz")
CHECKPOINT_DIR = "checkpoints"

TEACHER_CKPT = os.path.join(CHECKPOINT_DIR, "teacher_cnn_best.pt")


def load_splits(path: str = DATA_PATH) -> Tuple[np.ndarray, ...]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed .npz not found at {path}. Run scripts/preprocess_ptbxl.py first."
        )

    data = np.load(path, allow_pickle=True, mmap_mode="r")
    return (
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"],
        data["classes"],
    )


def compute_class_weights(labels: np.ndarray, gamma: float = 0.5) -> torch.Tensor:
    """
    Softer inverse-frequency class weights.

    gamma = 1.0 -> stronger rebalancing
    gamma = 0.5 -> softer rebalancing
    """
    classes, counts = np.unique(labels, return_counts=True)
    freq = counts.astype(np.float32) / counts.sum()
    inv = 1.0 / (freq + 1e-6)
    weights = np.power(inv, gamma)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def build_cosine_with_warmup_scheduler(
    optimizer: optim.Optimizer,
    total_epochs: int,
    warmup_epochs: int,
    min_lr_factor: float,
):
    def lr_lambda(epoch: int):
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

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
    scaler: torch.cuda.amp.GradScaler,
    device: torch.device,
    grad_clip: float,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    use_amp = device.type == "cuda"

    for X, y in loader:
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = model(X)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()

        if grad_clip is not None and grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / float(total), float(correct) / float(total)


def parse_args():
    parser = argparse.ArgumentParser(description="Train improved Teacher CNN on PTB-XL")

    parser.add_argument("--batch_size", type=int, default=48)
    parser.add_argument("--lr", type=float, default=2e-4)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--patience", type=int, default=20)

    parser.add_argument("--warmup_epochs", type=int, default=8)
    parser.add_argument("--min_lr_factor", type=float, default=0.08)

    parser.add_argument("--weight_decay", type=float, default=3e-4)
    parser.add_argument("--class_weight_gamma", type=float, default=0.5)
    parser.add_argument("--label_smoothing", type=float, default=0.05)

    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=2)

    return parser.parse_args()


def save_checkpoint(
    path: str,
    model: nn.Module,
    classes,
    n_leads: int,
    best_val_macro_f1: float,
    epoch: int,
    args,
):
    state_dict = copy.deepcopy(model.state_dict())

    torch.save(
        {
            "model_state_dict": state_dict,
            "classes": classes,
            "n_leads": n_leads,
            "teacher_arch": "TeacherCNN_v2",
            "best_val_macro_f1": best_val_macro_f1,
            "epoch": epoch,
            "train_args": vars(args),
        },
        path,
    )


def main():
    args = parse_args()

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)
    print(
        "Training: batch_size={}, lr={}, epochs={}, patience={}, warmup_epochs={}, "
        "min_lr_factor={}, weight_decay={}, class_weight_gamma={}, label_smoothing={}, "
        "grad_clip={}, num_workers={}".format(
            args.batch_size,
            args.lr,
            args.epochs,
            args.patience,
            args.warmup_epochs,
            args.min_lr_factor,
            args.weight_decay,
            args.class_weight_gamma,
            args.label_smoothing,
            args.grad_clip,
            args.num_workers,
        ),
        flush=True,
    )

    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_splits()
    n_leads = X_train.shape[1]
    n_classes = len(classes)

    print("Training set size:", X_train.shape[0], flush=True)
    print("Validation set size:", X_val.shape[0], flush=True)
    print("Test set size:", X_test.shape[0], flush=True)
    print("Number of leads:", n_leads, "classes:", n_classes, flush=True)
    print("Classes:", classes, flush=True)

    train_transform = ECGAugment()

    train_dataset = ECGDataset(X_train, y_train, transform=train_transform)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    model = TeacherCNN(n_leads=n_leads, n_classes=n_classes).to(device)

    print("Trainable parameters:", count_trainable_params(model), flush=True)
    print("Total parameters:", count_all_params(model), flush=True)
    print("Estimated model size (MB): {:.3f}".format(model_size_mb(model)), flush=True)

    class_weights = compute_class_weights(y_train, gamma=args.class_weight_gamma).to(device)
    print("Class weights:", class_weights.cpu().numpy(), flush=True)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=args.label_smoothing,
    )

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    scheduler = build_cosine_with_warmup_scheduler(
        optimizer=optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr_factor=args.min_lr_factor,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    ema = None
    if HAS_EMA:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        print("EMA enabled", flush=True)
    else:
        print("EMA not available (torch-ema not installed)", flush=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    best_val_macro_f1 = 0.0
    best_epoch = 0
    best_time_sec = 0.0
    epochs_without_improvement = 0
    training_start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}", flush=True)
        epoch_start_time = time.time()

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            scaler=scaler,
            device=device,
            grad_clip=args.grad_clip,
        )

        if ema is not None and epoch > args.warmup_epochs:
            ema.update()

        if ema is not None and epoch > args.warmup_epochs:
            with ema.average_parameters():
                val_metrics = evaluate_classification(model, val_loader, device)
        else:
            val_metrics = evaluate_classification(model, val_loader, device)

        val_acc = val_metrics["acc"]
        val_macro_f1 = val_metrics["macro_f1"]
        val_weighted_f1 = val_metrics["weighted_f1"]

        scheduler.step()
        epoch_time = time.time() - epoch_start_time

        print(
            "[Epoch {:03d}] lr={:.6f}, train_loss={:.4f}, train_acc={:.4f}, "
            "val_acc={:.4f}, val_macro_f1={:.4f}, val_weighted_f1={:.4f}, "
            "epoch_time={:.2f}s".format(
                epoch,
                optimizer.param_groups[0]["lr"],
                train_loss,
                train_acc,
                val_acc,
                val_macro_f1,
                val_weighted_f1,
                epoch_time,
            ),
            flush=True,
        )

        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            best_epoch = epoch
            best_time_sec = time.time() - training_start_time
            epochs_without_improvement = 0

            if ema is not None and epoch > args.warmup_epochs:
                with ema.average_parameters():
                    save_checkpoint(
                        path=TEACHER_CKPT,
                        model=model,
                        classes=classes,
                        n_leads=n_leads,
                        best_val_macro_f1=best_val_macro_f1,
                        epoch=epoch,
                        args=args,
                    )
            else:
                save_checkpoint(
                    path=TEACHER_CKPT,
                    model=model,
                    classes=classes,
                    n_leads=n_leads,
                    best_val_macro_f1=best_val_macro_f1,
                    epoch=epoch,
                    args=args,
                )

            print("  -> Saved NEW BEST model to", TEACHER_CKPT, flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping", flush=True)
                break

    if not os.path.exists(TEACHER_CKPT):
        raise FileNotFoundError(f"Best teacher checkpoint not found: {TEACHER_CKPT}")

    ckpt = torch.load(TEACHER_CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\nLoaded best checkpoint for final evaluation.", flush=True)
    print("Best validation macro-F1: {:.4f}".format(best_val_macro_f1), flush=True)
    print("Best epoch:", best_epoch, flush=True)
    print("Time to best model: {:.2f} seconds".format(best_time_sec), flush=True)
    print("Checkpoint size (MB): {:.3f}".format(checkpoint_size_mb(TEACHER_CKPT)), flush=True)

    test_metrics = evaluate_classification(model, test_loader, device)

    print("Test Accuracy: {:.4f}".format(test_metrics["acc"]), flush=True)
    print("Test Macro-F1: {:.4f}".format(test_metrics["macro_f1"]), flush=True)
    print("Test Weighted-F1: {:.4f}".format(test_metrics["weighted_f1"]), flush=True)
    print("Classification Report:", flush=True)
    print(test_metrics["report"], flush=True)
    print("Confusion Matrix:", flush=True)
    print(test_metrics["confusion_matrix"], flush=True)

    dummy_input = torch.randn(1, n_leads, X_train.shape[2], device=device)
    latency_sec = measure_latency(model, dummy_input, device)
    print("Inference latency: {:.4f} ms/sample".format(latency_sec * 1000.0), flush=True)


if __name__ == "__main__":
    main()