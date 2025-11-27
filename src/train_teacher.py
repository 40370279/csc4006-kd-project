#train_teacher.py
import os
import argparse
import math
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

# Try to use EMA if available (pip install torch-ema)
try:
    from torch_ema import ExponentialMovingAverage
    HAS_EMA = True
except ImportError:
    ExponentialMovingAverage = None
    HAS_EMA = False


DATA_PATH = os.path.join("processed", "ptbxl_500hz_10s.npz")
CHECKPOINT_DIR = "checkpoints"


# ---------------------------------------------------------
# Load PTB-XL splits
# ---------------------------------------------------------
def load_splits(path: str = DATA_PATH) -> Tuple[np.ndarray, ...]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Processed .npz not found at {}. Run src/preprocess_ptbxl.py first.".format(path)
        )
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    return (
        data["X_train"], data["y_train"],
        data["X_val"], data["y_val"],
        data["X_test"], data["y_test"],
        data["classes"],
    )


# ---------------------------------------------------------
# Softer class weights (biased a bit more towards accuracy)
# ---------------------------------------------------------
def compute_class_weights(labels: np.ndarray, gamma: float = 0.5) -> torch.Tensor:
    """
    Compute inverse-frequency class weights with a soft exponent gamma.

    freq_c = count_c / N
    w_c ∝ (1 / freq_c)^gamma

    gamma = 1.0  → very strong rebalancing (what you had before)
    gamma = 0.5  → softer; still helps minority, but less extreme,
                   generally better for overall accuracy.
    """
    classes, counts = np.unique(labels, return_counts=True)
    freq = counts.astype(np.float32) / counts.sum()
    inv = 1.0 / (freq + 1e-6)
    weights = np.power(inv, gamma)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


# ---------------------------------------------------------
# Cosine LR w/ warmup
# ---------------------------------------------------------
def build_cosine_with_warmup_scheduler(
    optimizer,
    total_epochs,
    warmup_epochs,
    min_lr_factor,
):

    def lr_lambda(epoch):
        # Linear warmup
        if epoch < warmup_epochs:
            return float(epoch + 1) / float(max(1, warmup_epochs))

        # Cosine decay
        progress = (epoch - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        progress = max(0.0, min(1.0, progress))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_factor + (1.0 - min_lr_factor) * cosine

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)


# ---------------------------------------------------------
# Training loop (no mixup, no weighted sampler)
# ---------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for X, y in loader:
        X, y = X.to(device), y.to(device)

        optimizer.zero_grad()

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return running_loss / float(total), float(correct) / float(total)


# ---------------------------------------------------------
# Argument parser
# ---------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train stable Teacher CNN on PTB-XL")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--warmup_epochs", type=int, default=5)
    parser.add_argument("--min_lr_factor", type=float, default=0.1)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    return parser.parse_args()


# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)
    print(
        "Training: batch_size={}, lr={}, epochs={}, patience={}, warmup_epochs={}, "
        "min_lr_factor={}, weight_decay={}".format(
            args.batch_size,
            args.lr,
            args.epochs,
            args.patience,
            args.warmup_epochs,
            args.min_lr_factor,
            args.weight_decay,
        ),
        flush=True,
    )

    # Load data
    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_splits()
    n_leads = X_train.shape[1]
    n_classes = len(classes)

    print("Training set size:", X_train.shape[0], flush=True)
    print("Validation set size:", X_val.shape[0], flush=True)
    print("Test set size:", X_test.shape[0], flush=True)
    print("Number of leads:", n_leads, "classes:", n_classes, flush=True)
    print("Classes:", classes, flush=True)

    # Augmentations
    train_transform = ECGAugment()

    train_dataset = ECGDataset(X_train, y_train, transform=train_transform)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)

    # NORMAL dataloaders (no sampler!)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    # Build model
    model = TeacherCNN(n_leads=n_leads, n_classes=n_classes).to(device)

    # Loss: softer class weights (gamma=0.5)
    class_weights = compute_class_weights(y_train, gamma=0.5).to(device)
    print("Class weights (gamma=0.5):", class_weights.cpu().numpy(), flush=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Scheduler
    scheduler = build_cosine_with_warmup_scheduler(
        optimizer,
        total_epochs=args.epochs,
        warmup_epochs=args.warmup_epochs,
        min_lr_factor=args.min_lr_factor,
    )

    # EMA (enabled after warmup for stability)
    ema = None
    if HAS_EMA:
        ema = ExponentialMovingAverage(model.parameters(), decay=0.999)
        print("EMA enabled", flush=True)
    else:
        print("EMA not available (torch-ema not installed)", flush=True)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    best_val_macro_f1 = 0.0
    epochs_without_improvement = 0

    # ---------------------------------------------------------
    # Training loop
    # ---------------------------------------------------------
    for epoch in range(1, args.epochs + 1):

        print("\nEpoch {}/{}".format(epoch, args.epochs), flush=True)

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        # EMA update AFTER warmup (crucial)
        if ema is not None and epoch > args.warmup_epochs:
            ema.update()

        # Evaluate (EMA version if available)
        if ema is not None and epoch > args.warmup_epochs:
            with ema.average_parameters():
                val_metrics = evaluate_classification(model, val_loader, device)
        else:
            val_metrics = evaluate_classification(model, val_loader, device)

        val_acc = val_metrics["acc"]
        val_macro_f1 = val_metrics["macro_f1"]

        scheduler.step()

        print(
            "[Epoch {:02d}] lr={:.6f}, train_loss={:.4f}, train_acc={:.4f}, "
            "val_acc={:.4f}, val_macro_f1={:.4f}".format(
                epoch,
                optimizer.param_groups[0]["lr"],
                train_loss,
                train_acc,
                val_acc,
                val_macro_f1,
            ),
            flush=True,
        )

        # Save best
        if val_macro_f1 > best_val_macro_f1:
            best_val_macro_f1 = val_macro_f1
            epochs_without_improvement = 0

            ckpt_path = os.path.join(CHECKPOINT_DIR, "teacher_cnn_best.pt")

            if ema is not None and epoch > args.warmup_epochs:
                with ema.average_parameters():
                    torch.save({
                        "model_state_dict": model.state_dict(),
                        "classes": classes,
                        "n_leads": n_leads,
                    }, ckpt_path)
            else:
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "classes": classes,
                    "n_leads": n_leads,
                }, ckpt_path)

            print("  → Saved NEW BEST model", flush=True)

        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping", flush=True)
                break

    # ---------------------------------------------------------
    # Final test evaluation
    # ---------------------------------------------------------
    ckpt_path = os.path.join(CHECKPOINT_DIR, "teacher_cnn_best.pt")
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])

    print("\nLoaded best checkpoint for final evaluation.", flush=True)

    test_metrics = evaluate_classification(model, test_loader, device)

    print("Test Accuracy: {:.4f}".format(test_metrics["acc"]), flush=True)
    print("Test Macro-F1: {:.4f}".format(test_metrics["macro_f1"]), flush=True)
    print("Classification Report:", flush=True)
    print(test_metrics["report"], flush=True)
    print("Confusion Matrix:", flush=True)
    print(test_metrics["confusion_matrix"], flush=True)


if __name__ == "__main__":
    main()
