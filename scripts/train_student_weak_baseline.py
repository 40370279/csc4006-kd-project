#train_student_weak_baseline.py
import os
import argparse
import time
import random
import numpy as np
from typing import Tuple

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

from src.data.dataset import ECGDataset
from src.data.augmentations import ECGAugment
from src.models.weak_student_cnn import WeakStudentCNN
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
STUDENT_CKPT = os.path.join(CHECKPOINT_DIR, "weak_student_baseline_best.pt")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def load_splits(path: str = DATA_PATH) -> Tuple[np.ndarray, ...]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            "Processed .npz not found at {}. Run scripts/preprocess_ptbxl.py first.".format(path)
        )

    print("DEBUG: loading splits from", path, flush=True)
    data = np.load(path, allow_pickle=True, mmap_mode="r")
    print("DEBUG: keys:", data.files, flush=True)

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_val = data["X_val"]
    y_val = data["y_val"]
    X_test = data["X_test"]
    y_test = data["y_test"]
    classes = data["classes"]

    print("DEBUG: shapes:", flush=True)
    print("  X_train:", X_train.shape, "y_train:", y_train.shape, flush=True)
    print("  X_val  :", X_val.shape, "y_val  :", y_val.shape, flush=True)
    print("  X_test :", X_test.shape, "y_test :", y_test.shape, flush=True)

    return X_train, y_train, X_val, y_val, X_test, y_test, classes


def compute_class_weights(labels: np.ndarray, gamma: float = 0.5) -> torch.Tensor:
    classes, counts = np.unique(labels, return_counts=True)
    freq = counts.astype(np.float32) / counts.sum()
    inv = 1.0 / (freq + 1e-6)
    weights = np.power(inv, gamma)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


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
        X = X.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        logits = model(X)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    avg_loss = running_loss / float(total)
    acc = float(correct) / float(total)

    return avg_loss, acc


def parse_args():
    parser = argparse.ArgumentParser(description="Train configurable weak baseline student on PTB-XL")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument(
        "--class_weight_gamma",
        type=float,
        default=0.5,
        help="Softness of inverse-frequency class weighting (default: 0.5)",
    )

    parser.add_argument(
        "--student_size",
        type=str,
        default="medium",
        choices=["small", "medium", "large"],
        help="Weak student capacity to use",
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="DataLoader workers",
    )

    parser.add_argument(
        "--student_ckpt",
        type=str,
        default=STUDENT_CKPT,
        help="Path to save the best weak baseline student checkpoint",
    )

    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)
    print("Seed:", args.seed, flush=True)
    print(
        "Training: batch_size={}, lr={}, epochs={}, patience={}, class_weight_gamma={}, weak_student_size={}, num_workers={}".format(
            args.batch_size,
            args.lr,
            args.epochs,
            args.patience,
            args.class_weight_gamma,
            args.student_size,
            args.num_workers,
        ),
        flush=True,
    )
    print("Student checkpoint path:", args.student_ckpt, flush=True)

    X_train, y_train, X_val, y_val, X_test, y_test, classes = load_splits()
    n_leads = X_train.shape[1]
    n_classes = len(classes)

    print("Training set size:", X_train.shape[0], flush=True)
    print("Validation set size:", X_val.shape[0], flush=True)
    print("Test set size:", X_test.shape[0], flush=True)
    print("Number of leads:", n_leads, "classes:", n_classes, flush=True)
    print("Classes:", classes, flush=True)

    # Matched exactly to KD script for fair comparison
    train_transform = ECGAugment(
        noise_std=0.005,
        scale_range=(0.95, 1.05),
        max_shift=50,
        lead_drop_prob=0.05,
    )

    train_dataset = ECGDataset(X_train, y_train, transform=train_transform)
    val_dataset = ECGDataset(X_val, y_val)
    test_dataset = ECGDataset(X_test, y_test)

    train_gen = torch.Generator()
    train_gen.manual_seed(args.seed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
        generator=train_gen,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
        worker_init_fn=seed_worker,
    )

    student = WeakStudentCNN(
        n_leads=n_leads,
        n_classes=n_classes,
        size=args.student_size,
    ).to(device)

    print("Trainable parameters:", count_trainable_params(student), flush=True)
    print("Total parameters:", count_all_params(student), flush=True)
    print("Estimated model size (MB): {:.3f}".format(model_size_mb(student)), flush=True)

    class_weights = compute_class_weights(np.array(y_train), gamma=args.class_weight_gamma).to(device)
    print("Class weights:", class_weights.cpu().numpy(), flush=True)

    criterion = nn.CrossEntropyLoss(weight=class_weights)

    optimizer = optim.Adam(student.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=2,
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

        train_loss, train_acc = train_one_epoch(
            student,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_metrics = evaluate_classification(student, val_loader, device)
        val_acc = val_metrics["acc"]
        val_macro_f1 = val_metrics["macro_f1"]
        val_weighted_f1 = val_metrics["weighted_f1"]

        scheduler.step(val_macro_f1)

        epoch_time = time.time() - epoch_start_time

        print(
            "[Epoch {:02d}] train_loss={:.4f}, train_acc={:.4f}, val_acc={:.4f}, val_macro_f1={:.4f}, val_weighted_f1={:.4f}, epoch_time={:.2f}s".format(
                epoch,
                train_loss,
                train_acc,
                val_acc,
                val_macro_f1,
                val_weighted_f1,
                epoch_time,
            ),
            flush=True,
        )

        if val_macro_f1 > best_val_metric:
            best_val_metric = val_macro_f1
            best_epoch = epoch
            best_time_sec = time.time() - training_start_time
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": student.state_dict(),
                    "classes": classes,
                    "n_leads": n_leads,
                    "student_size": args.student_size,
                    "class_weight_gamma": args.class_weight_gamma,
                    "seed": args.seed,
                    "best_val_macro_f1": best_val_metric,
                    "best_epoch": best_epoch,
                },
                args.student_ckpt,
            )
            print("  -> New best weak baseline student saved to", args.student_ckpt, flush=True)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= args.patience:
                print("Early stopping triggered.", flush=True)
                break

    if not os.path.exists(args.student_ckpt):
        raise FileNotFoundError(f"Best weak baseline student checkpoint not found: {args.student_ckpt}")

    ckpt = torch.load(args.student_ckpt, map_location=device, weights_only=False)
    student.load_state_dict(ckpt["model_state_dict"])
    print("Loaded best weak baseline student checkpoint for final test evaluation.", flush=True)

    print("Best validation macro-F1: {:.4f}".format(best_val_metric), flush=True)
    print("Best epoch:", best_epoch, flush=True)
    print("Time to best model: {:.2f} seconds".format(best_time_sec), flush=True)
    print("Checkpoint size (MB): {:.3f}".format(checkpoint_size_mb(args.student_ckpt)), flush=True)
    print("Weak student size:", ckpt.get("student_size", "UNKNOWN"), flush=True)
    print("Checkpoint seed:", ckpt.get("seed", "UNKNOWN"), flush=True)

    test_metrics = evaluate_classification(student, test_loader, device)
    print("Test accuracy (weak baseline student): {:.4f}".format(test_metrics["acc"]), flush=True)
    print("Test macro-F1 (weak baseline student): {:.4f}".format(test_metrics["macro_f1"]), flush=True)
    print("Test weighted-F1 (weak baseline student): {:.4f}".format(test_metrics["weighted_f1"]), flush=True)
    print("Test classification report:", flush=True)
    print(test_metrics["report"], flush=True)
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
        "macro_f1": test_metrics["macro_f1"],
        "weighted_f1": test_metrics["weighted_f1"],
    }


if __name__ == "__main__":
    results = main()