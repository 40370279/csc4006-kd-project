import os
import json
import argparse
import random
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from src.models.student_cnn import StudentCNN
from src.utils.metrics import evaluate_classification


DATA_PATH = os.path.join("processed", "ptbxl_500hz_10s.npz")


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def load_test_split(path: str = DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run preprocessing first."
        )

    data = np.load(path, allow_pickle=True, mmap_mode="r")
    return data["X_test"], data["y_test"], data["classes"]


class CorruptedECGDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        corruption: str = "clean",
        severity: float = 0.0,
        seed: int = 42,
    ):
        self.X = X
        self.y = y
        self.corruption = corruption
        self.severity = severity
        self.seed = seed

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = np.asarray(self.X[idx], dtype=np.float32).copy()
        y = int(self.y[idx])

        rng = np.random.default_rng(self.seed + idx)
        x = apply_corruption(
            x=x,
            corruption=self.corruption,
            severity=self.severity,
            rng=rng,
        )

        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)
        return x, y


def apply_corruption(
    x: np.ndarray,
    corruption: str,
    severity: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    x shape: (C, T)
    """
    if corruption == "clean":
        return x

    if corruption == "gaussian_noise":
        # severity is std of added noise relative to already-normalised signal scale
        noise = rng.normal(loc=0.0, scale=severity, size=x.shape).astype(np.float32)
        return x + noise

    if corruption == "amplitude_scale":
        # severity is max deviation from 1.0, sample one global factor
        scale = rng.uniform(1.0 - severity, 1.0 + severity)
        return x * np.float32(scale)

    if corruption == "lead_dropout":
        # severity = proportion of leads to zero out
        c, _ = x.shape
        num_drop = max(1, int(round(c * severity)))
        drop_idx = rng.choice(c, size=num_drop, replace=False)
        x[drop_idx, :] = 0.0
        return x

    if corruption == "time_mask":
        # severity = proportion of time axis to mask
        _, t = x.shape
        mask_len = max(1, int(round(t * severity)))
        if mask_len >= t:
            x[:, :] = 0.0
            return x
        start = rng.integers(0, t - mask_len + 1)
        x[:, start:start + mask_len] = 0.0
        return x

    raise ValueError(f"Unknown corruption type: {corruption}")


def build_loader(
    X_test: np.ndarray,
    y_test: np.ndarray,
    corruption: str,
    severity: float,
    batch_size: int,
    num_workers: int,
    seed: int,
) -> DataLoader:
    dataset = CorruptedECGDataset(
        X=X_test,
        y=y_test,
        corruption=corruption,
        severity=severity,
        seed=seed,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def load_student_checkpoint(
    ckpt_path: str,
    n_leads: int,
    n_classes: int,
    device: torch.device,
):
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    student_size = ckpt.get("student_size", "small")

    model = StudentCNN(
        n_leads=n_leads,
        n_classes=n_classes,
        size=student_size,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def summarise_drop(clean_metrics: Dict, corrupt_metrics: Dict) -> Dict[str, float]:
    return {
        "acc_drop": clean_metrics["acc"] - corrupt_metrics["acc"],
        "macro_auc_drop": clean_metrics["macro_auc"] - corrupt_metrics["macro_auc"],
        "macro_f1_drop": clean_metrics["macro_f1"] - corrupt_metrics["macro_f1"],
        "weighted_f1_drop": clean_metrics["weighted_f1"] - corrupt_metrics["weighted_f1"],
    }


def print_metrics_block(title: str, metrics: Dict):
    print(title, flush=True)
    print(f"  Accuracy    : {metrics['acc']:.4f}", flush=True)
    print(f"  Macro-AUC   : {metrics['macro_auc']:.4f}", flush=True)
    print(f"  Macro-F1    : {metrics['macro_f1']:.4f}", flush=True)
    print(f"  Weighted-F1 : {metrics['weighted_f1']:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Robustness evaluation for baseline vs KD students")
    parser.add_argument("--baseline_ckpt", type=str, required=True)
    parser.add_argument("--kd_ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_json", type=str, required=True)
    args = parser.parse_args()

    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)
    print("Seed:", args.seed, flush=True)

    X_test, y_test, classes = load_test_split()
    n_leads = X_test.shape[1]
    n_classes = len(classes)

    print("Test size:", X_test.shape[0], flush=True)
    print("Leads:", n_leads, "Classes:", n_classes, flush=True)
    print("Classes:", classes, flush=True)

    baseline_model, baseline_ckpt = load_student_checkpoint(
        args.baseline_ckpt, n_leads, n_classes, device
    )
    kd_model, kd_ckpt = load_student_checkpoint(
        args.kd_ckpt, n_leads, n_classes, device
    )

    print("Loaded baseline checkpoint:", args.baseline_ckpt, flush=True)
    print("Loaded KD checkpoint:", args.kd_ckpt, flush=True)
    print("Baseline student size:", baseline_ckpt.get("student_size", "UNKNOWN"), flush=True)
    print("KD student size:", kd_ckpt.get("student_size", "UNKNOWN"), flush=True)

    settings: List[Tuple[str, float]] = [
        ("clean", 0.0),
        ("gaussian_noise", 0.05),
        ("gaussian_noise", 0.10),
        ("amplitude_scale", 0.10),
        ("amplitude_scale", 0.20),
        ("lead_dropout", 1.0 / 12.0),
        ("lead_dropout", 2.0 / 12.0),
        ("time_mask", 0.05),
        ("time_mask", 0.10),
    ]

    results = {
        "seed": args.seed,
        "classes": [str(c) for c in classes],
        "baseline_ckpt": args.baseline_ckpt,
        "kd_ckpt": args.kd_ckpt,
        "settings": [],
    }

    clean_baseline = None
    clean_kd = None

    for corruption, severity in settings:
        print("\n==================================================", flush=True)
        print(f"Condition: corruption={corruption}, severity={severity:.4f}", flush=True)

        loader = build_loader(
            X_test=X_test,
            y_test=y_test,
            corruption=corruption,
            severity=severity,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            seed=args.seed,
        )

        baseline_metrics = evaluate_classification(baseline_model, loader, device)
        kd_metrics = evaluate_classification(kd_model, loader, device)

        print_metrics_block("Baseline metrics:", baseline_metrics)
        print_metrics_block("KD metrics:", kd_metrics)

        if corruption == "clean":
            clean_baseline = baseline_metrics
            clean_kd = kd_metrics
            baseline_drop = {
                "acc_drop": 0.0,
                "macro_auc_drop": 0.0,
                "macro_f1_drop": 0.0,
                "weighted_f1_drop": 0.0,
            }
            kd_drop = {
                "acc_drop": 0.0,
                "macro_auc_drop": 0.0,
                "macro_f1_drop": 0.0,
                "weighted_f1_drop": 0.0,
            }
        else:
            baseline_drop = summarise_drop(clean_baseline, baseline_metrics)
            kd_drop = summarise_drop(clean_kd, kd_metrics)

        kd_advantage = {
            "acc_advantage": kd_metrics["acc"] - baseline_metrics["acc"],
            "macro_auc_advantage": kd_metrics["macro_auc"] - baseline_metrics["macro_auc"],
            "macro_f1_advantage": kd_metrics["macro_f1"] - baseline_metrics["macro_f1"],
            "weighted_f1_advantage": kd_metrics["weighted_f1"] - baseline_metrics["weighted_f1"],
            "acc_drop_advantage": baseline_drop["acc_drop"] - kd_drop["acc_drop"],
            "macro_auc_drop_advantage": baseline_drop["macro_auc_drop"] - kd_drop["macro_auc_drop"],
            "macro_f1_drop_advantage": baseline_drop["macro_f1_drop"] - kd_drop["macro_f1_drop"],
            "weighted_f1_drop_advantage": baseline_drop["weighted_f1_drop"] - kd_drop["weighted_f1_drop"],
        }

        print("KD advantage over baseline:", flush=True)
        print(f"  Accuracy advantage           : {kd_advantage['acc_advantage']:.4f}", flush=True)
        print(f"  Macro-AUC advantage          : {kd_advantage['macro_auc_advantage']:.4f}", flush=True)
        print(f"  Macro-F1 advantage           : {kd_advantage['macro_f1_advantage']:.4f}", flush=True)
        print(f"  Weighted-F1 advantage        : {kd_advantage['weighted_f1_advantage']:.4f}", flush=True)
        print(f"  Macro-F1 drop advantage      : {kd_advantage['macro_f1_drop_advantage']:.4f}", flush=True)
        print(f"  Weighted-F1 drop advantage   : {kd_advantage['weighted_f1_drop_advantage']:.4f}", flush=True)

        results["settings"].append(
            {
                "corruption": corruption,
                "severity": severity,
                "baseline": {
                    "acc": baseline_metrics["acc"],
                    "macro_auc": baseline_metrics["macro_auc"],
                    "macro_f1": baseline_metrics["macro_f1"],
                    "weighted_f1": baseline_metrics["weighted_f1"],
                    "drop_vs_clean": baseline_drop,
                },
                "kd": {
                    "acc": kd_metrics["acc"],
                    "macro_auc": kd_metrics["macro_auc"],
                    "macro_f1": kd_metrics["macro_f1"],
                    "weighted_f1": kd_metrics["weighted_f1"],
                    "drop_vs_clean": kd_drop,
                },
                "kd_advantage": kd_advantage,
            }
        )

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\nSaved robustness results to:", args.output_json, flush=True)


if __name__ == "__main__":
    main()