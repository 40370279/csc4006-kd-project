"""
Plot aggregated ROC curves for the teacher, baseline student, and KD student.

This script is intended to live in:

    src/utils/plot_roc_curves.py

It still reads and writes paths relative to the PROJECT ROOT, not relative to
src/utils/. Therefore the default behaviour remains:

    processed/ptbxl_500hz_10s.npz
    checkpoints/teacher/...
    checkpoints/student_baseline/...
    checkpoints/student_kd/...
    results/roc_curves/

Recommended usage from the project root:

    python -m src.utils.plot_roc_curves

or:

    python src/utils/plot_roc_curves.py
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Project-root setup
# ---------------------------------------------------------------------------
# This file now lives in src/utils/.
# parents[0] = src/utils
# parents[1] = src
# parents[2] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Ensure imports such as `from src.models.teacher_cnn import TeacherCNN`
# work even when the script is run directly as:
#     python src/utils/plot_roc_curves.py
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Compatibility shim for older checkpoints / NumPy pickling edge cases
# ---------------------------------------------------------------------------
# Some older checkpoint files may reference numpy._core even where the runtime
# exposes numpy.core. This mirrors the compatibility behaviour used in the
# original script and avoids loading failures on Kelvin2.
import numpy.core  # noqa: E402

sys.modules["numpy._core"] = numpy.core


# ---------------------------------------------------------------------------
# Matplotlib setup
# ---------------------------------------------------------------------------
# Use a non-interactive backend because this is normally run on Kelvin2/SLURM
# or other headless environments.
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from src.models.teacher_cnn import TeacherCNN  # noqa: E402
from src.models.student_cnn import StudentCNN  # noqa: E402


# ---------------------------------------------------------------------------
# Default paths
# ---------------------------------------------------------------------------
# Defaults are deliberately stored as project-root-relative strings. They are
# resolved later using resolve_project_path(), so the script behaves the same
# whether it is run from the project root or from another working directory.
DATA_PATH = "processed/ptbxl_500hz_10s.npz"

# Do NOT read classes from the NPZ on Kelvin2; that field may crash there.
FIXED_CLASSES = np.array(["CD", "HYP", "MI", "NORM", "STTC"], dtype=object)

DEFAULT_TEACHER_CKPTS = [
    "checkpoints/teacher/teacher_cnn_seed42.pt",
    "checkpoints/teacher/teacher_cnn_seed123.pt",
    "checkpoints/teacher/teacher_cnn_seed999.pt",
]

DEFAULT_BASELINE_CKPTS = [
    "checkpoints/student_baseline/student_baseline_seed42.pt",
    "checkpoints/student_baseline/student_baseline_seed123.pt",
    "checkpoints/student_baseline/student_baseline_seed999.pt",
]

DEFAULT_KD_CKPTS = [
    "checkpoints/student_kd/student_kd_seed42.pt",
    "checkpoints/student_kd/student_kd_seed123.pt",
    "checkpoints/student_kd/student_kd_seed999.pt",
]


def resolve_project_path(path: str | os.PathLike) -> Path:
    """
    Resolve a path relative to the project root unless it is already absolute.

    This is the key change that keeps outputs and inputs in the same locations
    after moving this file from scripts/ into src/utils/.
    """
    path = Path(path)

    if path.is_absolute():
        return path

    return PROJECT_ROOT / path


def load_splits(path: str | os.PathLike = DATA_PATH):
    """
    Load preprocessed PTB-XL train/validation/test splits.

    The classes are returned from FIXED_CLASSES rather than from the NPZ file
    because the class field can cause loading issues on Kelvin2.
    """
    path = resolve_project_path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            "Run scripts/preprocess_ptbxl.py first."
        )

    data = np.load(path, allow_pickle=True)

    return (
        data["X_train"],
        data["y_train"],
        data["X_val"],
        data["y_val"],
        data["X_test"],
        data["y_test"],
        FIXED_CLASSES,
    )


def safe_torch_load(path: str | os.PathLike, device: torch.device):
    """
    Load a checkpoint safely across PyTorch versions.

    PyTorch 2.6 defaults to weights_only=True, which can break older
    checkpoint files that contain full pickle metadata. Setting
    weights_only=False preserves compatibility with the existing project
    checkpoints.
    """
    path = resolve_project_path(path)
    return torch.load(path, map_location=device, weights_only=False)


def load_teacher_model(
    ckpt_path: str | os.PathLike,
    n_leads: int,
    n_classes: int,
    device: torch.device,
) -> Tuple[TeacherCNN, dict]:
    """
    Load a TeacherCNN from a checkpoint and switch it to evaluation mode.
    """
    ckpt = safe_torch_load(ckpt_path, device)

    teacher_size = ckpt.get("teacher_size", "large")
    model = TeacherCNN(
        n_leads=n_leads,
        n_classes=n_classes,
        size=teacher_size,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def load_student_model(
    ckpt_path: str | os.PathLike,
    n_leads: int,
    n_classes: int,
    device: torch.device,
) -> Tuple[StudentCNN, dict]:
    """
    Load a StudentCNN from a checkpoint and switch it to evaluation mode.
    """
    ckpt = safe_torch_load(ckpt_path, device)

    student_size = ckpt.get("student_size", "medium")
    model = StudentCNN(
        n_leads=n_leads,
        n_classes=n_classes,
        size=student_size,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    return model, ckpt


def softmax_np(logits: np.ndarray) -> np.ndarray:
    """
    NumPy softmax implementation.

    Kept for completeness/utility, although model inference currently uses
    torch.softmax directly.
    """
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def infer_probs_numpy(
    model: torch.nn.Module,
    X: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """
    Run batched inference and return class probabilities as a NumPy array.
    """
    probs_list = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))

            xb = torch.from_numpy(np.asarray(X[start:end])).float().to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

            probs_list.append(probs)

    return np.concatenate(probs_list, axis=0)


def binary_roc_curve(
    y_true_binary: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute a one-vs-rest ROC curve using pure NumPy.

    Parameters
    ----------
    y_true_binary:
        Binary labels where 1 is the positive class and 0 is the negative class.
    y_score:
        Predicted probability or score for the positive class.

    Returns
    -------
    fpr, tpr:
        False-positive-rate and true-positive-rate arrays with endpoints
        included.
    """
    y_true_binary = np.asarray(y_true_binary).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    # Sort examples by descending score.
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true_binary[order]
    y_score_sorted = y_score[order]

    positives = np.sum(y_true_sorted == 1)
    negatives = np.sum(y_true_sorted == 0)

    # If a class is absent, return a diagonal fallback instead of crashing.
    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    # Keep only points where the score changes. This avoids redundant points
    # while preserving the ROC curve shape.
    distinct = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct, len(y_score_sorted) - 1]

    tps = tps[threshold_idxs]
    fps = fps[threshold_idxs]

    tpr = tps / positives
    fpr = fps / negatives

    # Add origin.
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]

    # Ensure the curve ends at (1, 1).
    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr = np.r_[fpr, 1.0]
        tpr = np.r_[tpr, 1.0]

    return fpr, tpr


def auc_trapezoid(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute area under a curve using trapezoidal integration.
    """
    return float(np.trapz(y, x))


def one_vs_rest_macro_auc(
    targets: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
) -> Tuple[float, Dict[int, float]]:
    """
    Compute macro-AUC from one-vs-rest class AUCs.
    """
    per_class_auc = {}

    for i in range(n_classes):
        y_true_binary = (targets == i).astype(np.int64)
        y_score = probs[:, i]

        fpr, tpr = binary_roc_curve(y_true_binary, y_score)
        per_class_auc[i] = auc_trapezoid(fpr, tpr)

    macro_auc = float(np.mean(list(per_class_auc.values())))

    return macro_auc, per_class_auc


def compute_macro_roc_from_probs(
    targets: np.ndarray,
    probs: np.ndarray,
    n_classes: int,
) -> Tuple[np.ndarray, np.ndarray, float, Dict[int, float]]:
    """
    Compute a macro-average ROC curve by interpolating one-vs-rest curves.

    This is used for plotting a single macro-average curve in addition to the
    individual one-vs-rest class curves.
    """
    fpr_dict = {}
    tpr_dict = {}
    auc_dict = {}

    for i in range(n_classes):
        y_true_binary = (targets == i).astype(np.int64)
        y_score = probs[:, i]

        fpr, tpr = binary_roc_curve(y_true_binary, y_score)

        fpr_dict[i] = fpr
        tpr_dict[i] = tpr
        auc_dict[i] = auc_trapezoid(fpr, tpr)

    all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(n_classes)]))
    mean_tpr = np.zeros_like(all_fpr)

    for i in range(n_classes):
        mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])

    mean_tpr /= n_classes
    macro_roc_auc = auc_trapezoid(all_fpr, mean_tpr)

    return all_fpr, mean_tpr, macro_roc_auc, auc_dict


def collect_seed_outputs(
    model_kind: str,
    ckpt_paths: Iterable[str | os.PathLike],
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_leads: int,
    n_classes: int,
    batch_size: int,
    device: torch.device,
) -> dict:
    """
    Load each seed checkpoint, run inference, and aggregate probabilities.

    Returns both:
    - mean/std macro-AUC across individual seed models; and
    - macro-AUC computed from the mean predicted probabilities.
    """
    per_seed_probs = []
    per_seed_macro_auc = []
    targets_ref = np.asarray(y_test).astype(np.int64)

    for ckpt_path in ckpt_paths:
        ckpt_path = resolve_project_path(ckpt_path)

        if not ckpt_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

        if model_kind == "teacher":
            model, _ = load_teacher_model(ckpt_path, n_leads, n_classes, device)
        elif model_kind == "student":
            model, _ = load_student_model(ckpt_path, n_leads, n_classes, device)
        else:
            raise ValueError(f"Unknown model_kind: {model_kind}")

        probs = infer_probs_numpy(model, X_test, batch_size=batch_size, device=device)
        macro_auc, _ = one_vs_rest_macro_auc(targets_ref, probs, n_classes)

        per_seed_probs.append(probs)
        per_seed_macro_auc.append(macro_auc)

        print(f"{ckpt_path.name} -> macro_auc={macro_auc:.4f}", flush=True)

        # Free GPU memory between checkpoints.
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_seed_probs = np.stack(per_seed_probs, axis=0)
    mean_probs = np.mean(per_seed_probs, axis=0)

    macro_auc_mean = float(np.mean(per_seed_macro_auc))
    macro_auc_std = (
        float(np.std(per_seed_macro_auc, ddof=1))
        if len(per_seed_macro_auc) > 1
        else 0.0
    )

    macro_auc_from_mean_probs, _ = one_vs_rest_macro_auc(
        targets_ref,
        mean_probs,
        n_classes,
    )

    return {
        "targets": targets_ref,
        "mean_probs": mean_probs,
        "macro_auc_mean": macro_auc_mean,
        "macro_auc_std": macro_auc_std,
        "macro_auc_from_mean_probs": macro_auc_from_mean_probs,
    }


def plot_single_aggregated_roc(
    targets: np.ndarray,
    probs: np.ndarray,
    class_names: List[str],
    title: str,
    save_path: str | os.PathLike,
    display_macro_auc: float | None = None,
) -> Tuple[float, Dict[str, float]]:
    """
    Plot class-wise ROC curves plus a macro-average ROC curve.

    Parameters
    ----------
    display_macro_auc:
        If provided, this value is displayed in the legend. This is useful when
        the reported result should be the mean across seed-specific AUCs rather
        than the AUC computed from averaged probabilities.
    """
    save_path = resolve_project_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(class_names)

    plt.figure(figsize=(8, 6))

    per_class_auc = {}
    for i, class_name in enumerate(class_names):
        y_true_binary = (targets == i).astype(np.int64)
        y_score = probs[:, i]

        fpr, tpr = binary_roc_curve(y_true_binary, y_score)
        class_auc = auc_trapezoid(fpr, tpr)
        per_class_auc[class_name] = class_auc

        plt.plot(
            fpr,
            tpr,
            linewidth=1.2,
            alpha=0.35,
            label=f"{class_name} (AUC={class_auc:.3f})",
        )

    macro_fpr, macro_tpr, macro_roc_auc, _ = compute_macro_roc_from_probs(
        targets=targets,
        probs=probs,
        n_classes=n_classes,
    )

    shown_auc = macro_roc_auc if display_macro_auc is None else display_macro_auc

    plt.plot(
        macro_fpr,
        macro_tpr,
        linewidth=3.0,
        label=f"Macro-average ROC (AUC={shown_auc:.3f})",
    )

    # Random-classifier reference line.
    plt.plot([0, 1], [0, 1], linestyle="--", linewidth=1.0)

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    return macro_roc_auc, per_class_auc


def plot_macro_auc_bar(summary: Dict[str, dict], save_path: str | os.PathLike) -> None:
    """
    Plot a bar chart comparing macro-AUC across model groups.
    """
    save_path = resolve_project_path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    model_names = list(summary.keys())
    means = [summary[k]["macro_auc_mean"] for k in model_names]
    stds = [summary[k]["macro_auc_std"] for k in model_names]

    plt.figure(figsize=(7, 5))
    plt.bar(model_names, means, yerr=stds, capsize=5)
    plt.ylabel("Macro-AUC")
    plt.title("Macro-AUC on PTB-XL test set (mean ± std across seeds)")
    plt.ylim(0.80, max(0.93, max(means) + 0.02))
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Plot aggregated ROC curves for the teacher, baseline student, "
            "and KD student."
        )
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size used during inference.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/roc_curves",
        help=(
            "Directory for ROC plots. Relative paths are resolved from the "
            "project root."
        ),
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default=DATA_PATH,
        help=(
            "Path to the processed PTB-XL npz file. Relative paths are "
            "resolved from the project root."
        ),
    )

    parser.add_argument(
        "--teacher_ckpts",
        nargs="+",
        default=DEFAULT_TEACHER_CKPTS,
        help="Teacher checkpoint paths.",
    )

    parser.add_argument(
        "--baseline_ckpts",
        nargs="+",
        default=DEFAULT_BASELINE_CKPTS,
        help="Baseline student checkpoint paths.",
    )

    parser.add_argument(
        "--kd_ckpts",
        nargs="+",
        default=DEFAULT_KD_CKPTS,
        help="KD student checkpoint paths.",
    )

    return parser.parse_args()


def main() -> None:
    """
    Entry point for command-line execution.
    """
    args = parse_args()

    # Keep CPU usage predictable on shared/HPC environments.
    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            # PyTorch raises if interop threads have already been initialised.
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    output_dir = resolve_project_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _, _, _, _, X_test, y_test, classes = load_splits(args.data_path)
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).astype(np.int64)

    n_leads = X_test.shape[1]
    n_classes = len(classes)
    class_names = [str(c) for c in classes]

    summary = {}

    model_groups = {
        "Teacher": (
            "teacher",
            args.teacher_ckpts,
            "teacher_aggregated_roc.png",
        ),
        "Baseline Student": (
            "student",
            args.baseline_ckpts,
            "baseline_student_aggregated_roc.png",
        ),
        "KD Student": (
            "student",
            args.kd_ckpts,
            "kd_student_aggregated_roc.png",
        ),
    }

    for display_name, (model_kind, ckpt_paths, filename) in model_groups.items():
        print(f"\n=== {display_name} ===", flush=True)

        result = collect_seed_outputs(
            model_kind=model_kind,
            ckpt_paths=ckpt_paths,
            X_test=X_test,
            y_test=y_test,
            n_leads=n_leads,
            n_classes=n_classes,
            batch_size=args.batch_size,
            device=device,
        )

        summary[display_name] = result
        save_path = output_dir / filename

        _, per_class_auc = plot_single_aggregated_roc(
            targets=result["targets"],
            probs=result["mean_probs"],
            class_names=class_names,
            title=f"{display_name}: Aggregated ROC Curves",
            save_path=save_path,
            display_macro_auc=result["macro_auc_mean"],
        )

        print(f"Saved figure: {save_path}", flush=True)
        print(
            f"{display_name} macro-AUC (mean ± std across seeds): "
            f"{result['macro_auc_mean']:.4f} ± {result['macro_auc_std']:.4f}",
            flush=True,
        )
        print(
            f"{display_name} macro-AUC from mean probabilities: "
            f"{result['macro_auc_from_mean_probs']:.4f}",
            flush=True,
        )

        print("Per-class AUCs from aggregated probabilities:", flush=True)
        for class_name, class_auc in per_class_auc.items():
            print(f"  {class_name}: {class_auc:.4f}", flush=True)

    bar_path = output_dir / "macro_auc_comparison_bar.png"
    plot_macro_auc_bar(summary, bar_path)
    print(f"\nSaved figure: {bar_path}", flush=True)

    print("\n=== Final summary ===", flush=True)
    for model_name, result in summary.items():
        print(
            f"{model_name}: "
            f"{result['macro_auc_mean']:.4f} ± {result['macro_auc_std']:.4f} "
            f"(mean across seeds), "
            f"aggregated-prob ROC AUC={result['macro_auc_from_mean_probs']:.4f}",
            flush=True,
        )


if __name__ == "__main__":
    main()