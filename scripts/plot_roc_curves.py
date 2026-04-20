import os
import sys
import argparse
import numpy as np

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy.core
sys.modules["numpy._core"] = numpy.core

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

from src.models.teacher_cnn import TeacherCNN
from src.models.student_cnn import StudentCNN


DATA_PATH = os.path.join("processed", "ptbxl_500hz_10s.npz")

# Do NOT read classes from the NPZ on Kelvin2; that field crashes there.
FIXED_CLASSES = np.array(["CD", "HYP", "MI", "NORM", "STTC"], dtype=object)

DEFAULT_TEACHER_CKPTS = [
    os.path.join("checkpoints", "teacher", "teacher_cnn_seed42.pt"),
    os.path.join("checkpoints", "teacher", "teacher_cnn_seed123.pt"),
    os.path.join("checkpoints", "teacher", "teacher_cnn_seed999.pt"),
]

DEFAULT_BASELINE_CKPTS = [
    os.path.join("checkpoints", "student_baseline", "student_baseline_seed42.pt"),
    os.path.join("checkpoints", "student_baseline", "student_baseline_seed123.pt"),
    os.path.join("checkpoints", "student_baseline", "student_baseline_seed999.pt"),
]

DEFAULT_KD_CKPTS = [
    os.path.join("checkpoints", "student_kd", "student_kd_seed42.pt"),
    os.path.join("checkpoints", "student_kd", "student_kd_seed123.pt"),
    os.path.join("checkpoints", "student_kd", "student_kd_seed999.pt"),
]


def load_splits(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. "
            f"Run scripts/preprocess_ptbxl.py first."
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


def safe_torch_load(path, device):
    """
    Load checkpoints created with older/full pickle metadata.
    PyTorch 2.6 defaults to weights_only=True, which breaks these files.
    """
    return torch.load(path, map_location=device, weights_only=False)


def load_teacher_model(ckpt_path, n_leads, n_classes, device):
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


def load_student_model(ckpt_path, n_leads, n_classes, device):
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

def softmax_np(logits):
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(logits)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def infer_probs_numpy(model, X, batch_size, device):
    probs_list = []

    with torch.no_grad():
        for start in range(0, len(X), batch_size):
            end = min(start + batch_size, len(X))
            xb = torch.from_numpy(np.asarray(X[start:end])).float().to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            probs_list.append(probs)

    return np.concatenate(probs_list, axis=0)


def binary_roc_curve(y_true_binary, y_score):
    """
    Pure NumPy ROC curve for binary labels.
    Returns fpr, tpr with endpoints included.
    """
    y_true_binary = np.asarray(y_true_binary).astype(np.int64)
    y_score = np.asarray(y_score).astype(np.float64)

    # Sort descending by score
    order = np.argsort(-y_score, kind="mergesort")
    y_true_sorted = y_true_binary[order]
    y_score_sorted = y_score[order]

    positives = np.sum(y_true_sorted == 1)
    negatives = np.sum(y_true_sorted == 0)

    if positives == 0 or negatives == 0:
        return np.array([0.0, 1.0]), np.array([0.0, 1.0])

    tps = np.cumsum(y_true_sorted == 1)
    fps = np.cumsum(y_true_sorted == 0)

    # Keep only score change points
    distinct = np.where(np.diff(y_score_sorted))[0]
    threshold_idxs = np.r_[distinct, len(y_score_sorted) - 1]

    tps = tps[threshold_idxs]
    fps = fps[threshold_idxs]

    tpr = tps / positives
    fpr = fps / negatives

    # Add origin
    tpr = np.r_[0.0, tpr]
    fpr = np.r_[0.0, fpr]

    # Add end if needed
    if fpr[-1] != 1.0 or tpr[-1] != 1.0:
        fpr = np.r_[fpr, 1.0]
        tpr = np.r_[tpr, 1.0]

    return fpr, tpr


def auc_trapezoid(x, y):
    return float(np.trapz(y, x))


def one_vs_rest_macro_auc(targets, probs, n_classes):
    per_class_auc = {}

    for i in range(n_classes):
        y_true_binary = (targets == i).astype(np.int64)
        y_score = probs[:, i]
        fpr, tpr = binary_roc_curve(y_true_binary, y_score)
        per_class_auc[i] = auc_trapezoid(fpr, tpr)

    macro_auc = float(np.mean(list(per_class_auc.values())))
    return macro_auc, per_class_auc


def compute_macro_roc_from_probs(targets, probs, n_classes):
    """
    Macro-average ROC via interpolation over one-vs-rest ROC curves.
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


def collect_seed_outputs(model_kind, ckpt_paths, X_test, y_test, n_leads, n_classes, batch_size, device):
    per_seed_probs = []
    per_seed_macro_auc = []
    targets_ref = np.asarray(y_test).astype(np.int64)

    for ckpt_path in ckpt_paths:
        if not os.path.exists(ckpt_path):
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

        print(f"{os.path.basename(ckpt_path)} -> macro_auc={macro_auc:.4f}", flush=True)

        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    per_seed_probs = np.stack(per_seed_probs, axis=0)
    mean_probs = np.mean(per_seed_probs, axis=0)

    macro_auc_mean = float(np.mean(per_seed_macro_auc))
    macro_auc_std = float(np.std(per_seed_macro_auc, ddof=1)) if len(per_seed_macro_auc) > 1 else 0.0

    macro_auc_from_mean_probs, _ = one_vs_rest_macro_auc(targets_ref, mean_probs, n_classes)

    return {
        "targets": targets_ref,
        "mean_probs": mean_probs,
        "macro_auc_mean": macro_auc_mean,
        "macro_auc_std": macro_auc_std,
        "macro_auc_from_mean_probs": macro_auc_from_mean_probs,
    }


def plot_single_aggregated_roc(
    targets,
    probs,
    class_names,
    title,
    save_path,
    display_macro_auc=None,
):
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


def plot_macro_auc_bar(summary, save_path):
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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot aggregated ROC curves for teacher, baseline student, and KD student."
    )
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--output_dir", type=str, default="results/roc_curves")

    parser.add_argument("--teacher_ckpts", nargs="+", default=DEFAULT_TEACHER_CKPTS)
    parser.add_argument("--baseline_ckpts", nargs="+", default=DEFAULT_BASELINE_CKPTS)
    parser.add_argument("--kd_ckpts", nargs="+", default=DEFAULT_KD_CKPTS)

    return parser.parse_args()


def main():
    args = parse_args()

    torch.set_num_threads(1)
    if hasattr(torch, "set_num_interop_threads"):
        try:
            torch.set_num_interop_threads(1)
        except Exception:
            pass

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device, flush=True)

    os.makedirs(args.output_dir, exist_ok=True)

    _, _, _, _, X_test, y_test, classes = load_splits()
    X_test = np.asarray(X_test)
    y_test = np.asarray(y_test).astype(np.int64)

    n_leads = X_test.shape[1]
    n_classes = len(classes)
    class_names = [str(c) for c in classes]

    summary = {}

    model_groups = {
        "Teacher": ("teacher", args.teacher_ckpts, "teacher_aggregated_roc.png"),
        "Baseline Student": ("student", args.baseline_ckpts, "baseline_student_aggregated_roc.png"),
        "KD Student": ("student", args.kd_ckpts, "kd_student_aggregated_roc.png"),
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
        save_path = os.path.join(args.output_dir, filename)

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

    bar_path = os.path.join(args.output_dir, "macro_auc_comparison_bar.png")
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