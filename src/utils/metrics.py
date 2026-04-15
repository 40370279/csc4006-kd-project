from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
)


def evaluate_classification(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Evaluate a classification model on a dataloader.

    Returns
    -------
    dict containing:
        - acc
        - macro_f1
        - weighted_f1
        - macro_auc
        - macro_auc_ovr
        - confusion_matrix
        - report
        - logits
        - probs
        - targets
        - preds
    """
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            logits = model(X)

            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    if len(all_logits) == 0 or len(all_targets) == 0:
        raise ValueError("Evaluation loader produced no batches.")

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    probs = torch.softmax(logits, dim=1)
    preds = logits.argmax(dim=1)

    targets_np = targets.numpy()
    preds_np = preds.numpy()
    probs_np = probs.numpy()

    acc = (preds == targets).float().mean().item()
    macro_f1 = f1_score(targets_np, preds_np, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets_np, preds_np, average="weighted", zero_division=0)

    try:
        macro_auc = roc_auc_score(
            targets_np,
            probs_np,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        macro_auc = float("nan")

    cm = confusion_matrix(targets_np, preds_np)
    report = classification_report(
        targets_np,
        preds_np,
        output_dict=False,
        zero_division=0,
    )

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "macro_auc": macro_auc,
        "macro_auc_ovr": macro_auc,  # backward-compatible alias
        "confusion_matrix": cm,
        "report": report,
        "logits": logits.numpy(),
        "probs": probs_np,
        "targets": targets_np,
        "preds": preds_np,
    }