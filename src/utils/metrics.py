from typing import Dict, Any

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    f1_score,
    confusion_matrix,
    classification_report,
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
        - acc: overall accuracy
        - macro_f1: macro-averaged F1-score
        - weighted_f1: weighted-averaged F1-score
        - confusion_matrix: sklearn confusion matrix
        - report: sklearn text classification report
    """
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)

            logits = model(X)

            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    if len(all_logits) == 0 or len(all_targets) == 0:
        raise ValueError("Evaluation loader produced no batches.")

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)
    preds = logits.argmax(dim=1)

    targets_np = targets.numpy()
    preds_np = preds.numpy()

    acc = (preds == targets).float().mean().item()
    macro_f1 = f1_score(targets_np, preds_np, average="macro", zero_division=0)
    weighted_f1 = f1_score(targets_np, preds_np, average="weighted", zero_division=0)

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
        "confusion_matrix": cm,
        "report": report,
    }