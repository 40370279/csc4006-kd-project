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

    # Set model to evaluation mode (disables dropout, uses running BN stats)
    model.eval()

    # Lists to store outputs across all batches
    all_logits = []
    all_targets = []

    # Disable gradient computation for faster inference and lower memory usage
    with torch.no_grad():
        for X, y in loader:
            # Move input data and labels to the specified device (CPU/GPU)
            # non_blocking=True can improve performance with pinned memory
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Forward pass to compute logits (raw class scores)
            logits = model(X)

            # Store logits and targets on CPU for later aggregation
            all_logits.append(logits.detach().cpu())
            all_targets.append(y.detach().cpu())

    # Ensure that the dataloader actually produced data
    if len(all_logits) == 0 or len(all_targets) == 0:
        raise ValueError("Evaluation loader produced no batches.")

    # Concatenate all batch outputs into single tensors
    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    # Convert logits to probabilities using softmax
    probs = torch.softmax(logits, dim=1)

    # Predicted class is the index of the highest logit
    preds = logits.argmax(dim=1)

    # Convert tensors to NumPy arrays for sklearn metric functions
    targets_np = targets.numpy()
    preds_np = preds.numpy()
    probs_np = probs.numpy()

    # Compute accuracy (fraction of correct predictions)
    acc = (preds == targets).float().mean().item()

    # Macro-F1: treats all classes equally (important for imbalance)
    macro_f1 = f1_score(targets_np, preds_np, average="macro", zero_division=0)

    # Weighted-F1: accounts for class frequency
    weighted_f1 = f1_score(targets_np, preds_np, average="weighted", zero_division=0)

    # Compute macro AUC using one-vs-rest strategy for multiclass classification
    # If computation fails (e.g., missing class), return NaN
    try:
        macro_auc = roc_auc_score(
            targets_np,
            probs_np,
            multi_class="ovr",
            average="macro",
        )
    except ValueError:
        macro_auc = float("nan")

    # Confusion matrix (true vs predicted labels)
    cm = confusion_matrix(targets_np, preds_np)

    # Detailed classification report (precision, recall, F1 per class)
    report = classification_report(
        targets_np,
        preds_np,
        output_dict=False,
        zero_division=0,
    )

    # Return all computed metrics and raw outputs for further analysis
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