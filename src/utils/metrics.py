#metrics.py
from typing import Dict

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, confusion_matrix, classification_report


def evaluate_classification(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Dict[str, object]:
    model.eval()
    all_logits = []
    all_targets = []

    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            y = y.to(device)
            logits = model(X)
            all_logits.append(logits.cpu())
            all_targets.append(y.cpu())

    logits = torch.cat(all_logits, dim=0)
    targets = torch.cat(all_targets, dim=0)

    preds = logits.argmax(dim=1)
    acc = (preds == targets).float().mean().item()
    macro_f1 = f1_score(targets.numpy(), preds.numpy(), average="macro")

    cm = confusion_matrix(targets.numpy(), preds.numpy())
    report = classification_report(targets.numpy(), preds.numpy(), output_dict=False)

    return {
        "acc": acc,
        "macro_f1": macro_f1,
        "confusion_matrix": cm,
        "report": report,
    }
