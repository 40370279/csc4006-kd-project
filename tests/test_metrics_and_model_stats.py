import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.student_cnn import StudentCNN
from src.utils.metrics import evaluate_classification
from src.utils.model_stats import (
    checkpoint_size_mb,
    count_all_params,
    count_trainable_params,
    model_size_mb,
)


class FixedLogitModel(torch.nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed_logits", torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, x):
        return self.fixed_logits[: x.shape[0]]


def test_evaluate_classification_returns_expected_keys_and_shapes():
    X = torch.zeros(5, 12, 20)
    y = torch.arange(5)
    logits = torch.eye(5) * 5.0

    loader = DataLoader(TensorDataset(X, y), batch_size=5)
    metrics = evaluate_classification(
        model=FixedLogitModel(logits),
        loader=loader,
        device=torch.device("cpu"),
    )

    assert metrics["acc"] == pytest.approx(1.0)
    assert metrics["macro_f1"] == pytest.approx(1.0)
    assert metrics["weighted_f1"] == pytest.approx(1.0)
    assert metrics["macro_auc"] == pytest.approx(1.0)
    assert metrics["confusion_matrix"].shape == (5, 5)
    assert metrics["logits"].shape == (5, 5)
    assert metrics["probs"].shape == (5, 5)
    assert metrics["targets"].tolist() == [0, 1, 2, 3, 4]
    assert metrics["preds"].tolist() == [0, 1, 2, 3, 4]


def test_evaluate_classification_rejects_empty_loader():
    loader = DataLoader(TensorDataset(torch.empty(0, 12, 20), torch.empty(0, dtype=torch.long)))
    model = FixedLogitModel(torch.empty(0, 5))

    with pytest.raises(ValueError):
        evaluate_classification(model, loader, torch.device("cpu"))


def test_model_stat_utilities_return_positive_values(tmp_path):
    model = StudentCNN(n_leads=12, n_classes=5, size="small")

    trainable = count_trainable_params(model)
    total = count_all_params(model)
    size = model_size_mb(model)

    assert trainable > 0
    assert total >= trainable
    assert size > 0

    checkpoint = tmp_path / "student.pt"
    torch.save({"model_state_dict": model.state_dict(), "metadata": {"seed": 42}}, checkpoint)
    assert checkpoint_size_mb(str(checkpoint)) > 0


def test_checkpoint_size_raises_for_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        checkpoint_size_mb(str(tmp_path / "missing.pt"))
