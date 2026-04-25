import pytest
import torch

from src.models.student_cnn import StudentCNN
from src.models.teacher_cnn import GlobalStatsPool1D, TeacherCNN


def _add_tiny_teacher_config(monkeypatch):
    monkeypatch.setitem(
        TeacherCNN.SIZE_CONFIGS,
        "tiny",
        {
            "stem": 16,
            "channels": [48],
            "strides": [1],
            "dropout": 0.0,
            "fc_dim": 16,
            "cls_drop1": 0.0,
            "cls_drop2": 0.0,
        },
    )


def test_student_forward_returns_logits_and_features():
    model = StudentCNN(n_leads=12, n_classes=5, size="small")
    model.eval()

    x = torch.randn(2, 12, 64)
    with torch.no_grad():
        logits, features = model(x, return_features=True)

    assert logits.shape == (2, 5)
    assert features.shape == (2, model.feature_dim)
    assert torch.isfinite(logits).all()
    assert torch.isfinite(features).all()


def test_teacher_forward_returns_logits_and_temporal_features(monkeypatch):
    _add_tiny_teacher_config(monkeypatch)
    model = TeacherCNN(n_leads=12, n_classes=5, size="tiny")
    model.eval()

    x = torch.randn(2, 12, 64)
    with torch.no_grad():
        logits, features = model(x, return_features=True)

    assert logits.shape == (2, 5)
    assert features.ndim == 3
    assert features.shape[0] == 2
    assert torch.isfinite(logits).all()
    assert torch.isfinite(features).all()


def test_global_stats_pool_doubles_channel_dimension():
    pool = GlobalStatsPool1D()
    x = torch.randn(4, 8, 32)

    pooled = pool(x)

    assert pooled.shape == (4, 16)
    assert torch.isfinite(pooled).all()


def test_invalid_model_size_is_rejected():
    with pytest.raises(ValueError):
        StudentCNN(size="does-not-exist")

    with pytest.raises(ValueError):
        TeacherCNN(size="does-not-exist")
