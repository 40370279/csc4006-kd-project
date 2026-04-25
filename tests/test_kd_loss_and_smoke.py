import pytest
import torch
import torch.nn as nn
import torch.optim as optim

from scripts.train_student_kd import FeatureProjector, kd_loss
from src.models.student_cnn import StudentCNN


def test_kd_loss_is_finite_and_backpropagates():
    batch_size = 4
    n_classes = 5
    student_dim = 18
    teacher_dim = 36

    student_logits = torch.randn(batch_size, n_classes, requires_grad=True)
    teacher_logits = torch.randn(batch_size, n_classes)
    student_pooled = torch.randn(batch_size, student_dim, requires_grad=True)
    teacher_pooled = torch.randn(batch_size, teacher_dim)
    targets = torch.tensor([0, 1, 2, 3])

    projector = FeatureProjector(in_dim=teacher_dim, out_dim=student_dim)
    hard_criterion = nn.CrossEntropyLoss()

    total_loss, hard_loss, soft_loss, logit_mse, feature_mse = kd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_pooled=student_pooled,
        teacher_pooled=teacher_pooled,
        teacher_projector=projector,
        targets=targets,
        hard_criterion=hard_criterion,
        alpha=0.1,
        temperature=4.0,
        feature_mse_weight=0.03,
        logit_mse_weight=0.03,
    )

    assert total_loss.ndim == 0
    assert torch.isfinite(total_loss)
    assert torch.isfinite(hard_loss)
    assert torch.isfinite(soft_loss)
    assert torch.isfinite(logit_mse)
    assert torch.isfinite(feature_mse)

    total_loss.backward()
    assert student_logits.grad is not None
    assert student_pooled.grad is not None


def test_kd_loss_rejects_feature_shape_mismatch():
    student_logits = torch.randn(2, 5)
    teacher_logits = torch.randn(2, 5)
    student_pooled = torch.randn(2, 18)
    teacher_pooled = torch.randn(2, 36)
    targets = torch.tensor([0, 1])

    projector = FeatureProjector(in_dim=36, out_dim=17)

    with pytest.raises(ValueError):
        kd_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            student_pooled=student_pooled,
            teacher_pooled=teacher_pooled,
            teacher_projector=projector,
            targets=targets,
            hard_criterion=nn.CrossEntropyLoss(),
            alpha=0.1,
            temperature=4.0,
            feature_mse_weight=0.03,
            logit_mse_weight=0.03,
        )


def test_synthetic_kd_student_training_step_runs_without_ptbxl_or_gpu():
    torch.manual_seed(7)

    student = StudentCNN(n_leads=12, n_classes=5, size="small")
    student.train()

    x = torch.randn(2, 12, 128)
    y = torch.tensor([0, 1])

    student_logits, student_pooled = student(x, return_features=True)
    teacher_logits = torch.randn_like(student_logits)
    teacher_pooled = torch.randn(2, 36)
    projector = FeatureProjector(in_dim=36, out_dim=student_pooled.shape[1])

    optimizer = optim.AdamW(
        list(student.parameters()) + list(projector.parameters()),
        lr=1e-3,
    )

    loss, *_ = kd_loss(
        student_logits=student_logits,
        teacher_logits=teacher_logits,
        student_pooled=student_pooled,
        teacher_pooled=teacher_pooled,
        teacher_projector=projector,
        targets=y,
        hard_criterion=nn.CrossEntropyLoss(),
        alpha=0.1,
        temperature=2.0,
        feature_mse_weight=0.03,
        logit_mse_weight=0.03,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    assert torch.isfinite(loss)
