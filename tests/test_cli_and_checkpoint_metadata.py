import sys
from types import SimpleNamespace

import torch

from scripts import train_student_baseline, train_student_kd, train_teacher
from src.models.student_cnn import StudentCNN
from src.models.teacher_cnn import TeacherCNN


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


def test_teacher_cli_parses_capacity_and_reproducibility_flags(monkeypatch, tmp_path):
    ckpt = tmp_path / "teacher.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_teacher.py",
            "--teacher_size",
            "medium",
            "--seed",
            "123",
            "--teacher_ckpt",
            str(ckpt),
            "--disable_augmentation",
        ],
    )

    args = train_teacher.parse_args()

    assert args.teacher_size == "medium"
    assert args.seed == 123
    assert args.teacher_ckpt == str(ckpt)
    assert args.disable_augmentation is True


def test_student_cli_parses_capacity_and_checkpoint_path(monkeypatch, tmp_path):
    ckpt = tmp_path / "student.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_student_baseline.py",
            "--student_size",
            "large",
            "--student_ckpt",
            str(ckpt),
            "--use_weighted_sampler",
        ],
    )

    args = train_student_baseline.parse_args()

    assert args.student_size == "large"
    assert args.student_ckpt == str(ckpt)
    assert args.use_weighted_sampler is True


def test_kd_cli_parses_distillation_parameters(monkeypatch, tmp_path):
    teacher_ckpt = tmp_path / "teacher.pt"
    student_ckpt = tmp_path / "student_kd.pt"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_student_kd.py",
            "--student_size",
            "small",
            "--temperature",
            "8",
            "--alpha_start",
            "0.1",
            "--alpha_end",
            "0.3",
            "--teacher_ckpt",
            str(teacher_ckpt),
            "--student_ckpt",
            str(student_ckpt),
        ],
    )

    args = train_student_kd.parse_args()

    assert args.student_size == "small"
    assert args.temperature == 8.0
    assert args.alpha_start == 0.1
    assert args.alpha_end == 0.3
    assert args.teacher_ckpt == str(teacher_ckpt)
    assert args.student_ckpt == str(student_ckpt)


def test_teacher_checkpoint_contains_reproducibility_metadata(monkeypatch, tmp_path):
    _add_tiny_teacher_config(monkeypatch)
    model = TeacherCNN(n_leads=12, n_classes=5, size="tiny")
    args = SimpleNamespace(teacher_size="tiny", seed=42)
    checkpoint = tmp_path / "teacher.pt"

    train_teacher.save_checkpoint(
        path=str(checkpoint),
        model=model,
        classes=["CD", "HYP", "MI", "NORM", "STTC"],
        n_leads=12,
        best_val_macro_auc=0.91,
        epoch=5,
        args=args,
    )

    payload = torch.load(checkpoint, map_location="cpu")

    assert payload["teacher_arch"] == "TeacherCNN"
    assert payload["teacher_size"] == "tiny"
    assert payload["seed"] == 42
    assert payload["epoch"] == 5
    assert payload["classes"] == ["CD", "HYP", "MI", "NORM", "STTC"]
    assert "model_state_dict" in payload
    assert "train_args" in payload


def test_student_checkpoint_can_store_expected_metadata(tmp_path):
    model = StudentCNN(n_leads=12, n_classes=5, size="small")
    checkpoint = tmp_path / "student.pt"

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "classes": ["CD", "HYP", "MI", "NORM", "STTC"],
            "student_arch": "StudentCNN",
            "student_size": "small",
            "seed": 42,
            "best_val_macro_auc": 0.89,
            "train_args": {"student_size": "small"},
        },
        checkpoint,
    )

    payload = torch.load(checkpoint, map_location="cpu")

    assert payload["student_arch"] == "StudentCNN"
    assert payload["student_size"] == "small"
    assert payload["best_val_macro_auc"] == 0.89
    assert "model_state_dict" in payload
