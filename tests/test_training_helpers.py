import numpy as np
import pytest

from scripts.train_student_baseline import compute_class_weights
from scripts.train_student_kd import get_alpha
from scripts.train_teacher import load_splits


def test_compute_class_weights_mean_is_one():
    labels = np.array([0, 0, 0, 1, 1, 2])
    weights = compute_class_weights(labels, gamma=0.20)
    assert pytest.approx(float(weights.mean()), rel=1e-6) == 1.0


def test_compute_class_weights_rejects_non_contiguous_labels():
    labels = np.array([0, 2, 2])
    with pytest.raises(ValueError):
        compute_class_weights(labels)


def test_get_alpha_hits_start_and_end_values():
    assert get_alpha(epoch=1, total_epochs=10, alpha_start=0.2, alpha_end=0.5) == pytest.approx(0.2)
    assert get_alpha(epoch=10, total_epochs=10, alpha_start=0.2, alpha_end=0.5) == pytest.approx(0.5)


def test_load_splits_raises_if_processed_file_missing(tmp_path):
    missing = tmp_path / 'does_not_exist.npz'
    with pytest.raises(FileNotFoundError):
        load_splits(str(missing))
