import numpy as np
import torch

from src.data.augmentations import ECGAugment
from src.data.dataset import ECGDataset


def test_ecg_dataset_returns_expected_tensor_shapes_and_dtypes():
    X = np.random.randn(3, 12, 5000).astype(np.float64)
    y = np.array([0, 1, 2])

    dataset = ECGDataset(X, y)
    x_item, y_item = dataset[1]

    assert len(dataset) == 3
    assert isinstance(x_item, torch.Tensor)
    assert x_item.shape == (12, 5000)
    assert x_item.dtype == torch.float32
    assert y_item.dtype == torch.long
    assert int(y_item) == 1


def test_ecg_dataset_applies_transform_before_tensor_conversion():
    X = np.zeros((1, 12, 20), dtype=np.float32)
    y = np.array([2])

    dataset = ECGDataset(X, y, transform=lambda sample: sample + 1.0)
    x_item, _ = dataset[0]

    assert torch.allclose(x_item, torch.ones_like(x_item))


def test_ecg_augment_preserves_shape_dtype_and_finiteness():
    np.random.seed(42)
    sample = np.random.randn(12, 500).astype(np.float32)

    augment = ECGAugment(
        noise_std=0.01,
        scale_range=(0.98, 1.02),
        max_shift=5,
        lead_drop_prob=0.10,
        baseline_wander_std=0.01,
        max_mask_width=50,
        stretch_range=(0.99, 1.01),
        p_scale=1.0,
        p_noise=1.0,
        p_shift=1.0,
        p_lead_drop=1.0,
        p_baseline=1.0,
        p_mask=1.0,
        p_stretch=1.0,
    )

    augmented = augment(sample)

    assert augmented.shape == sample.shape
    assert augmented.dtype == np.float32
    assert np.isfinite(augmented).all()


def test_ecg_augment_with_disabled_transforms_preserves_values():
    sample = np.random.randn(12, 100).astype(np.float32)
    augment = ECGAugment(
        noise_std=0.0,
        max_shift=0,
        lead_drop_prob=0.0,
        baseline_wander_std=0.0,
        max_mask_width=0,
        p_scale=0.0,
        p_noise=0.0,
        p_shift=0.0,
        p_lead_drop=0.0,
        p_baseline=0.0,
        p_mask=0.0,
        p_stretch=0.0,
    )

    augmented = augment(sample)

    assert np.allclose(augmented, sample)
