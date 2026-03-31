import numpy as np


class ECGAugment:
    """
    ECG augmentation pipeline for samples shaped (C, T),
    where C = number of leads and T = signal length.

    Designed to remain safe/conservative for ECG classification while
    giving more useful variability than simple noise + scaling alone.

    Included augmentations:
    - amplitude scaling
    - additive Gaussian noise
    - temporal shift
    - random lead dropout
    - baseline wander
    - random temporal masking
    - mild time stretch

    Notes:
    - all transforms are applied on a copy
    - output shape is always preserved as (C, T)
    """

    def __init__(
        self,
        noise_std: float = 0.005,
        scale_range=(0.95, 1.05),
        max_shift: int = 40,
        lead_drop_prob: float = 0.02,
        baseline_wander_std: float = 0.03,
        baseline_freq_range=(0.05, 0.5),
        max_mask_width: int = 200,
        stretch_range=(0.98, 1.02),
        p_scale: float = 0.8,
        p_noise: float = 0.8,
        p_shift: float = 0.5,
        p_lead_drop: float = 0.3,
        p_baseline: float = 0.4,
        p_mask: float = 0.3,
        p_stretch: float = 0.2,
    ):
        self.noise_std = float(noise_std)
        self.scale_range = tuple(scale_range)
        self.max_shift = int(max_shift)
        self.lead_drop_prob = float(lead_drop_prob)

        self.baseline_wander_std = float(baseline_wander_std)
        self.baseline_freq_range = tuple(baseline_freq_range)

        self.max_mask_width = int(max_mask_width)
        self.stretch_range = tuple(stretch_range)

        self.p_scale = float(p_scale)
        self.p_noise = float(p_noise)
        self.p_shift = float(p_shift)
        self.p_lead_drop = float(p_lead_drop)
        self.p_baseline = float(p_baseline)
        self.p_mask = float(p_mask)
        self.p_stretch = float(p_stretch)

    def _time_stretch(self, x: np.ndarray, factor: float) -> np.ndarray:
        """
        Mild temporal resampling while preserving output length.
        Factor > 1.0 stretches, factor < 1.0 compresses.
        """
        c, t = x.shape
        new_t = max(2, int(round(t * factor)))

        stretched = np.empty((c, new_t), dtype=np.float32)
        orig_idx = np.arange(t, dtype=np.float32)
        new_idx = np.linspace(0, t - 1, new_t, dtype=np.float32)

        for i in range(c):
            stretched[i] = np.interp(new_idx, orig_idx, x[i]).astype(np.float32)

        if new_t > t:
            start = (new_t - t) // 2
            stretched = stretched[:, start:start + t]
        elif new_t < t:
            pad_left = (t - new_t) // 2
            pad_right = t - new_t - pad_left
            stretched = np.pad(
                stretched,
                ((0, 0), (pad_left, pad_right)),
                mode="edge",
            )

        return stretched.astype(np.float32)

    def _baseline_wander(self, x: np.ndarray) -> np.ndarray:
        """
        Add a smooth low-frequency drift to simulate baseline wander.
        """
        c, t = x.shape
        time = np.linspace(0.0, 1.0, t, dtype=np.float32)

        freq = np.random.uniform(
            self.baseline_freq_range[0],
            self.baseline_freq_range[1]
        )
        phase = np.random.uniform(0.0, 2.0 * np.pi)
        amp = np.random.uniform(0.0, self.baseline_wander_std)

        drift = amp * np.sin(2.0 * np.pi * freq * time + phase)
        drift = drift.astype(np.float32)[None, :]  # shape (1, T)
        return x + drift

    def _random_mask(self, x: np.ndarray) -> np.ndarray:
        """
        Zero out a short temporal window across all leads.
        """
        _, t = x.shape
        if self.max_mask_width <= 0 or self.max_mask_width >= t:
            return x

        width = np.random.randint(20, self.max_mask_width + 1)
        start = np.random.randint(0, t - width + 1)
        x[:, start:start + width] = 0.0
        return x

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32).copy()

        # Amplitude scaling
        if self.p_scale > 0.0 and np.random.rand() < self.p_scale:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            x *= np.float32(scale)

        # Add Gaussian noise
        if self.noise_std > 0.0 and self.p_noise > 0.0 and np.random.rand() < self.p_noise:
            noise = np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
            x += noise

        # Baseline wander
        if self.baseline_wander_std > 0.0 and self.p_baseline > 0.0 and np.random.rand() < self.p_baseline:
            x = self._baseline_wander(x)

        # Mild temporal stretch
        if self.p_stretch > 0.0 and np.random.rand() < self.p_stretch:
            factor = np.random.uniform(self.stretch_range[0], self.stretch_range[1])
            if abs(factor - 1.0) > 1e-3:
                x = self._time_stretch(x, factor)

        # Temporal shift
        if self.max_shift > 0 and self.p_shift > 0.0 and np.random.rand() < self.p_shift:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            if shift != 0:
                x = np.roll(x, shift, axis=1)

        # Random temporal masking
        if self.max_mask_width > 0 and self.p_mask > 0.0 and np.random.rand() < self.p_mask:
            x = self._random_mask(x)

        # Random lead dropout
        if self.lead_drop_prob > 0.0 and self.p_lead_drop > 0.0 and np.random.rand() < self.p_lead_drop:
            drop_mask = np.random.rand(x.shape[0]) < self.lead_drop_prob
            if np.any(drop_mask):
                x[drop_mask, :] = 0.0

        return x.astype(np.float32)