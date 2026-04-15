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
        noise_std: float = 0.005,             # Standard deviation of Gaussian noise
        scale_range=(0.95, 1.05),             # Range for amplitude scaling
        max_shift: int = 40,                  # Maximum temporal shift (in samples)
        lead_drop_prob: float = 0.02,         # Probability of dropping each lead
        baseline_wander_std: float = 0.03,    # Max amplitude of baseline drift
        baseline_freq_range=(0.05, 0.5),      # Frequency range for baseline drift (Hz-like)
        max_mask_width: int = 200,            # Max width of temporal masking window
        stretch_range=(0.98, 1.02),           # Range for time stretching factor
        p_scale: float = 0.8,                 # Probability of applying scaling
        p_noise: float = 0.8,                 # Probability of adding noise
        p_shift: float = 0.5,                 # Probability of temporal shift
        p_lead_drop: float = 0.3,             # Probability of lead dropout
        p_baseline: float = 0.4,              # Probability of baseline wander
        p_mask: float = 0.3,                  # Probability of temporal masking
        p_stretch: float = 0.2,               # Probability of time stretching
    ):
        # Store all parameters as class attributes
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
        # Get shape: channels (leads), time
        c, t = x.shape

        # Compute new length after stretching/compressing
        new_t = max(2, int(round(t * factor)))

        # Allocate output array
        stretched = np.empty((c, new_t), dtype=np.float32)

        # Original and new time indices for interpolation
        orig_idx = np.arange(t, dtype=np.float32)
        new_idx = np.linspace(0, t - 1, new_t, dtype=np.float32)

        # Interpolate each channel independently
        for i in range(c):
            stretched[i] = np.interp(new_idx, orig_idx, x[i]).astype(np.float32)

        # Adjust back to original length (crop or pad)
        if new_t > t:
            # Crop center if too long
            start = (new_t - t) // 2
            stretched = stretched[:, start:start + t]
        elif new_t < t:
            # Pad edges if too short
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

        # Create normalized time axis (0 → 1)
        time = np.linspace(0.0, 1.0, t, dtype=np.float32)

        # Random sinusoid parameters
        freq = np.random.uniform(
            self.baseline_freq_range[0],
            self.baseline_freq_range[1]
        )
        phase = np.random.uniform(0.0, 2.0 * np.pi)
        amp = np.random.uniform(0.0, self.baseline_wander_std)

        # Generate sinusoidal drift
        drift = amp * np.sin(2.0 * np.pi * freq * time + phase)

        # Expand to match channel dimension
        drift = drift.astype(np.float32)[None, :]  # shape (1, T)

        # Add drift to all channels
        return x + drift

    def _random_mask(self, x: np.ndarray) -> np.ndarray:
        """
        Zero out a short temporal window across all leads.
        """
        _, t = x.shape

        # If masking is invalid (too large/small), skip
        if self.max_mask_width <= 0 or self.max_mask_width >= t:
            return x

        # Randomly choose mask width and start position
        width = np.random.randint(20, self.max_mask_width + 1)
        start = np.random.randint(0, t - width + 1)

        # Zero out selected region
        x[:, start:start + width] = 0.0
        return x

    def __call__(self, x):
        # Ensure numpy float32 array and copy to avoid modifying original
        x = np.asarray(x, dtype=np.float32).copy()

        # Amplitude scaling
        if self.p_scale > 0.0 and np.random.rand() < self.p_scale:
            scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            x *= np.float32(scale)

        # Add Gaussian noise
        if self.noise_std > 0.0 and self.p_noise > 0.0 and np.random.rand() < self.p_noise:
            noise = np.random.normal(0.0, self.noise_std, size=x.shape).astype(np.float32)
            x += noise

        # Baseline wander (low-frequency drift)
        if self.baseline_wander_std > 0.0 and self.p_baseline > 0.0 and np.random.rand() < self.p_baseline:
            x = self._baseline_wander(x)

        # Mild temporal stretch/compression
        if self.p_stretch > 0.0 and np.random.rand() < self.p_stretch:
            factor = np.random.uniform(self.stretch_range[0], self.stretch_range[1])
            if abs(factor - 1.0) > 1e-3:  # avoid unnecessary computation
                x = self._time_stretch(x, factor)

        # Temporal shift (circular shift)
        if self.max_shift > 0 and self.p_shift > 0.0 and np.random.rand() < self.p_shift:
            shift = np.random.randint(-self.max_shift, self.max_shift + 1)
            if shift != 0:
                x = np.roll(x, shift, axis=1)

        # Random temporal masking
        if self.max_mask_width > 0 and self.p_mask > 0.0 and np.random.rand() < self.p_mask:
            x = self._random_mask(x)

        # Random lead dropout (simulate missing leads)
        if self.lead_drop_prob > 0.0 and self.p_lead_drop > 0.0 and np.random.rand() < self.p_lead_drop:
            drop_mask = np.random.rand(x.shape[0]) < self.lead_drop_prob
            if np.any(drop_mask):
                x[drop_mask, :] = 0.0

        # Return final augmented sample
        return x.astype(np.float32)