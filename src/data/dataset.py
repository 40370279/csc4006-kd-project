import numpy as np
import torch
from torch.utils.data import Dataset


class ECGDataset(Dataset):
    """
    PyTorch Dataset for ECG classification.

    Expected input:
        X: numpy array of shape (N, C, T)
        y: numpy array of shape (N,)
        transform: optional callable applied to each ECG sample

    Returns:
        x: torch.FloatTensor of shape (C, T)
        y: torch.LongTensor scalar
    """

    def __init__(self, X, y, transform=None):
        # Store ECG signals (shape: N samples, C leads, T timesteps)
        self.X = X

        # Store corresponding labels (shape: N,)
        self.y = y

        # Optional transform (e.g., augmentation pipeline)
        self.transform = transform

    def __len__(self):
        # Return total number of samples in dataset
        return len(self.X)

    def __getitem__(self, idx):
        # Retrieve ECG sample and label at index
        x = self.X[idx]
        y = self.y[idx]

        # Ensure ECG sample is a NumPy array with float32 precision
        # (important for compatibility with PyTorch and GPU efficiency)
        x = np.asarray(x, dtype=np.float32)

        # Ensure label is a standard Python integer
        y = int(y)

        # Apply augmentation / preprocessing if a transform is provided
        # This is typically used for training-time data augmentation
        if self.transform is not None:
            x = self.transform(x)

        # Convert ECG signal to PyTorch tensor (float32)
        x = torch.tensor(x, dtype=torch.float32)

        # Convert label to PyTorch tensor (long for classification tasks)
        y = torch.tensor(y, dtype=torch.long)

        # Return (input, label) pair
        return x, y