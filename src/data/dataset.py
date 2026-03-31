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
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]
        y = self.y[idx]

        # Ensure numpy array with float32 dtype
        x = np.asarray(x, dtype=np.float32)
        y = int(y)

        # Apply augmentation / transform if provided
        if self.transform is not None:
            x = self.transform(x)

        # Convert to tensors
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        return x, y    