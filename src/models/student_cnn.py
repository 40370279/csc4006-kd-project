import torch
import torch.nn as nn


class StudentCNN(nn.Module):
    SIZE_CONFIGS = {
        "small": {"channels": [48, 96, 128, 128], "hidden_dim": 64},
        "medium": {"channels": [64, 128, 192, 192], "hidden_dim": 96},
        "large": {"channels": [96, 192, 256, 256], "hidden_dim": 128},
    }

    def __init__(self, n_leads: int = 12, n_classes: int = 5, size: str = "small"):
        super().__init__()

        if size not in self.SIZE_CONFIGS:
            raise ValueError(
                f"Invalid student size '{size}'. Must be one of {list(self.SIZE_CONFIGS.keys())}"
            )

        cfg = self.SIZE_CONFIGS[size]
        c1, c2, c3, c4 = cfg["channels"]
        hidden_dim = cfg["hidden_dim"]

        self.feature_extractor = nn.Sequential(
            nn.Conv1d(n_leads, c1, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(c1, c2, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),

            nn.Conv1d(c2, c3, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),

            nn.Conv1d(c3, c4, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(c4),
            nn.ReLU(inplace=True),
        )

        self.global_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.25),
            nn.Linear(c4, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.25),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        features = self.feature_extractor(x)
        pooled = self.global_pool(features)
        logits = self.classifier(pooled)

        if return_features:
            return logits, features

        return logits