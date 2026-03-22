import torch
import torch.nn as nn


class WeakStudentCNN(nn.Module):

    SIZE_CONFIGS = {
        "small": {"channels": [12, 24, 32], "hidden_dim": 24},
        "medium": {"channels": [16, 32, 48], "hidden_dim": 32},
        "large": {"channels": [24, 48, 64], "hidden_dim": 48},
    }

    def __init__(self, n_leads=12, n_classes=5, size="medium"):
        super().__init__()

        cfg = self.SIZE_CONFIGS[size]
        c1, c2, c3 = cfg["channels"]
        hidden = cfg["hidden_dim"]

        self.features = nn.Sequential(
            nn.Conv1d(n_leads, c1, 7, padding=3, bias=False),
            nn.BatchNorm1d(c1),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(c1, c2, 5, padding=2, bias=False),
            nn.BatchNorm1d(c2),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(c2, c3, 5, padding=2, bias=False),
            nn.BatchNorm1d(c3),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.1),
            nn.Linear(c3, hidden),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x, return_features=False):

        x = self.features[:-1](x)

        features = x

        x = self.features[-1](x)

        logits = self.classifier(x)

        if return_features:
            return logits, features

        return logits