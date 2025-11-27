#student_cnn.py
import torch
import torch.nn as nn


class StudentCNN(nn.Module):
    """
    Compact Student CNN for knowledge distillation.

    Input
    -----
    x : (batch_size, n_leads, seq_len)

    Output
    ------
    logits : (batch_size, n_classes)
    """

    def __init__(self, n_leads: int = 12, n_classes: int = 5):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 96, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(96),
            nn.ReLU(inplace=True),

            nn.AdaptiveAvgPool1d(output_size=1),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=0.3),
            nn.Linear(96, 48),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.3),
            nn.Linear(48, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x
