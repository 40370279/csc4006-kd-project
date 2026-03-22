from src.models.teacher_cnn import (
    SEBlock,
    ResidualSEBlock1D,
    DualPoolHead,
    TeacherCNN,
)

import torch
import torch.nn as nn


class WeakTeacherCNN(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 32, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(32),
            nn.GELU(),
        )

        self.block1 = ResidualSEBlock1D(32, 48, kernel_size=11, use_se=False)
        self.block2 = ResidualSEBlock1D(48, 64, kernel_size=9, use_se=False)
        self.block3 = ResidualSEBlock1D(64, 96, kernel_size=7, dilation=2, use_se=False)
        self.block4 = ResidualSEBlock1D(96, 128, kernel_size=5, dilation=2, use_se=False)
        self.block5 = ResidualSEBlock1D(128, 192, kernel_size=3, dilation=3, pool=False, use_se=False)

        self.pool_head = DualPoolHead()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.25),
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(p=0.20),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        features = x
        x = self.pool_head(x)
        logits = self.classifier(x)

        if return_features:
            return logits, features
        return logits


class StrongTeacherCNN(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 80, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(80),
            nn.GELU(),
        )

        self.block1 = ResidualSEBlock1D(80, 128, kernel_size=11)
        self.block2 = ResidualSEBlock1D(128, 160, kernel_size=9)
        self.block3 = ResidualSEBlock1D(160, 224, kernel_size=7, dilation=2)
        self.block4 = ResidualSEBlock1D(224, 320, kernel_size=7, dilation=2)
        self.block5 = ResidualSEBlock1D(320, 448, kernel_size=5, dilation=3)
        self.block6 = ResidualSEBlock1D(448, 640, kernel_size=5, dilation=3)
        self.block7 = ResidualSEBlock1D(640, 768, kernel_size=3, dilation=4)
        self.block8 = ResidualSEBlock1D(768, 768, kernel_size=3, dilation=4, pool=False)

        self.pool_head = DualPoolHead()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.40),
            nn.Linear(1536, 384),
            nn.GELU(),
            nn.Dropout(p=0.35),
            nn.Linear(384, n_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)

        features = x
        x = self.pool_head(x)
        logits = self.classifier(x)

        if return_features:
            return logits, features
        return logits