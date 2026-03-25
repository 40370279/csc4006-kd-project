import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock1D(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()

        self.conv1 = ConvBNAct(in_ch, out_ch, kernel_size=5, stride=stride)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + identity
        out = self.act(out)
        return out


class StudentCNN(nn.Module):
    SIZE_CONFIGS = {
        "small": {
            "channels": [6, 10, 14, 18],
            "dropout": 0.12,
            "head_dim": 16,
        },
        "medium": {
            "channels": [10, 14, 20, 28],
            "dropout": 0.14,
            "head_dim": 28,
        },
        "large": {
            "channels": [14, 20, 28, 40],
            "dropout": 0.16,
            "head_dim": 48,
        },
    }

    def __init__(self, n_leads: int = 12, n_classes: int = 5, size: str = "small"):
        super().__init__()

        if size not in self.SIZE_CONFIGS:
            raise ValueError(
                f"Invalid student size '{size}'. Must be one of {list(self.SIZE_CONFIGS.keys())}"
            )

        cfg = self.SIZE_CONFIGS[size]
        c1, c2, c3, c4 = cfg["channels"]
        drop = cfg["dropout"]
        head_dim = cfg["head_dim"]

        self.feature_dim = c4
        self.head_dim = head_dim

        self.stem = nn.Sequential(
            ConvBNAct(n_leads, c1, kernel_size=5, stride=2),
        )

        self.stage1 = ResidualBlock1D(c1, c1, stride=1, dropout=drop * 0.5)
        self.stage2 = ResidualBlock1D(c1, c2, stride=2, dropout=drop * 0.5)
        self.stage3 = ResidualBlock1D(c2, c3, stride=2, dropout=drop)
        self.stage4 = ResidualBlock1D(c3, c4, stride=2, dropout=drop)

        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        self.classifier = nn.Sequential(
            nn.Linear(c4, head_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(drop),
            nn.Linear(head_dim, n_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        features = self.stage4(x)

        pooled = self.avg_pool(features).flatten(1)
        logits = self.classifier(pooled)

        if return_features:
            return logits, pooled

        return logits