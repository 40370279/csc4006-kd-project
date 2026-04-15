import torch
import torch.nn as nn


class ConvBNAct(nn.Module):
    # Basic building block: Convolution → BatchNorm → ReLU
    # Common pattern in CNNs for stable and efficient training
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1):
        super().__init__()

        # Padding ensures output length is preserved (same padding)
        padding = kernel_size // 2

        # Sequential block combining conv, normalisation, and activation
        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=False,  # bias not needed due to BatchNorm
            ),
            nn.BatchNorm1d(out_ch),  # stabilises training
            nn.ReLU(inplace=True),   # non-linearity
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pass input through conv → BN → ReLU block
        return self.block(x)


class ResidualBlock1D(nn.Module):
    # Residual block inspired by ResNet architecture
    # Helps with gradient flow and deeper network training
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, dropout: float = 0.0):
        super().__init__()

        # First convolution block (Conv → BN → ReLU)
        self.conv1 = ConvBNAct(in_ch, out_ch, kernel_size=5, stride=stride)

        # Second convolution (no activation here; applied after residual addition)
        self.conv2 = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )

        # Activation applied after adding residual connection
        self.act = nn.ReLU(inplace=True)

        # Optional dropout for regularisation (identity if dropout = 0)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Define shortcut (skip connection)
        # If dimensions change (channels or stride), use projection
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            # If dimensions match, use identity mapping
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Save input for residual connection
        identity = self.shortcut(x)

        # Main convolution path
        out = self.conv1(x)
        out = self.dropout(out)  # apply dropout after first conv
        out = self.conv2(out)

        # Add residual (skip connection)
        out = out + identity

        # Apply activation after addition
        out = self.act(out)

        return out


class StudentCNN(nn.Module):
    # Lightweight CNN designed for efficient ECG classification
    # Supports multiple capacity configurations (small, medium, large)

    SIZE_CONFIGS = {
        # Defines channel sizes, dropout rates, and classifier dimensions
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

        # Validate size choice
        if size not in self.SIZE_CONFIGS:
            raise ValueError(
                f"Invalid student size '{size}'. Must be one of {list(self.SIZE_CONFIGS.keys())}"
            )

        # Extract configuration for selected model size
        cfg = self.SIZE_CONFIGS[size]
        c1, c2, c3, c4 = cfg["channels"]  # channel sizes for each stage
        drop = cfg["dropout"]             # dropout rate
        head_dim = cfg["head_dim"]        # hidden size for classifier

        # Store feature dimensions for potential use (e.g., KD feature matching)
        self.feature_dim = c4
        self.head_dim = head_dim

        # Initial convolution layer (reduces temporal resolution via stride=2)
        self.stem = nn.Sequential(
            ConvBNAct(n_leads, c1, kernel_size=5, stride=2),
        )

        # Residual stages (progressively increase channels and downsample)
        self.stage1 = ResidualBlock1D(c1, c1, stride=1, dropout=drop * 0.5)
        self.stage2 = ResidualBlock1D(c1, c2, stride=2, dropout=drop * 0.5)
        self.stage3 = ResidualBlock1D(c2, c3, stride=2, dropout=drop)
        self.stage4 = ResidualBlock1D(c3, c4, stride=2, dropout=drop)

        # Global average pooling to reduce time dimension → 1
        self.avg_pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected classifier head
        self.classifier = nn.Sequential(
            nn.Linear(c4, head_dim),   # project features to hidden dimension
            nn.ReLU(inplace=True),     # non-linearity
            nn.Dropout(drop),          # regularisation
            nn.Linear(head_dim, n_classes),  # output logits for classification
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # Initial convolution
        x = self.stem(x)

        # Pass through residual stages
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)

        # Final feature representation
        features = self.stage4(x)

        # Global pooling → flatten to (batch, channels)
        pooled = self.avg_pool(features).flatten(1)

        # Classification logits
        logits = self.classifier(pooled)

        # Optionally return intermediate features (used in knowledge distillation)
        if return_features:
            return logits, pooled

        return logits