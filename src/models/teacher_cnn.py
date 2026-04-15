import torch
import torch.nn as nn


class SEBlock(nn.Module):
    # Squeeze-and-Excitation (SE) block
    # Learns channel-wise attention weights to recalibrate feature importance
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()

        # Hidden dimension for bottleneck (reduced channel size)
        hidden = max(channels // reduction, 16)

        # Global average pooling over time dimension
        self.pool = nn.AdaptiveAvgPool1d(1)

        # Fully connected layers for channel attention
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.GELU()  # smoother activation than ReLU
        self.fc2 = nn.Linear(hidden, channels)

        # Sigmoid gate to produce attention weights in [0,1]
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Squeeze: global pooling → (B, C)
        w = self.pool(x).squeeze(-1)

        # Excitation: bottleneck MLP
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)

        # Generate attention weights and reshape
        w = self.gate(w).unsqueeze(-1)

        # Reweight input features channel-wise
        return x * w


class ConvBNAct(nn.Module):
    # Convolution → BatchNorm → GELU block with optional dilation and grouping
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__()

        # Padding adjusted for dilation to preserve length
        padding = (kernel_size // 2) * dilation

        self.block = nn.Sequential(
            nn.Conv1d(
                in_ch,
                out_ch,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False,
            ),
            nn.BatchNorm1d(out_ch),  # stabilises training
            nn.GELU(),               # smoother activation than ReLU
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class MultiScaleResidualSEBlock1D(nn.Module):
    # Multi-scale residual block with SE attention
    # Captures features at different temporal resolutions
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        branch_ratio: tuple = (0.25, 0.35, 0.40),
        k_small: int = 3,
        k_mid: int = 7,
        k_large: int = 15,
        d_small: int = 1,
        d_mid: int = 1,
        d_large: int = 2,
        stride: int = 1,
        use_se: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        # Ensure exactly 3 branches
        if len(branch_ratio) != 3:
            raise ValueError("branch_ratio must have exactly 3 values")

        # Compute channel split across branches
        b1 = max(16, int(out_ch * branch_ratio[0]))
        b2 = max(16, int(out_ch * branch_ratio[1]))
        b3 = max(16, out_ch - b1 - b2)

        # Ensure total channels sum correctly
        branch_sum = b1 + b2 + b3
        if branch_sum != out_ch:
            b3 += out_ch - branch_sum

        # Small receptive field branch (fine-grained features)
        self.branch_small = nn.Sequential(
            ConvBNAct(in_ch, b1, kernel_size=1),
            ConvBNAct(b1, b1, kernel_size=k_small, stride=stride, dilation=d_small),
        )

        # Medium receptive field branch
        self.branch_mid = nn.Sequential(
            ConvBNAct(in_ch, b2, kernel_size=1),
            ConvBNAct(b2, b2, kernel_size=k_mid, stride=stride, dilation=d_mid),
        )

        # Large receptive field branch (captures long-range dependencies)
        self.branch_large = nn.Sequential(
            ConvBNAct(in_ch, b3, kernel_size=1),
            ConvBNAct(b3, b3, kernel_size=k_large, stride=stride, dilation=d_large),
        )

        # Fuse concatenated branches
        self.fuse = nn.Sequential(
            nn.Conv1d(out_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_ch),
        )

        # Optional SE attention
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

        # Optional dropout for regularisation
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

        # Activation after residual addition
        self.act = nn.GELU()

        # Shortcut connection (projection if needed)
        if in_ch != out_ch or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Residual path
        identity = self.shortcut(x)

        # Multi-scale feature extraction
        x1 = self.branch_small(x)
        x2 = self.branch_mid(x)
        x3 = self.branch_large(x)

        # Concatenate features across channel dimension
        out = torch.cat([x1, x2, x3], dim=1)

        # Fuse features
        out = self.fuse(out)

        # Apply SE attention
        out = self.se(out)

        # Apply dropout
        out = self.dropout(out)

        # Add residual connection
        out = out + identity

        # Final activation
        out = self.act(out)
        return out


class GlobalStatsPool1D(nn.Module):
    """
    Deterministic alternative to avg+max pooling.
    Returns concatenated mean and std across time.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps  # small constant for numerical stability

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute mean across time dimension
        mean = x.mean(dim=-1)

        # Compute variance (unbiased=False for stability)
        var = x.var(dim=-1, unbiased=False)

        # Standard deviation
        std = torch.sqrt(var + self.eps)

        # Concatenate mean and std features
        return torch.cat([mean, std], dim=1)


class TeacherCNN(nn.Module):
    """
    Improved teacher for PTB-XL:
    - multi-scale residual blocks
    - SE attention
    - statistics pooling
    - cleaner and less over-regularised than the old teacher
    """

    SIZE_CONFIGS = {
    "medium": {
        "stem": 56,
        "channels": [80, 128, 176, 224, 288],
        "strides":  [1,   2,   2,   2,   2],
        "dropout":  0.04,
        "fc_dim":   224,
        "cls_drop1": 0.18,
        "cls_drop2": 0.08,
    },
    "large": {
        "stem": 80,
        "channels": [128, 192, 256, 352, 448],
        "strides":  [1,   2,   2,   2,   2],
        "dropout":  0.06,
        "fc_dim":   320,
        "cls_drop1": 0.22,
        "cls_drop2": 0.12,
    },
    "xlarge": {
        "stem": 112,
        "channels": [160, 256, 384, 512, 640],
        "strides":  [1,   2,   2,   2,   2],
        "dropout":  0.08,
        "fc_dim":   448,
        "cls_drop1": 0.25,
        "cls_drop2": 0.15,
    },
}

    def __init__(self, n_leads: int = 12, n_classes: int = 5, size: str = "large"):
        super().__init__()

        # Validate size selection
        if size not in self.SIZE_CONFIGS:
            raise ValueError(
                f"Invalid teacher size '{size}'. Must be one of {list(self.SIZE_CONFIGS.keys())}"
            )

        # Extract configuration
        cfg = self.SIZE_CONFIGS[size]
        stem_ch = cfg["stem"]
        channels = cfg["channels"]
        strides = cfg["strides"]
        block_dropout = cfg["dropout"]
        fc_dim = cfg["fc_dim"]

        self.size = size

        # Initial convolutional stem (captures low-level features)
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, stem_ch, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(stem_ch),
            nn.GELU(),
            nn.Conv1d(stem_ch, stem_ch, kernel_size=7, padding=3, bias=False),
            nn.BatchNorm1d(stem_ch),
            nn.GELU(),
        )

        # Build sequence of multi-scale residual blocks
        blocks = []
        in_ch = stem_ch

        # Increasing dilation for long-range temporal dependencies
        long_dilations = [1, 2, 2, 3, 4]

        for i, (out_ch, stride) in enumerate(zip(channels, strides)):
            blocks.append(
                MultiScaleResidualSEBlock1D(
                    in_ch=in_ch,
                    out_ch=out_ch,
                    branch_ratio=(0.25, 0.35, 0.40),
                    k_small=3,
                    k_mid=7,
                    k_large=15,
                    d_small=1,
                    d_mid=1 if i < 2 else 2,
                    d_large=long_dilations[i],
                    stride=stride,
                    use_se=True,
                    dropout=block_dropout,
                )
            )
            in_ch = out_ch

        # Stack all blocks
        self.blocks = nn.Sequential(*blocks)

        # Statistics pooling (mean + std)
        self.pool_head = GlobalStatsPool1D()

        # Output dimension after pooling (mean + std doubles channels)
        pooled_dim = channels[-1] * 2

        # Final classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(p=cfg["cls_drop1"]),
            nn.Linear(pooled_dim, fc_dim),
            nn.GELU(),
            nn.Dropout(p=cfg["cls_drop2"]),
            nn.Linear(fc_dim, n_classes),
        )

    def forward(self, x: torch.Tensor, return_features: bool = False):
        # Initial feature extraction
        x = self.stem(x)

        # Deep feature extraction via multi-scale residual blocks
        x = self.blocks(x)

        # Store features (useful for knowledge distillation)
        features = x

        # Pool features into fixed-length representation
        pooled = self.pool_head(x)

        # Classification logits
        logits = self.classifier(pooled)

        # Optionally return features for KD
        if return_features:
            return logits, features

        return logits