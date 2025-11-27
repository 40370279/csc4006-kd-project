import torch
import torch.nn as nn
from typing import Optional


class SEBlock(nn.Module):
    """
    Squeeze-and-Excitation (SE) block for 1D feature maps.

    Applies channel-wise attention:
      - Global average pool over time
      - Two-layer MLP (reduction -> expansion)
      - Sigmoid gating per channel
    """
    def __init__(self, channels, reduction=8):
        super(SEBlock, self).__init__()
        hidden = max(channels // reduction, 4)
        self.net = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # (B, C, 1)
            nn.Flatten(),              # (B, C)
            nn.Linear(channels, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (B, C, T)
        w = self.net(x)          # (B, C)
        w = w.unsqueeze(-1)      # (B, C, 1)
        return x * w             # scale each channel


class ConvBlock1D(nn.Module):
    """
    Basic 1D conv block: Conv -> BN -> ReLU -> optional SE -> optional MaxPool.
    """
    def __init__(
        self,
        in_ch,
        out_ch,
        kernel_size=7,
        stride=1,
        padding=None,
        pool=True,
        use_se=True,
    ):
        super(ConvBlock1D, self).__init__()
        if padding is None:
            padding = kernel_size // 2

        self.conv = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm1d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        x = self.pool(x)
        return x


class TeacherCNN(nn.Module):
    """
    Stronger Teacher CNN for 12-lead ECG classification on PTB-XL.

    Architecture:
      - 5 convolutional blocks with increasing channels and SE attention
      - Progressive temporal downsampling via MaxPool
      - Global average pooling over time
      - 2-layer MLP classifier with dropout

    Input
    -----
    x : torch.Tensor of shape (batch_size, n_leads, seq_len)

    Output
    ------
    logits : torch.Tensor of shape (batch_size, n_classes)
    """

    def __init__(self, n_leads=12, n_classes=5):
        super(TeacherCNN, self).__init__()

        # For 5000 samples, 4 pooling layers with kernel_size=2
        # give ~5000 / 2^4 = ~312 time steps before global pooling.
        self.block1 = ConvBlock1D(
            in_ch=n_leads,
            out_ch=64,
            kernel_size=9,
            pool=True,
            use_se=True,
        )
        self.block2 = ConvBlock1D(
            in_ch=64,
            out_ch=128,
            kernel_size=7,
            pool=True,
            use_se=True,
        )
        self.block3 = ConvBlock1D(
            in_ch=128,
            out_ch=256,
            kernel_size=5,
            pool=True,
            use_se=True,
        )
        self.block4 = ConvBlock1D(
            in_ch=256,
            out_ch=384,
            kernel_size=5,
            pool=True,
            use_se=True,
        )
        # final block without pooling (keep temporal resolution before GAP)
        self.block5 = ConvBlock1D(
            in_ch=384,
            out_ch=512,
            kernel_size=3,
            pool=False,
            use_se=True,
        )

        self.global_pool = nn.AdaptiveAvgPool1d(output_size=1)

        self.classifier = nn.Sequential(
            nn.Flatten(),         # (B, 512, 1) -> (B, 512)
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            nn.Linear(256, n_classes),
        )

    def forward(self, x):
        # x: (B, n_leads, T)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.global_pool(x)      # (B, 512, 1)
        x = self.classifier(x)       # (B, n_classes)
        return x
