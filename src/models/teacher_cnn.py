#teacher_cnn.py
import torch
import torch.nn as nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        hidden = max(channels // reduction, 16)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc1 = nn.Linear(channels, hidden)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(hidden, channels)
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x).squeeze(-1)
        w = self.fc1(w)
        w = self.act(w)
        w = self.fc2(w)
        w = self.gate(w).unsqueeze(-1)
        return x * w


class ResidualSEBlock1D(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        kernel_size: int = 7,
        dilation: int = 1,
        pool: bool = True,
        use_se: bool = True,
        dropout: float = 0.0,
    ):
        super().__init__()

        padding = (kernel_size // 2) * dilation

        self.conv1 = nn.Conv1d(
            in_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_ch)
        self.act = nn.GELU()

        self.conv2 = nn.Conv1d(
            out_ch,
            out_ch,
            kernel_size=kernel_size,
            padding=padding,
            dilation=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_ch)

        self.se = SEBlock(out_ch) if use_se else nn.Identity()
        self.dropout = nn.Dropout1d(dropout) if dropout > 0 else nn.Identity()

        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_ch),
            )
        else:
            self.shortcut = nn.Identity()

        self.pool = nn.MaxPool1d(kernel_size=2, stride=2) if pool else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = self.shortcut(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = self.dropout(out)

        out = out + identity
        out = self.act(out)
        out = self.pool(out)
        return out


class DualPoolHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_feat = self.avg_pool(x).squeeze(-1)
        max_feat = self.max_pool(x).squeeze(-1)
        return torch.cat([avg_feat, max_feat], dim=1)


class TeacherCNN(nn.Module):
    def __init__(self, n_leads: int = 12, n_classes: int = 5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, 64, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(64),
            nn.GELU(),
        )

        self.block1 = ResidualSEBlock1D(64, 96, kernel_size=11)
        self.block2 = ResidualSEBlock1D(96, 128, kernel_size=9)
        self.block3 = ResidualSEBlock1D(128, 192, kernel_size=7, dilation=2)
        self.block4 = ResidualSEBlock1D(192, 256, kernel_size=7, dilation=2)
        self.block5 = ResidualSEBlock1D(256, 384, kernel_size=5, dilation=3)
        self.block6 = ResidualSEBlock1D(384, 512, kernel_size=5, dilation=3)
        self.block7 = ResidualSEBlock1D(512, 512, kernel_size=3, dilation=4, pool=False)

        self.pool_head = DualPoolHead()

        self.classifier = nn.Sequential(
            nn.Dropout(p=0.35),
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(p=0.30),
            nn.Linear(256, n_classes),
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

        features = x

        x = self.pool_head(x)
        logits = self.classifier(x)

        if return_features:
            return logits, features

        return logits