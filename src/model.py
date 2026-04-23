import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block: learns per-channel importance weights."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, C, T]
        w = self.pool(x).squeeze(-1)   # [B, C]
        w = self.fc(w).unsqueeze(-1)   # [B, C, 1]
        return x * w


class ResBlock1D(nn.Module):
    """Residual block with two Conv1d layers and an SE block."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 15, stride: int = 1):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.se = SEBlock(out_channels)
        self.dropout = nn.Dropout(0.2)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out = self.dropout(out)
        return F.relu(out + self.shortcut(x))


class ECGResNet(nn.Module):
    """
    1D ResNet for 12-lead ECG binary classification.

    Input:  [B, 12, 4000]
    Output: [B] logits (use BCEWithLogitsLoss)
    """

    def __init__(self, n_leads: int = 12, base_channels: int = 64, n_blocks: int = 4):
        super().__init__()

        # Stem: project 12 leads into base_channels
        self.stem = nn.Sequential(
            nn.Conv1d(n_leads, base_channels, kernel_size=15, padding=7, bias=False),
            nn.BatchNorm1d(base_channels),
            nn.ReLU(),
            nn.MaxPool1d(2),  # 4000 → 2000
        )

        # Stack residual blocks, doubling channels every two blocks, halving time dim
        channels = [base_channels * (2 ** (i // 2)) for i in range(n_blocks)]
        strides = [2 if i > 0 else 1 for i in range(n_blocks)]

        blocks = []
        in_ch = base_channels
        for out_ch, stride in zip(channels, strides):
            blocks.append(ResBlock1D(in_ch, out_ch, stride=stride))
            in_ch = out_ch
        self.blocks = nn.ModuleList(blocks)

        self.gap = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(in_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for block in self.blocks:
            x = block(x)
        x = self.gap(x).squeeze(-1)  # [B, C]
        return self.head(x).squeeze(-1)  # [B]

    def get_se_weights(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Return SE channel weights from each block for one forward pass."""
        weights = []
        x = self.stem(x)
        for block in self.blocks:
            # Run up to SE, capture weights
            out = F.relu(block.bn1(block.conv1(x)))
            out = block.bn2(block.conv2(out))
            w = block.se.pool(out).squeeze(-1)
            w = block.se.fc(w)
            weights.append(w.detach())
            out = block.se(out)
            out = block.dropout(out)
            x = F.relu(out + block.shortcut(x))
        return weights
