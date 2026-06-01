"""1D-CNN over raw guided-wave signals + late fusion with env features.

Input per sample:
    guided_wave: (8, 2000) float32 (already normalized per channel)
    env_features: (D,) float32

Output: scalar logit -> sigmoid -> P(damage).
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, ks: int = 7, pool: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=ks // 2)
        self.bn = nn.BatchNorm1d(out_ch)
        self.pool = nn.MaxPool1d(pool)

    def forward(self, x):
        return self.pool(F.relu(self.bn(self.conv(x))))


class CNN1D(nn.Module):
    def __init__(self, in_channels: int = 8, env_dim: int = 0, base: int = 32):
        super().__init__()
        self.b1 = ConvBlock(in_channels, base, ks=15, pool=4)
        self.b2 = ConvBlock(base, base * 2, ks=11, pool=4)
        self.b3 = ConvBlock(base * 2, base * 4, ks=7, pool=2)
        self.b4 = ConvBlock(base * 4, base * 8, ks=5, pool=2)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(base * 8 + env_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor, env: torch.Tensor | None = None) -> torch.Tensor:
        h = self.b4(self.b3(self.b2(self.b1(x))))
        h = self.gap(h).squeeze(-1)            # (B, 8*base)
        h = self.dropout(h)
        if env is not None:
            h = torch.cat([h, env], dim=1)
        return self.head(h).squeeze(-1)        # (B,)


class FocalLoss(nn.Module):
    """Binary focal loss on logits.

    Standard form: -(1-p_t)^gamma * log(p_t) with optional alpha balancing.
    """

    def __init__(self, gamma: float = 2.0, alpha: float | None = None):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p = torch.sigmoid(logits)
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ((1 - p_t) ** self.gamma) * bce
        if self.alpha is not None:
            a = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = a * loss
        return loss.mean()
