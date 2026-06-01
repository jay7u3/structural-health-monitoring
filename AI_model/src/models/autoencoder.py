"""Convolutional autoencoder trained on healthy guided-wave signals only.

Reconstruction error becomes an anomaly score for the long-term health profile.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ConvAE(nn.Module):
    def __init__(self, in_channels: int = 8, base: int = 32, latent_dim: int = 64):
        super().__init__()
        # Encoder
        self.enc = nn.Sequential(
            nn.Conv1d(in_channels, base, 15, stride=2, padding=7), nn.BatchNorm1d(base), nn.ReLU(),
            nn.Conv1d(base, base * 2, 11, stride=2, padding=5), nn.BatchNorm1d(base * 2), nn.ReLU(),
            nn.Conv1d(base * 2, base * 4, 7, stride=2, padding=3), nn.BatchNorm1d(base * 4), nn.ReLU(),
            nn.Conv1d(base * 4, base * 8, 5, stride=2, padding=2), nn.BatchNorm1d(base * 8), nn.ReLU(),
        )
        self.bottleneck_in = nn.AdaptiveAvgPool1d(1)
        self.fc_enc = nn.Linear(base * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, base * 8 * 125)  # 2000 / 16 = 125
        self.dec = nn.Sequential(
            nn.ConvTranspose1d(base * 8, base * 4, 5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(base * 4), nn.ReLU(),
            nn.ConvTranspose1d(base * 4, base * 2, 7, stride=2, padding=3, output_padding=1),
            nn.BatchNorm1d(base * 2), nn.ReLU(),
            nn.ConvTranspose1d(base * 2, base, 11, stride=2, padding=5, output_padding=1),
            nn.BatchNorm1d(base), nn.ReLU(),
            nn.ConvTranspose1d(base, in_channels, 15, stride=2, padding=7, output_padding=1),
        )
        self._base = base

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        h = self.enc(x)
        h = self.bottleneck_in(h).squeeze(-1)
        return self.fc_enc(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_dec(z).view(-1, self._base * 8, 125)
        return self.dec(h)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


def reconstruction_anomaly_score(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE over channels and time -> (B,)."""
    return ((x - x_hat) ** 2).mean(dim=(1, 2))
