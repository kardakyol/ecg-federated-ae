from __future__ import annotations
import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.base import BaseAutoencoder, AEOutput
from configs.vae_config import VAEArchitectureConfig


class EncoderBlock(nn.Module):
    """Single encoder block: Conv1d -> GroupNorm -> LeakyReLU -> Dropout."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int,
                 dropout: float) -> None:
        super().__init__()
        padding = (kernel - 1) // 2
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=stride,
                              padding=padding, bias=False)
        self.norm = nn.GroupNorm(
            num_groups=min(32, out_ch), num_channels=out_ch
        )
        self.act = nn.LeakyReLU(0.2, inplace=False)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.act(self.norm(self.conv(x))))


class DecoderBlock(nn.Module):
    """Single decoder block: ConvTranspose1d -> GroupNorm -> LeakyReLU."""

    def __init__(self, in_ch: int, out_ch: int, kernel: int, stride: int,
                 is_last: bool = False) -> None:
        super().__init__()
        padding = (kernel - 1) // 2
        self.deconv = nn.ConvTranspose1d(
            in_ch, out_ch, kernel, stride=stride,
            padding=padding, output_padding=stride - 1, bias=False
        )
        self.is_last = is_last
        if not is_last:
            self.norm = nn.GroupNorm(
                num_groups=min(32, out_ch), num_channels=out_ch
            )
            self.act = nn.LeakyReLU(0.2, inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.deconv(x)
        if self.is_last:
            return x
        return self.act(self.norm(x))


class VAE(BaseAutoencoder):
    """Variational Autoencoder for 12-lead ECG anomaly detection.
    Updated with Loss Scaling and Annealing support to prevent Posterior Collapse.
    """

    def __init__(self, config: VAEArchitectureConfig | None = None) -> None:
        super().__init__()
        cfg = config or VAEArchitectureConfig()
        self.config = cfg

        # --- ENCODER ---
        channels = [cfg.in_channels] + cfg.encoder_channels
        self.encoder = nn.ModuleList([
            EncoderBlock(channels[i], channels[i + 1], cfg.kernel_size,
                         cfg.stride, cfg.dropout)
            for i in range(len(cfg.encoder_channels))
        ])

        self._enc_temporal = cfg.seq_len
        for _ in cfg.encoder_channels:
            self._enc_temporal = math.floor(
                (self._enc_temporal + 2 * ((cfg.kernel_size - 1) // 2) - cfg.kernel_size) / cfg.stride
            ) + 1
        self._flat_dim = cfg.encoder_channels[-1] * self._enc_temporal

        self.fc_mu = nn.Linear(self._flat_dim, cfg.latent_dim)
        self.fc_logvar = nn.Linear(self._flat_dim, cfg.latent_dim)

        # --- DECODER ---
        self.fc_decode = nn.Linear(cfg.latent_dim, self._flat_dim)

        dec_channels = list(reversed(cfg.encoder_channels))
        dec_channels.append(cfg.in_channels)
        self.decoder = nn.ModuleList([
            DecoderBlock(
                dec_channels[i], dec_channels[i + 1], cfg.kernel_size,
                cfg.stride, is_last=(i == len(dec_channels) - 2)
            )
            for i in range(len(dec_channels) - 1)
        ])

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = x
        for block in self.encoder:
            h = block(h)
        h = h.flatten(start_dim=1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + std * eps

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc_decode(z)
        h = h.view(-1, self.config.encoder_channels[-1], self._enc_temporal)
        for block in self.decoder:
            h = block(h)
        if h.shape[-1] != self.config.seq_len:
            h = F.interpolate(h, size=self.config.seq_len, mode='linear', align_corners=False)
        return h

    def forward(self, x: torch.Tensor) -> AEOutput:
        mu, logvar = self.encode(x)
        z = self.reparameterise(mu, logvar)
        x_hat = self.decode(z)
        return AEOutput(x_hat=x_hat, mu=mu, logvar=logvar, z=z)

    def compute_loss(self, x: torch.Tensor, output: AEOutput,
                     beta: float = 0.1, kl_weight: float = 1.0,
                     **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Beta-VAE loss with balanced scaling.
        
        Args:
            beta: KL weighting coefficient (0.1 is a good start for ECG).
            kl_weight: Can be used for annealing (0.0 at start, 1.0 at end of training).
        """
        # Reconstruction Loss (Mean over all 12000 points)
        mse = F.mse_loss(output.x_hat, x, reduction='mean')

        # KL Divergence
        kl = -0.5 * torch.sum(1 + output.logvar - output.mu.pow(2) - output.logvar.exp())

        num_features = x.shape[1] * x.shape[2] # 12 * 1000 = 12000
        kl = kl / (x.shape[0] * num_features)

        total = mse + beta * kl_weight * kl
        return total, mse, kl