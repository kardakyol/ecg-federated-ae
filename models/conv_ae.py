"""
1D Convolutional Autoencoder — Updated Sprint 3
=================================================
Best config from ablation: bottleneck=128, lr=1e-3
AUROC: 0.795 ± 0.004 on PTB-XL z-score data (max_auroc_pipeline)

Changes from Sprint 2:
  - Default bottleneck: 32 → 128 (ablation proved larger = better)
  - No other architectural changes (backward compatible)
  - Still uses GroupNorm + inplace=False (Opacus compatible)
  - AEOutput interface unchanged (Raheeb/Ghadah/Hilal unaffected)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from models.base import BaseAutoencoder, AEOutput


class ConvAE(BaseAutoencoder):
    """1D Convolutional Autoencoder.

    Args:
        bottleneck: Latent dimension size. Default 128 (Sprint 3 best).
                    Ablation tested {8, 16, 32, 64, 128}.
        n_leads: Number of ECG leads. Default 12 for PTB-XL.
        seq_len: Number of time steps per lead. Default 1000 (100 Hz x 10s).
    """

    def __init__(self, bottleneck: int = 128, n_leads: int = 12, seq_len: int = 1000):
        super().__init__()
        self.bottleneck = bottleneck
        self.n_leads = n_leads
        self.seq_len = seq_len

        # --- Encoder ---
        # (B, 12, 1000) -> (B, 32, 500)
        self.enc_conv1 = nn.Conv1d(n_leads, 32, kernel_size=7, stride=2, padding=3)
        self.enc_gn1 = nn.GroupNorm(num_groups=min(32, 32), num_channels=32)
        self.enc_act1 = nn.ReLU(inplace=False)

        # (B, 32, 500) -> (B, 64, 250)
        self.enc_conv2 = nn.Conv1d(32, 64, kernel_size=7, stride=2, padding=3)
        self.enc_gn2 = nn.GroupNorm(num_groups=min(32, 64), num_channels=64)
        self.enc_act2 = nn.ReLU(inplace=False)

        # (B, 64, 250) -> (B, 128, 125)
        self.enc_conv3 = nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2)
        self.enc_gn3 = nn.GroupNorm(num_groups=min(32, 128), num_channels=128)
        self.enc_act3 = nn.ReLU(inplace=False)

        # (B, 128, 125) -> (B, 256, 63)
        self.enc_conv4 = nn.Conv1d(128, 256, kernel_size=5, stride=2, padding=2)
        self.enc_gn4 = nn.GroupNorm(num_groups=min(32, 256), num_channels=256)
        self.enc_act4 = nn.ReLU(inplace=False)

        curr_len = seq_len
        for k, s, p in [(7, 2, 3), (7, 2, 3), (5, 2, 2), (5, 2, 2)]:
            curr_len = math.floor((curr_len + 2*p - k) / s) + 1

        self._enc_temporal = curr_len
        self._enc_flat_dim = 256 * self._enc_temporal

        # Flatten -> FC bottleneck
        self.enc_fc = nn.Linear(self._enc_flat_dim, bottleneck)
        self.enc_fc_act = nn.ReLU(inplace=False)

        # --- Decoder ---
        # FC -> reshape
        self.dec_fc = nn.Linear(bottleneck, self._enc_flat_dim)
        self.dec_fc_act = nn.ReLU(inplace=False)

        # (B, 256, 63) -> (B, 128, 125)
        self.dec_conv1 = nn.ConvTranspose1d(256, 128, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_gn1 = nn.GroupNorm(num_groups=min(32, 128), num_channels=128)
        self.dec_act1 = nn.ReLU(inplace=False)

        # (B, 128, 125) -> (B, 64, 250)
        self.dec_conv2 = nn.ConvTranspose1d(128, 64, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.dec_gn2 = nn.GroupNorm(num_groups=min(32, 64), num_channels=64)
        self.dec_act2 = nn.ReLU(inplace=False)

        # (B, 64, 250) -> (B, 32, 500)
        self.dec_conv3 = nn.ConvTranspose1d(64, 32, kernel_size=7, stride=2, padding=3, output_padding=1)
        self.dec_gn3 = nn.GroupNorm(num_groups=min(32, 32), num_channels=32)
        self.dec_act3 = nn.ReLU(inplace=False)

        # (B, 32, 500) -> (B, 12, 1000)
        self.dec_conv4 = nn.ConvTranspose1d(32, n_leads, kernel_size=7, stride=2, padding=3, output_padding=1)

    def forward(self, x: torch.Tensor) -> AEOutput:
        # Encode
        h = self.enc_act1(self.enc_gn1(self.enc_conv1(x)))
        h = self.enc_act2(self.enc_gn2(self.enc_conv2(h)))
        h = self.enc_act3(self.enc_gn3(self.enc_conv3(h)))
        h = self.enc_act4(self.enc_gn4(self.enc_conv4(h)))
        h_flat = h.view(h.shape[0], -1)
        z = self.enc_fc_act(self.enc_fc(h_flat))

        # Decode
        h = self.dec_fc_act(self.dec_fc(z))
        h = h.view(h.size(0), 256, -1)
        h = self.dec_act1(self.dec_gn1(self.dec_conv1(h)))
        h = self.dec_act2(self.dec_gn2(self.dec_conv2(h)))
        h = self.dec_act3(self.dec_gn3(self.dec_conv3(h)))
        x_hat = self.dec_conv4(h)

        if x_hat.shape[-1] != x.shape[-1]:
            x_hat = F.interpolate(x_hat, size=x.shape[-1], mode='linear', align_corners=False)

        return AEOutput(x_hat=x_hat)

    def compute_loss(self, x: torch.Tensor, output: AEOutput, **kwargs) -> tuple:
        mse = F.mse_loss(output.x_hat, x)
        return (mse,)