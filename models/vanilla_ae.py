import torch
import torch.nn as nn
import torch.nn.functional as F

from models.base import BaseAutoencoder, AEOutput


class VanillaAE(BaseAutoencoder):
    """Fully connected autoencoder with symmetric encoder-decoder. 
      Args:
        bottleneck: Latent dimension size. Default 32 per roadmap spec.
        n_leads: Number of ECG leads. Default 12 for PTB-XL.
        seq_len: Number of time steps per lead. Default 1000 (100 Hz x 10s).
    """

    def __init__(self, bottleneck: int = 32, n_leads: int = 12, seq_len: int = 1000):
        super().__init__()
        self.bottleneck = bottleneck
        self.n_leads = n_leads
        self.seq_len = seq_len
        self.input_dim = n_leads * seq_len  # 12000

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, 512),
            nn.ReLU(inplace=False),
            nn.Linear(512, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, bottleneck),
        )

        self.decoder = nn.Sequential(
    nn.Linear(bottleneck, 64),
    nn.ReLU(inplace=False),
    nn.Linear(64, 256),
    nn.ReLU(inplace=False),
    nn.Linear(256, 512),
    nn.ReLU(inplace=False),
    nn.Linear(512, self.input_dim),
)
        

    def forward(self, x: torch.Tensor) -> AEOutput:
        """
        Args:
            x: (B, 12, 1000) — channels-first ECG signal

        Returns:
            AEOutput with x_hat: (B, 12, 1000) — same shape as input
        """
        batch_size = x.shape[0]

        # Flatten: (B, 12, 1000) -> (B, 12000)
        x_flat = x.view(batch_size, -1)

        # Encode -> decode
        z = self.encoder(x_flat)
        x_hat_flat = self.decoder(z)

        # Reshape back: (B, 12000) -> (B, 12, 1000)
        x_hat = x_hat_flat.view(batch_size, self.n_leads, self.seq_len)

        return AEOutput(x_hat=x_hat)

    def compute_loss(self, x: torch.Tensor, output: AEOutput, **kwargs) -> tuple:
        """MSE reconstruction loss.

        Args:
            x: (B, 12, 1000) — original input
            output: AEOutput from forward()

        Returns:
            (mse,) — single-element tuple; first element is total loss for backward()
        """
        mse = F.mse_loss(output.x_hat, x)
        return (mse,)