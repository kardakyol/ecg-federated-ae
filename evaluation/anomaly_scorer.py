from __future__ import annotations
import logging
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from models.base import BaseAutoencoder

logger = logging.getLogger(__name__)


@torch.no_grad()
def compute_anomaly_scores(
    model: BaseAutoencoder,
    loader: DataLoader,
    alpha: float = 0.0,
    n_mc_samples: int = 10,
    device: torch.device | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute anomaly scores for all samples in a DataLoader."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    all_scores = []
    all_labels = []

    for batch_signals, batch_labels in loader:
        batch_signals = batch_signals.to(device)
        batch_size = batch_signals.shape[0]

        batch_scores = torch.zeros(batch_size, device=device)

        # Determine if model is a VAE (has stochastic sampling)
        test_output = model(batch_signals)
        is_vae = test_output.mu is not None
        effective_mc = n_mc_samples if is_vae else 1

        for mc_idx in range(effective_mc):
            output = model(batch_signals) if mc_idx > 0 else test_output

            # Per-sample MSE: mean over channels and timesteps
            per_sample_mse = torch.mean(
                (output.x_hat - batch_signals) ** 2, dim=(1, 2)
            )

            if alpha > 0 and is_vae and output.mu is not None:
                logvar = torch.clamp(output.logvar, min=-20.0, max=2.0)
                per_sample_kl = -0.5 * torch.sum(
                    1 + logvar - output.mu.pow(2) - logvar.exp(), dim=1
                )
                batch_scores += per_sample_mse + alpha * per_sample_kl
            else:
                batch_scores += per_sample_mse

        batch_scores /= effective_mc
        all_scores.append(batch_scores.cpu().numpy())
        all_labels.append(batch_labels.numpy())

    return np.concatenate(all_scores), np.concatenate(all_labels)


@torch.no_grad()
def calibrate_threshold(
    model: BaseAutoencoder,
    val_normal_loader: DataLoader,
    percentile: float = 95.0,
    alpha: float = 0.5,
    n_mc_samples: int = 10,
    device: torch.device | None = None,
) -> float:
    """Calibrate anomaly threshold on normal validation data."""
    scores, _ = compute_anomaly_scores(
        model, val_normal_loader, alpha=alpha,
        n_mc_samples=n_mc_samples, device=device,
    )
    threshold = float(np.percentile(scores, percentile))
    logger.info(
        f"Threshold calibrated: {threshold:.6f} "
        f"({percentile}th percentile of {len(scores)} normal validation scores)"
    )
    return threshold