from __future__ import annotations
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader

from models.vae import VAE
from configs.vae_config import VAETrainingConfig

logger = logging.getLogger(__name__)


class EarlyStopping:
    """Early stopping to halt training when validation loss stops improving."""

    def __init__(self, patience: int = 15, delta: float = 1e-6) -> None:
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_loss = float('inf')
        self.should_stop = False

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


class VAETrainer:
    """Centralised VAE training with KL annealing, early stopping, and checkpointing."""

    def __init__(
        self,
        model: VAE,
        config: VAETrainingConfig,
        device: torch.device,
        checkpoint_dir: Optional[str] = None,
    ) -> None:
        self.model = model.to(device)
        self.config = config
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _get_kl_weight(self, epoch: int) -> float:
        """Compute KL annealing weight for the current epoch."""
        if self.config.kl_annealing_epochs <= 0:
            return 1.0
        return min(1.0, epoch / self.config.kl_annealing_epochs)

    def _train_epoch(self, loader: DataLoader, optimizer: torch.optim.Optimizer,
                     beta: float, kl_weight: float) -> Dict[str, float]:
        """Run one training epoch. Returns dict of mean losses."""
        self.model.train()
        total_loss_sum = 0.0
        mse_sum = 0.0
        kl_sum = 0.0
        n_batches = 0

        for signals, _labels in loader:
            signals = signals.to(self.device)
            optimizer.zero_grad()

            output = self.model(signals)
            total_loss, mse_loss, kl_loss = self.model.compute_loss(
                signals, output, beta=beta, kl_weight=kl_weight
            )

            total_loss.backward()

            # Gradient clipping prevents exploding gradients, especially
            # during the KL annealing phase when loss landscape is steep
            if self.config.grad_clip_norm > 0:
                nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.grad_clip_norm
                )

            optimizer.step()

            total_loss_sum += total_loss.item()
            mse_sum += mse_loss.item()
            kl_sum += kl_loss.item()
            n_batches += 1

        return {
            "total_loss": total_loss_sum / n_batches,
            "mse_loss": mse_sum / n_batches,
            "kl_loss": kl_sum / n_batches,
        }

    @torch.no_grad()
    def _validate(self, loader: DataLoader, beta: float,
                  kl_weight: float) -> Dict[str, float]:
        """Run validation. Returns dict of mean losses."""
        self.model.eval()
        total_loss_sum = 0.0
        mse_sum = 0.0
        kl_sum = 0.0
        n_batches = 0

        for signals, _labels in loader:
            signals = signals.to(self.device)
            output = self.model(signals)
            total_loss, mse_loss, kl_loss = self.model.compute_loss(
                signals, output, beta=beta, kl_weight=kl_weight
            )
            total_loss_sum += total_loss.item()
            mse_sum += mse_loss.item()
            kl_sum += kl_loss.item()
            n_batches += 1

        return {
            "total_loss": total_loss_sum / n_batches,
            "mse_loss": mse_sum / n_batches,
            "kl_loss": kl_sum / n_batches,
        }

    def _save_checkpoint(self, beta: float, epoch: int) -> Path:
        """Save model checkpoint. Returns path to saved file."""
        if self.checkpoint_dir is None:
            return None
        path = self.checkpoint_dir / f"vae_beta{beta}_best.pt"
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "epoch": epoch,
            "beta": beta,
        }, path)
        return path

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        beta: float = 1.0,
    ) -> Dict[str, List[float]]:
        """Full training loop with KL annealing, early stopping, LR scheduling."""
        optimizer = Adam(
            self.model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min',
            patience=self.config.lr_scheduler_patience,
            factor=self.config.lr_scheduler_factor,
            min_lr=self.config.min_lr,
        )
        early_stop = EarlyStopping(patience=self.config.patience)

        history = {
            "train_total_loss": [], "train_mse_loss": [], "train_kl_loss": [],
            "val_total_loss": [], "val_mse_loss": [], "val_kl_loss": [],
            "kl_weight": [], "lr": [], "epoch_time_s": [],
        }

        logger.info(f"Training VAE | beta={beta} | device={self.device} | "
                     f"params={self.model.count_parameters():,}")

        best_val_mse = float('inf')

        for epoch in range(self.config.epochs):
            t0 = time.time()
            kl_weight = self._get_kl_weight(epoch)
            current_lr = optimizer.param_groups[0]["lr"]

            train_metrics = self._train_epoch(
                train_loader, optimizer, beta, kl_weight
            )
            val_metrics = self._validate(val_loader, beta, kl_weight)

            scheduler.step(val_metrics["total_loss"])
            epoch_time = time.time() - t0

            # Record history
            for key in ["total_loss", "mse_loss", "kl_loss"]:
                history[f"train_{key}"].append(train_metrics[key])
                history[f"val_{key}"].append(val_metrics[key])
            history["kl_weight"].append(kl_weight)
            history["lr"].append(current_lr)
            history["epoch_time_s"].append(epoch_time)

            # Logging
            logger.info(
                f"Epoch {epoch+1:3d}/{self.config.epochs} | "
                f"Train: {train_metrics['total_loss']:.6f} "
                f"(MSE={train_metrics['mse_loss']:.6f} KL={train_metrics['kl_loss']:.4f}) | "
                f"Val: {val_metrics['total_loss']:.6f} | "
                f"KLw={kl_weight:.2f} LR={current_lr:.2e} | "
                f"{epoch_time:.1f}s"
            )

            # Checkpoint best model
            if val_metrics["mse_loss"] < best_val_mse:
                best_val_mse = val_metrics["mse_loss"]
                ckpt_path = self._save_checkpoint(beta, epoch)
                if ckpt_path:
                    logger.info(f"  New best model saved: {ckpt_path}")

            # Early stopping
            if early_stop.step(val_metrics["mse_loss"]):
                logger.info(
                    f"Early stopping at epoch {epoch+1} "
                    f"(no improvement for {self.config.patience} epochs)"
                )
                break

        # Load best checkpoint back into model
        if self.checkpoint_dir:
            ckpt_path = self.checkpoint_dir / f"vae_beta{beta}_best.pt"
            if ckpt_path.exists():
                checkpoint = torch.load(
                    ckpt_path, map_location=self.device, weights_only=False
                )
                self.model.load_state_dict(checkpoint["model_state_dict"])
                logger.info(f"Loaded best model from epoch {checkpoint.get('epoch', -1) + 1}")

        return history
