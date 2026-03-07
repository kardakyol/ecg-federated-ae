from __future__ import annotations
from dataclasses import dataclass, field
from typing import List


@dataclass
class VAEArchitectureConfig:
    """Encoder-decoder architecture hyperparameters."""
    in_channels: int = 12
    seq_len: int = 1000
    encoder_channels: List[int] = field(default_factory=lambda: [32, 64, 128, 256])
    kernel_size: int = 7
    stride: int = 2
    latent_dim: int = 32
    use_groupnorm: bool = True
    dropout: float = 0.1


@dataclass
class VAETrainingConfig:
    """Training hyperparameters."""
    batch_size: int = 64
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 100
    patience: int = 15
    lr_scheduler_patience: int = 7
    lr_scheduler_factor: float = 0.5
    min_lr: float = 1e-6
    kl_annealing_epochs: int = 20
    betas: List[float] = field(default_factory=lambda: [0.1, 0.5, 1.0])
    grad_clip_norm: float = 1.0


@dataclass
class AnomalyScoringConfig:
    """Anomaly scoring hyperparameters."""
    alpha: float = 0.0
    n_mc_samples: int = 10
    threshold_percentile: float = 95.0


@dataclass
class VAEConfig:
    """Master configuration combining all sub-configs."""
    architecture: VAEArchitectureConfig = field(default_factory=VAEArchitectureConfig)
    training: VAETrainingConfig = field(default_factory=VAETrainingConfig)
    scoring: AnomalyScoringConfig = field(default_factory=AnomalyScoringConfig)