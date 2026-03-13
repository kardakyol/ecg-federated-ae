from dataclasses import dataclass, field
from typing import List


@dataclass
class AEConfig:
    """Vanilla AE and Conv AE training Hyperparameters.

    Updated Sprint 3: bottleneck=128, epochs=200 based on ablation results.
    """

    bottleneck: int = 128
    n_leads: int = 12
    seq_len: int = 1000

    batch_size: int = 64
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 1.0

    # CosineAnnealingWarmRestarts params
    cosine_T0: int = 20
    cosine_T_mult: int = 2
    cosine_eta_min: float = 1e-6

    # Early stopping
    patience: int = 25

    data_dir: str = "data/ptb-xl-zscore"

    checkpoint_dir: str = "checkpoints"
    output_dir: str = "outputs"
    figure_dir: str = "outputs/figures"

    bottleneck_ablation: List[int] = field(default_factory=lambda: [8, 16, 32, 64, 128])