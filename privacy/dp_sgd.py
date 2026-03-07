"""
HILAL: Opacus DP-SGD integration.

CRITICAL - Opacus constraints:
    BatchNorm CANNOT be used (no per-sample gradients).
    inplace=True CANNOT be used.
    Shardul and Kaan: use GroupNorm, inplace=False.

Model interface (from models/base.py):
    loss, *_ = model.compute_loss(x, output)
    loss.backward()  # Opacus hooks into gradients here
"""
# TODO: Hilal implementation
from __future__ import annotations

from typing import Any

from opacus import PrivacyEngine


def make_private(
    model: Any,
    optimizer: Any,
    dataloader: Any,
    noise_multiplier: float = 1.0,
    max_grad_norm: float = 1.0,
    poisson_sampling: bool = True,
):
    """
    Wrap a model, optimizer, and dataloader with Opacus to enable DP-SGD.

    Args:
        model: PyTorch model to privatize.
        optimizer: Optimizer used for training.
        dataloader: Training dataloader.
        noise_multiplier: Gaussian noise multiplier for DP-SGD.
        max_grad_norm: Per-sample gradient clipping norm.
        poisson_sampling: Whether to use Poisson sampling.

    Returns:
        tuple: (private_model, private_optimizer, private_dataloader, privacy_engine)
    """
    if noise_multiplier < 0:
        raise ValueError("noise_multiplier must be non-negative.")
    if max_grad_norm <= 0:
        raise ValueError("max_grad_norm must be positive.")

    privacy_engine = PrivacyEngine()

    private_model, private_optimizer, private_dataloader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=dataloader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=max_grad_norm,
        poisson_sampling=poisson_sampling,
    )

    return private_model, private_optimizer, private_dataloader, privacy_engine


def get_epsilon(privacy_engine: PrivacyEngine, delta: float):
    """
    Return the current epsilon value for a given delta.

    Args:
        privacy_engine: Opacus PrivacyEngine instance.
        delta: Target delta value for (epsilon, delta)-DP.

    Returns:
        Current epsilon value, or None if privacy_engine is None.
    """
    if privacy_engine is None:
        return None
    if not (0 < delta < 1):
        raise ValueError("delta must be in the interval (0, 1).")

    return privacy_engine.get_epsilon(delta=delta)
