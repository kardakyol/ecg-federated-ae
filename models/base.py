"""
BASE AUTOENCODER INTERFACE - DO NOT MODIFY without team consensus.

Shardul extends for VanillaAE/ConvAE, Kaan extends for VAE.
Raheeb calls get_parameters/set_parameters for Flower FedAvg.
Ghadah calls forward() for quantisation benchmarks.
Hilal calls compute_loss() through Opacus DP-SGD wrapper.
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn


@dataclass
class AEOutput:
    """Standard output for ALL autoencoder forward passes.
    x_hat is REQUIRED (B, 12, 1000). mu/logvar/z are VAE-only (Kaan).
    Shardul leaves optional fields as None. Downstream code only uses x_hat."""
    x_hat: torch.Tensor
    mu: torch.Tensor | None = None
    logvar: torch.Tensor | None = None
    z: torch.Tensor | None = None


class BaseAutoencoder(nn.Module, ABC):
    """Abstract base. Extend this, implement forward() and compute_loss().

    Downstream usage:
        Raheeb: model.get_parameters() / model.set_parameters(params)
        Ghadah: output = model(x); model.model_size_mb()
        Hilal:  loss, *_ = model.compute_loss(x, output); loss.backward()
    """

    @abstractmethod
    def forward(self, x: torch.Tensor) -> AEOutput:
        """x: (B,12,1000) -> AEOutput. x_hat.shape MUST equal x.shape."""
        ...

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, output: AEOutput, **kwargs) -> Tuple[torch.Tensor, ...]:
        """[0]=total_loss for backward. Shardul: (mse,) Kaan: (total,mse,kl)"""
        ...

    def get_parameters(self) -> List[np.ndarray]:
        """For Raheeb's Flower FedAvg."""
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """For Raheeb's Flower FedAvg. np.copy prevents buffer reuse bugs."""
        for param, new_val in zip(self.parameters(), parameters):
            param.data = torch.from_numpy(np.copy(new_val)).to(param.device)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        """FP32 size in MB. Ghadah uses for efficiency table."""
        return self.count_parameters() * 4 / (1024 ** 2)
