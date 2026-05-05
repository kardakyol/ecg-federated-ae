from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Tuple
import numpy as np
import torch
import torch.nn as nn


@dataclass
class AEOutput:
    """Standard output for ALL autoencoder forward passes."""
    x_hat: torch.Tensor
    mu: torch.Tensor | None = None
    logvar: torch.Tensor | None = None
    z: torch.Tensor | None = None


class BaseAutoencoder(nn.Module, ABC):
    """Abstract base. Extend this, implement forward() and compute_loss()."""

    @abstractmethod
    def forward(self, x: torch.Tensor) -> AEOutput:
        """x: (B,12,1000) -> AEOutput. x_hat.shape MUST equal x.shape."""
        ...

    @abstractmethod
    def compute_loss(self, x: torch.Tensor, output: AEOutput, **kwargs) -> Tuple[torch.Tensor, ...]:
        """[0]=total_loss for backward."""
        ...

    def get_parameters(self) -> List[np.ndarray]:
        return [p.detach().cpu().numpy() for p in self.parameters()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        for param, new_val in zip(self.parameters(), parameters):
            param.data = torch.from_numpy(np.copy(new_val)).to(param.device)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self) -> float:
        return self.count_parameters() * 4 / (1024 ** 2)
