"""Surrogate-gradient spiking activation modules for vision blocks."""

from __future__ import annotations

from typing import Any

from neuroforge.core.surrogate import surrogate_spike
from neuroforge.core.torch_utils import require_torch

torch = require_torch()
nn = torch.nn

__all__ = ["SurrogateSpike"]


class SurrogateSpike(nn.Module):
    """Element-wise hard spike with fast-sigmoid surrogate gradient."""

    def __init__(self, *, threshold: float = 0.0, beta: float = 5.0) -> None:
        super().__init__()
        self.threshold = float(threshold)
        self.beta = float(beta)

    def forward(self, x: Any) -> Any:
        """Return ``(x >= threshold)`` in forward with surrogate backward."""
        return surrogate_spike(x - self.threshold, beta=self.beta)

