"""Surrogate-gradient spiking activation modules for vision blocks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuroforge.kernel.surrogate import surrogate_spike
from neuroforge.kernel.torch_utils import require_torch

if TYPE_CHECKING:
    from torch import nn
else:
    torch = require_torch()
    nn = torch.nn

__all__ = ["SurrogateSpike"]


class SurrogateSpike(nn.Module):
    """Element-wise hard spike with fast-sigmoid surrogate gradient."""

    def __init__(self, *, threshold: float = 0.0, beta: float = 5.0) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        self.threshold = float(threshold)
        self.beta = float(beta)

    def forward(self, x: Any) -> Any:
        """Return ``(x >= threshold)`` in forward with surrogate backward."""
        return surrogate_spike(x - self.threshold, beta=self.beta)

