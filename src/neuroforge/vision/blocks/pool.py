"""Pooling blocks for spiking vision backbones."""

from __future__ import annotations

from typing import Any

from neuroforge.core.torch_utils import require_torch

torch = require_torch()
nn = torch.nn

__all__ = ["SpikingPool"]


class SpikingPool(nn.Module):
    """Wrapper for MaxPool2d / AvgPool2d."""

    def __init__(
        self,
        *,
        mode: str = "max",
        kernel_size: int | tuple[int, int] = 2,
        stride: int | tuple[int, int] | None = None,
        padding: int | tuple[int, int] = 0,
    ) -> None:
        super().__init__()
        pool_mode = mode.strip().lower()
        if pool_mode == "max":
            self.pool = nn.MaxPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        elif pool_mode == "avg":
            self.pool = nn.AvgPool2d(
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
        else:
            msg = f"Unsupported pooling mode: {mode!r}. Use 'max' or 'avg'."
            raise ValueError(msg)
        self.mode = pool_mode

    def forward(self, x: Any) -> Any:
        return self.pool(x)

