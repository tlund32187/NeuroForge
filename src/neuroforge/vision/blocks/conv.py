"""Spiking convolution block primitives for vision backbones."""

from __future__ import annotations

from typing import Any

from neuroforge.core.torch_utils import require_torch
from neuroforge.vision.blocks.spike import SurrogateSpike

torch = require_torch()
nn = torch.nn

__all__ = ["SpikingConvBlock", "build_2d_norm"]


def _to_pair(value: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(value, int):
        return (value, value)
    return value


def build_2d_norm(
    norm: str | None,
    *,
    channels: int,
    num_groups: int = 8,
) -> Any | None:
    """Build a 2D norm layer for convolution outputs."""
    if norm is None:
        return None
    mode = norm.strip().lower()
    if mode in {"", "none"}:
        return None
    if mode in {"batch", "batchnorm", "bn"}:
        return nn.BatchNorm2d(channels)
    if mode in {"group", "groupnorm", "gn"}:
        groups = max(1, min(int(num_groups), int(channels)))
        while channels % groups != 0 and groups > 1:
            groups -= 1
        return nn.GroupNorm(groups, channels)
    msg = f"Unsupported norm: {norm!r}. Use one of [None, 'batch', 'group']."
    raise ValueError(msg)


class SpikingConvBlock(nn.Module):
    """``Conv2d -> optional Norm -> SurrogateSpike`` block."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | None = None,
        bias: bool = True,
        norm: str | None = None,
        norm_groups: int = 8,
        spike_threshold: float = 0.0,
        spike_beta: float = 5.0,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        if padding is None:
            k_h, k_w = _to_pair(kernel_size)
            padding = (k_h // 2, k_w // 2)

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.norm = build_2d_norm(
            norm,
            channels=out_channels,
            num_groups=norm_groups,
        )
        self.spike = SurrogateSpike(threshold=spike_threshold, beta=spike_beta)

    def forward(self, x: Any) -> Any:
        y = self.conv(x)
        if self.norm is not None:
            y = self.norm(y)
        return self.spike(y)

