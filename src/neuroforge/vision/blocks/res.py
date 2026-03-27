"""Residual spiking blocks for vision backbones."""

from __future__ import annotations

from typing import Any

from neuroforge.core.torch_utils import require_torch
from neuroforge.vision.blocks.conv import SpikingConvBlock, build_2d_norm
from neuroforge.vision.blocks.spike import SurrogateSpike

torch = require_torch()
nn = torch.nn

__all__ = ["SpikingResBlock"]


class SpikingResBlock(nn.Module):
    """Two spiking conv blocks with residual skip connection."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        padding: int | tuple[int, int] | None = None,
        bias: bool = False,
        norm: str | None = "batch",
        norm_groups: int = 8,
        spike_threshold: float = 0.0,
        spike_beta: float = 5.0,
        downsample: Any | None = None,
    ) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]

        self.block1 = SpikingConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
            norm=norm,
            norm_groups=norm_groups,
            spike_threshold=spike_threshold,
            spike_beta=spike_beta,
        )
        self.block2 = SpikingConvBlock(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=padding,
            bias=bias,
            norm=norm,
            norm_groups=norm_groups,
            spike_threshold=spike_threshold,
            spike_beta=spike_beta,
        )
        self.post_add_spike = SurrogateSpike(
            threshold=spike_threshold,
            beta=spike_beta,
        )

        if downsample is not None:
            self.downsample = downsample
        elif stride != 1 or in_channels != out_channels:
            down_layers: list[Any] = [
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    padding=0,
                    bias=False,
                )
            ]
            down_norm = build_2d_norm(norm, channels=out_channels, num_groups=norm_groups)
            if down_norm is not None:
                down_layers.append(down_norm)
            self.downsample = nn.Sequential(*down_layers)
        else:
            self.downsample = nn.Identity()

    def forward(self, x: Any) -> Any:
        identity = self.downsample(x)
        out = self.block1(x)
        out = self.block2(out)
        if out.shape != identity.shape:
            msg = (
                "SpikingResBlock shape mismatch between residual and skip paths: "
                f"main={tuple(out.shape)} skip={tuple(identity.shape)}"
            )
            raise ValueError(msg)
        return self.post_add_spike(out + identity)

