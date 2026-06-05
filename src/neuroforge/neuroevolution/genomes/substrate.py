"""HyperNEAT substrate geometry."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

__all__ = ["DEFAULT_SUBSTRATE", "Substrate", "SubstrateConfig"]


@dataclass(frozen=True, slots=True)
class SubstrateConfig:
    """Geometric layout queried by a HyperNEAT CPPN."""

    input_shape: tuple[int, int, int] = (2, 28, 32)
    hidden_shape: tuple[int, int] = (16, 16)
    leo: bool = True
    weight_threshold: float = 0.05

    def __post_init__(self) -> None:
        channels, height, width = self.input_shape
        hidden_h, hidden_w = self.hidden_shape
        if channels < 1 or height < 1 or width < 1:
            msg = "SubstrateConfig.input_shape dimensions must be >= 1"
            raise ValueError(msg)
        if hidden_h < 1 or hidden_w < 1:
            msg = "SubstrateConfig.hidden_shape dimensions must be >= 1"
            raise ValueError(msg)
        if self.weight_threshold < 0.0:
            msg = "SubstrateConfig.weight_threshold must be >= 0"
            raise ValueError(msg)

    def input_count(self) -> int:
        """Number of input coordinates in the configured grid."""
        channels, height, width = self.input_shape
        return int(channels * height * width)

    def query_dim(self) -> int:
        """Feature count per queried connection."""
        return 5

    def resolve_input_grid(self, n_input: int) -> tuple[int, int, int]:
        """Return the configured grid, or a square fallback for another encoder."""
        if int(n_input) == self.input_count():
            return self.input_shape
        side = max(1, int(round(math.sqrt(int(n_input)))))
        if side * side == int(n_input):
            return (1, side, side)
        height = side
        width = max(1, math.ceil(int(n_input) / height))
        return (1, height, width)


DEFAULT_SUBSTRATE = SubstrateConfig()


class Substrate:
    """Coordinate generator for HyperNEAT phenotype compilation."""

    def __init__(
        self,
        config: SubstrateConfig,
        *,
        n_input: int,
        n_buttons: int,
        motor_per_button: int,
        torch: Any,
        dev: Any,
        dtype: Any,
    ) -> None:
        self.config = config
        self._n_input = int(n_input)
        self._n_buttons = int(n_buttons)
        self._motor_per_button = int(motor_per_button)
        self._torch = torch
        self._dev = dev
        self._dtype = dtype

    def count(self, layer: str) -> int:
        """Return neuron count for a substrate layer."""
        if layer == "input":
            return self._n_input
        if layer == "hidden":
            hidden_h, hidden_w = self.config.hidden_shape
            return int(hidden_h * hidden_w)
        if layer == "motor":
            return int(self._n_buttons * self._motor_per_button)
        msg = f"unknown substrate layer {layer!r}"
        raise KeyError(msg)

    def connection_pairs(self) -> tuple[tuple[str, str], ...]:
        """Canonical feed-forward substrate projections."""
        return (("input", "hidden"), ("hidden", "motor"))

    def query_features(self, src: str, dst: str) -> Any:
        """Return pre-major coordinate features for all potential ``src -> dst`` links."""
        pre = self._coords(src)
        post = self._coords(dst)
        n_pre = int(pre.shape[0])
        n_post = int(post.shape[0])
        pre_rep = pre.repeat_interleave(n_post, dim=0)
        post_rep = post.repeat(n_pre, 1)
        dx = pre_rep[:, 0] - post_rep[:, 0]
        dy = pre_rep[:, 1] - post_rep[:, 1]
        dist = self._torch.sqrt(dx * dx + dy * dy)
        return self._torch.stack(
            [pre_rep[:, 0], pre_rep[:, 1], post_rep[:, 0], post_rep[:, 1], dist],
            dim=1,
        ).to(device=self._dev, dtype=self._dtype)

    def _coords(self, layer: str) -> Any:
        torch = self._torch
        if layer == "input":
            channels, height, width = self.config.resolve_input_grid(self._n_input)
            xs, ys = _grid_xy(width, height, torch=torch, dev=self._dev, dtype=self._dtype)
            coords = torch.stack([xs, ys], dim=1).repeat(channels, 1)
            return coords[: self._n_input]
        if layer == "hidden":
            height, width = self.config.hidden_shape
            xs, ys = _grid_xy(width, height, torch=torch, dev=self._dev, dtype=self._dtype)
            return torch.stack([xs, ys], dim=1)
        if layer == "motor":
            total = self.count("motor")
            xs = torch.linspace(-1.0, 1.0, total, device=self._dev, dtype=self._dtype)
            ys = torch.ones(total, device=self._dev, dtype=self._dtype)
            return torch.stack([xs, ys], dim=1)
        msg = f"unknown substrate layer {layer!r}"
        raise KeyError(msg)


def _grid_xy(width: int, height: int, *, torch: Any, dev: Any, dtype: Any) -> tuple[Any, Any]:
    xs = torch.linspace(-1.0, 1.0, width, device=dev, dtype=dtype)
    ys = torch.linspace(-1.0, 1.0, height, device=dev, dtype=dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    return xx.reshape(-1), yy.reshape(-1)
