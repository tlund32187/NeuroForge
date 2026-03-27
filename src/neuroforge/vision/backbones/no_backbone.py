"""Optional no-backbone adapter for vision classification pipelines."""

from __future__ import annotations

from typing import Any

from neuroforge.core.torch_utils import require_torch
from neuroforge.vision.backbones.state import VisionState

torch = require_torch()
nn = torch.nn

__all__ = ["NoBackbone"]


class NoBackbone(nn.Module):
    """Bypass backbone that maps image tensors directly to flat features."""

    def __init__(self, *, channels: int, height: int, width: int) -> None:
        super().__init__()  # pyright: ignore[reportUnknownMemberType]
        if channels <= 0 or height <= 0 or width <= 0:
            msg = "NoBackbone requires positive channels/height/width"
            raise ValueError(msg)
        self.type = "none"
        self.input_channels = int(channels)
        self.input_height = int(height)
        self.input_width = int(width)
        self.output_dim = int(channels * height * width)
        self.block_names = ("input",)

    def forward(
        self,
        x: Any,
        state: VisionState | None = None,
    ) -> tuple[Any, VisionState]:
        if x.ndim == 4:
            if int(x.shape[1]) != self.input_channels:
                msg = (
                    "Input channel mismatch: "
                    f"expected {self.input_channels}, got {int(x.shape[1])}"
                )
                raise ValueError(msg)
            spatial = x
            time_steps = 1
        elif x.ndim == 5:
            # Event-frame tensors arrive as [B, T, C, H, W]; aggregate over T.
            if int(x.shape[2]) != self.input_channels:
                msg = (
                    "Input channel mismatch: "
                    f"expected {self.input_channels}, got {int(x.shape[2])}"
                )
                raise ValueError(msg)
            time_steps = int(x.shape[1])
            if time_steps <= 0:
                msg = "Temporal dimension T must be > 0"
                raise ValueError(msg)
            spatial = x.mean(dim=1)
        else:
            msg = f"Expected [B,C,H,W] or [B,T,C,H,W], got shape {tuple(x.shape)}"
            raise ValueError(msg)

        if int(spatial.shape[2]) != self.input_height or int(spatial.shape[3]) != self.input_width:
            msg = (
                "Input spatial mismatch: expected "
                f"({self.input_height}, {self.input_width}), got "
                f"({int(spatial.shape[2])}, {int(spatial.shape[3])})"
            )
            raise ValueError(msg)

        features = spatial.reshape(int(spatial.shape[0]), -1)
        counts = features.detach().abs().sum(dim=1)
        next_state = state if state is not None else VisionState()
        next_state.per_layer_spike_counts = {"input": counts}
        next_state.per_layer_neuron_count = {"input": int(features.shape[1])}
        next_state.time_steps = int(max(1, time_steps))
        next_state.encoding_mode = "none"
        return features, next_state

