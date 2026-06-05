"""Runtime state containers for vision backbones."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

__all__ = ["VisionState"]


@dataclass(slots=True)
class VisionState:
    """Per-forward summary state for a vision backbone.

    Attributes
    ----------
    per_layer_spike_counts:
        Mapping ``layer_name -> [B]`` spike totals accumulated over time.
    per_layer_neuron_count:
        Mapping ``layer_name -> neurons_per_sample`` for the current forward.
    time_steps:
        Number of encoded time steps processed in the current forward.
    encoding_mode:
        Encoding mode used for the current forward pass.
    """

    per_layer_spike_counts: dict[str, Any] = field(default_factory=lambda: {})
    per_layer_neuron_count: dict[str, int] = field(default_factory=lambda: {})
    time_steps: int = 0
    encoding_mode: str = "rate"
