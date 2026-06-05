"""Parameters for Leaky Integrate-and-Fire neuron models."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LIFParams"]


@dataclass(frozen=True, slots=True)
class LIFParams:
    """Parameters for the LIF neuron model."""

    tau_mem: float = 20e-3
    v_thresh: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0
