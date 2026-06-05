"""Axon compartment parameters."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AxonParams"]


@dataclass(frozen=True, slots=True)
class AxonParams:
    """Parameters for axonal propagation."""

    delay_steps: int = 0
