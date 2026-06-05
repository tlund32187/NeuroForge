"""Soma compartment parameters."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SomaParams"]


@dataclass(frozen=True, slots=True)
class SomaParams:
    """Parameters for soma compartment dynamics."""

    capacitance: float = 1.0
    leak_conductance: float = 0.0
