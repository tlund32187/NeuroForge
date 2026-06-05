"""Neuromodulator contracts."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NeuromodulatorSignal"]


@dataclass(frozen=True, slots=True)
class NeuromodulatorSignal:
    """Scalar neuromodulator signal DTO."""

    name: str
    value: float = 0.0
