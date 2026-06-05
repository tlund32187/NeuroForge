"""Neuromodulator field state."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["NeuromodulatorField"]


@dataclass(frozen=True, slots=True)
class NeuromodulatorField:
    """Scalar neuromodulator field value."""

    name: str
    value: float = 0.0
