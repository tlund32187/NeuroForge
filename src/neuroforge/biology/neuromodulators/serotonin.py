"""Serotonin signal."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["SerotoninSignal"]


@dataclass(frozen=True, slots=True)
class SerotoninSignal:
    """Serotonin modulation signal."""

    value: float = 0.0
