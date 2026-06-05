"""Dopamine signal."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["DopamineSignal"]


@dataclass(frozen=True, slots=True)
class DopamineSignal:
    """Dopamine reward-modulation signal."""

    value: float = 0.0
