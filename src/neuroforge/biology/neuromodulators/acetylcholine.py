"""Acetylcholine signal."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AcetylcholineSignal"]


@dataclass(frozen=True, slots=True)
class AcetylcholineSignal:
    """Acetylcholine attentional modulation signal."""

    value: float = 0.0
