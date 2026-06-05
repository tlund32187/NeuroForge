"""Eligibility trace helpers."""

from __future__ import annotations

import math

__all__ = ["eligibility_decay"]


def eligibility_decay(dt: float, tau: float) -> float:
    """Compute exponential eligibility decay for one step."""
    return math.exp(-dt / tau)
