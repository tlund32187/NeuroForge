"""Plasticity trace state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

__all__ = ["PlasticityTrace"]


@dataclass(slots=True)
class PlasticityTrace:
    """Named tensor trace used by plasticity rules."""

    name: str
    value: Any
