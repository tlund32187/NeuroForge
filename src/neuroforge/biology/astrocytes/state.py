"""Astrocyte state containers."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["AstrocyteState"]


@dataclass(frozen=True, slots=True)
class AstrocyteState:
    """Minimal astrocyte calcium state."""

    calcium: float = 0.0
