"""Receptor contracts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["IReceptorModel"]


@runtime_checkable
class IReceptorModel(Protocol):
    """Protocol for receptor current models."""

    def current(self, activation: Any) -> Any:
        """Convert receptor activation to current."""
        ...
