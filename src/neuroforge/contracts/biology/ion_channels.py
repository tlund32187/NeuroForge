"""Ion-channel contracts."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

__all__ = ["IIonChannel"]


@runtime_checkable
class IIonChannel(Protocol):
    """Protocol for ion-channel current models."""

    def current(self, voltage: Any) -> Any:
        """Compute current from voltage."""
        ...
