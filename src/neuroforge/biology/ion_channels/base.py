"""Base ion-channel model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["IonChannelBase"]


class IonChannelBase(ABC):
    """Base class for ion-channel current models."""

    @abstractmethod
    def current(self, voltage: Any) -> Any:
        """Compute channel current from voltage."""
        ...
