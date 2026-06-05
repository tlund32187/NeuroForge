"""Base receptor model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["ReceptorModelBase"]


class ReceptorModelBase(ABC):
    """Base class for receptor models."""

    @abstractmethod
    def current(self, activation: Any) -> Any:
        """Convert receptor activation to current."""
        ...
