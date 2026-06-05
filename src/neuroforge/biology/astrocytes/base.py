"""Base astrocyte model."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.biology.astrocytes.state import AstrocyteState

__all__ = ["AstrocyteModelBase"]


class AstrocyteModelBase(ABC):
    """Base class for astrocyte models."""

    @abstractmethod
    def step(self, state: AstrocyteState, dt: float) -> AstrocyteState:
        """Advance astrocyte state one step."""
        ...
