"""Base classes for compartment models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuroforge.biology.compartments.state import CompartmentState

__all__ = ["CompartmentModelBase"]


class CompartmentModelBase(ABC):
    """Base class for compartment-local dynamics."""

    @abstractmethod
    def init_state(self, n: int, device: str, dtype: str) -> CompartmentState:
        """Create initial state for `
`` compartment instances."""
        ...

    @abstractmethod
    def step(self, state: CompartmentState, drive: Any, dt: float) -> CompartmentState:
        """Advance the compartment state by one time-step."""
        ...
