"""Base classes for biological plasticity rules."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__all__ = ["PlasticityRuleBase"]


class PlasticityRuleBase(ABC):
    """Base class for biological plasticity rules."""

    @abstractmethod
    def init_state(self, n_edges: int, device: str, dtype: str) -> dict[str, Any]:
        """Allocate rule state for `
_edges`` synapses."""
        ...

    @abstractmethod
    def state_tensors(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return persistable state tensors."""
        ...
