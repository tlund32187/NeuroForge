"""Base class for synapse models."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuroforge.biology.synapses.state import SynapseInputs, SynapseStepResult
    from neuroforge.biology.synapses.topology import SynapseTopology

__all__ = ["SynapseModelBase"]


class SynapseModelBase(ABC):
    """Base class for edge-population synapse models."""

    @abstractmethod
    def init_state(
        self,
        topology: SynapseTopology,
        device: str,
        dtype: str,
    ) -> dict[str, Any]:
        """Allocate synapse-specific state."""
        ...

    @abstractmethod
    def step(
        self,
        state: dict[str, Any],
        topology: SynapseTopology,
        inputs: SynapseInputs,
        ctx: object,
    ) -> SynapseStepResult:
        """Propagate spikes through the projection for one step."""
        ...

    def state_tensors(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return persistable state tensors."""
        return state
