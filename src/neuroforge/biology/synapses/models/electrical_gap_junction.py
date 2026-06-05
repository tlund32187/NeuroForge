"""Electrical gap-junction synapse model extension point."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuroforge.biology.compartments.types import Compartment
from neuroforge.biology.synapses.state import SynapseStepResult

if TYPE_CHECKING:
    from neuroforge.biology.synapses.state import SynapseInputs
    from neuroforge.biology.synapses.topology import SynapseTopology

__all__ = ["ElectricalGapJunctionModel"]


class ElectricalGapJunctionModel:
    """Minimal electrical gap-junction model.

    The current implementation contributes zero post current until a voltage
    coupled model is introduced.
    """

    def init_state(
        self,
        topology: SynapseTopology,
        device: str,
        dtype: str,
    ) -> dict[str, Any]:
        """Electrical gap junctions have no state in this minimal model."""
        _ = (topology, device, dtype)
        return {}

    def step(
        self,
        state: dict[str, Any],
        topology: SynapseTopology,
        inputs: SynapseInputs,
        ctx: object,
    ) -> SynapseStepResult:
        """Return zero current for the target population."""
        from neuroforge.kernel.torch_utils import require_torch

        _ = (state, inputs, ctx)
        torch = require_torch()
        post_current = torch.zeros_like(topology.weights.new_zeros(topology.n_post))
        return SynapseStepResult(post_current={Compartment.SOMA: post_current})

    def state_tensors(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return persistable state tensors."""
        return state
