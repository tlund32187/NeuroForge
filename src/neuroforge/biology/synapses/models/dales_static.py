"""Dale's-law static synapse model."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuroforge.biology.compartments.types import Compartment
from neuroforge.biology.synapses.state import SynapseInputs, SynapseStepResult

if TYPE_CHECKING:
    from neuroforge.biology.synapses.topology import SynapseTopology

__all__ = ["DalesStaticSynapseModel"]


class DalesStaticSynapseModel:
    """Static synapse model with Dale's-law weight reparameterisation."""

    def __init__(
        self,
        sign_pre: Any = None,
        *,
        use_active_edge_filter: bool = False,
        active_edge_max_fraction: float = 0.2,
        **_kwargs: object,
    ) -> None:
        self._sign_pre = sign_pre
        self._use_active_edge_filter = bool(use_active_edge_filter)
        self._active_edge_max_fraction = float(active_edge_max_fraction)

    def init_state(
        self,
        topology: SynapseTopology,
        device: str,
        dtype: str,
    ) -> dict[str, Any]:
        """Dale synapse has no internal state."""
        del topology, device, dtype
        return {}

    def step(
        self,
        state: dict[str, Any],
        topology: SynapseTopology,
        inputs: SynapseInputs,
        ctx: object,
    ) -> SynapseStepResult:
        """Propagate spikes through Dale-constrained weights."""
        del state, ctx

        pre_spikes = inputs.pre_spikes
        if topology.weight_matrix is not None:
            from neuroforge.biology.synapses.operators.dales_dense import dales_dense_current

            post_current = dales_dense_current(
                pre_spikes,
                topology.weight_matrix,
                self._sign_pre,
            )
        else:
            from neuroforge.biology.synapses.operators.sparse_static import sparse_static_current
            from neuroforge.kernel.torch_utils import require_torch

            torch = require_torch()
            weights = topology.weights
            if self._sign_pre is None:
                effective_weights = weights
            else:
                sign_pre = self._sign_pre.to(device=weights.device)
                sign_e = sign_pre[topology.pre_idx]
                effective_weights = torch.abs(weights) * sign_e.to(weights.dtype)

            post_current = sparse_static_current(
                pre_spikes=pre_spikes,
                pre_idx=topology.pre_idx,
                post_idx=topology.post_idx,
                weights=effective_weights,
                n_post=topology.n_post,
                use_active_edge_filter=self._use_active_edge_filter,
                active_edge_max_fraction=self._active_edge_max_fraction,
            )

        return SynapseStepResult(
            post_current={Compartment.SOMA: post_current},
        )

    def state_tensors(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return state tensors."""
        return state
