"""Static synapse model.

Dense projections use a matrix-backed operator. Sparse projections use an
edge-list scatter operator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuroforge.biology.compartments.types import Compartment
from neuroforge.biology.synapses.state import SynapseInputs, SynapseStepResult

if TYPE_CHECKING:
    from neuroforge.biology.synapses.topology import SynapseTopology

__all__ = ["StaticSynapseModel"]


class StaticSynapseModel:
    """Static synapse model with no plasticity and no delay state."""

    def __init__(
        self,
        *,
        use_active_edge_filter: bool = False,
        active_edge_max_fraction: float = 0.2,
        **_kwargs: object,
    ) -> None:  # noqa: B027
        """Create static synapse model.

        Parameters
        ----------
        use_active_edge_filter:
            When ``True``, bool-spike sparse propagation can process only
            active edges when the active-edge ratio is low.
        active_edge_max_fraction:
            Maximum active-edge ratio allowed for the sparse active-edge fast
            path. If the ratio is higher, the model falls back to dense masking.
        """
        self._use_active_edge_filter = bool(use_active_edge_filter)
        self._active_edge_max_fraction = float(active_edge_max_fraction)

    def init_state(
        self,
        topology: SynapseTopology,
        device: str,
        dtype: str,
    ) -> dict[str, Any]:
        """Static synapse has no internal state."""
        del topology, device, dtype
        return {}

    def step(
        self,
        state: dict[str, Any],
        topology: SynapseTopology,
        inputs: SynapseInputs,
        ctx: object,
    ) -> SynapseStepResult:
        """Propagate pre-synaptic spikes to post-synaptic currents."""
        del state, ctx

        pre_spikes = inputs.pre_spikes
        if topology.weight_matrix is not None:
            from neuroforge.biology.synapses.operators.dense_static import dense_static_current

            post_current = dense_static_current(pre_spikes, topology.weight_matrix)
        else:
            from neuroforge.biology.synapses.operators.sparse_static import sparse_static_current

            post_current = sparse_static_current(
                pre_spikes=pre_spikes,
                pre_idx=topology.pre_idx,
                post_idx=topology.post_idx,
                weights=topology.weights,
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

    @staticmethod
    def predict_post_current(
        pre_fired: list[int],
        edges: list[tuple[int, int, float]],
        n_post: int,
    ) -> list[float]:
        """Analytically compute post-synaptic current."""
        result = [0.0] * n_post
        fired_set = set(pre_fired)
        for pre, post, weight in edges:
            if pre in fired_set:
                result[post] += weight
        return result
