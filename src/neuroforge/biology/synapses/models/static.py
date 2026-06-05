"""Static synapse model â€” zero-delay, weight-only propagation.

For each edge (i, j) in the topology, when pre-neuron i fires the
post-synaptic current at neuron j receives a contribution of w_ij.

Math:
    I_post[j] = Î£_{i âˆˆ fired, (i,j) âˆˆ E} w_ij

This is implemented via ``scatter_add`` for O(E) efficiency.
Delays are ignored in this model (Phase 9+ adds myelinated delays).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuroforge.biology.compartments.types import Compartment
from neuroforge.biology.synapses.state import SynapseInputs, SynapseStepResult

if TYPE_CHECKING:
    from neuroforge.biology.synapses.topology import SynapseTopology

__all__ = ["StaticSynapseModel"]


class StaticSynapseModel:
    """Static synapse model â€” no plasticity, no delays.

    Computes post-synaptic current as the sum of weights of
    all edges whose pre-synaptic neuron fired.
    """

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
            When ``True``, bool-spike propagation can process only active
            edges when the active-edge ratio is low.
        active_edge_max_fraction:
            Maximum active-edge ratio allowed for the active-edge fast path.
            If the ratio is higher, the model falls back to dense masking.
        """
        self._use_active_edge_filter = bool(use_active_edge_filter)
        self._active_edge_max_fraction = float(active_edge_max_fraction)

    #

    def init_state(
        self,
        topology: SynapseTopology,
        device: str,
        dtype: str,
    ) -> dict[str, Any]:
        """Static synapse has no internal state â€” return empty dict."""
        return {}

    def step(
        self,
        state: dict[str, Any],
        topology: SynapseTopology,
        inputs: SynapseInputs,
        ctx: object,
    ) -> SynapseStepResult:
        """Propagate pre-synaptic spikes to post-synaptic currents.

        Parameters
        ----------
        state:
            Unused (static synapse has no state).
        topology:
            Wiring specification.
        inputs:
            Pre/post spike boolean tensors.
        ctx:
            Step context (unused here).

        Returns
        -------
        SynapseStepResult:
            Post-synaptic currents for SOMA compartment.
        """
        from neuroforge.kernel.torch_utils import require_torch

        torch = require_torch()

        pre_spikes = inputs.pre_spikes  # [N_pre] bool or float
        weights = topology.weights  # [E]
        pre_idx = topology.pre_idx  # [E] int
        post_idx = topology.post_idx  # [E] int
        n_post = topology.n_post

        # Mask/scale edges by pre-neuron activation.
        # Boolean spikes â†’ existing bool-index path.
        # Float spikes  â†’ multiply weight by spike value (0..1).
        spike_vals = pre_spikes[pre_idx]  # [E]
        if spike_vals.dtype == torch.bool:
            if self._use_active_edge_filter:
                active_e = spike_vals.nonzero(as_tuple=False).squeeze(1)
                n_active = int(active_e.numel())
                if n_active == 0:
                    post_current = torch.zeros(
                        n_post,
                        device=weights.device,
                        dtype=weights.dtype,
                    )
                    return SynapseStepResult(
                        post_current={Compartment.SOMA: post_current},
                    )
                n_edges = int(spike_vals.numel())
                if n_edges > 0 and (n_active / n_edges) <= self._active_edge_max_fraction:
                    contrib = weights[active_e]
                    active_post = post_idx[active_e]
                    post_current = torch.zeros(
                        n_post,
                        device=weights.device,
                        dtype=weights.dtype,
                    )
                    post_current.scatter_add_(0, active_post, contrib)
                    return SynapseStepResult(
                        post_current={Compartment.SOMA: post_current},
                    )
            contrib = torch.where(spike_vals, weights, torch.zeros_like(weights))
        else:
            contrib = weights * spike_vals.to(weights.dtype)

        # Accumulate into post-synaptic current vector
        post_current = torch.zeros(n_post, device=weights.device, dtype=weights.dtype)
        post_current.scatter_add_(0, post_idx, contrib)

        return SynapseStepResult(
            post_current={Compartment.SOMA: post_current},
        )

    def state_tensors(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return state tensors (empty for static synapse)."""
        return state

    #

    @staticmethod
    def predict_post_current(
        pre_fired: list[int],
        edges: list[tuple[int, int, float]],
        n_post: int,
    ) -> list[float]:
        """Analytically compute post-synaptic current.

        Parameters
        ----------
        pre_fired:
            Indices of pre-neurons that fired.
        edges:
            List of (pre_idx, post_idx, weight) tuples.
        n_post:
            Number of post-synaptic neurons.

        Returns
        -------
        list[float]:
            Expected post-synaptic current for each post-neuron.
        """
        result = [0.0] * n_post
        fired_set = set(pre_fired)
        for pre, post, w in edges:
            if pre in fired_set:
                result[post] += w
        return result
