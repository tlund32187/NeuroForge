"""Static synapse model — zero-delay, weight-only propagation.

For each edge (i, j) in the topology, when pre-neuron i fires the
post-synaptic current at neuron j receives a contribution of w_ij.

Math:
    I_post[j] = Σ_{i ∈ fired, (i,j) ∈ E} w_ij

This is implemented via ``scatter_add`` for O(E) efficiency.
Delays are ignored in this model (Phase 9+ adds myelinated delays).
"""

from __future__ import annotations

from typing import Any

from neuroforge.contracts.synapses import (
    SynapseInputs,
    SynapseStepResult,
    SynapseTopology,
)
from neuroforge.contracts.types import Compartment

__all__ = ["StaticSynapseModel"]


class StaticSynapseModel:
    """Static synapse model — no plasticity, no delays.

    Computes post-synaptic current as the sum of weights of
    all edges whose pre-synaptic neuron fired.
    """

    # ── ISynapseModel implementation ────────────────────────────────

    def init_state(
        self,
        topology: SynapseTopology,
        device: str,
        dtype: str,
    ) -> dict[str, Any]:
        """Static synapse has no internal state — return empty dict."""
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
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()

        pre_spikes = inputs.pre_spikes  # [N_pre] bool
        weights = topology.weights  # [E]
        pre_idx = topology.pre_idx  # [E] int
        post_idx = topology.post_idx  # [E] int
        n_post = topology.n_post

        # Mask edges where pre-neuron fired
        active = pre_spikes[pre_idx]  # [E] bool

        # Weighted contribution from active edges
        contrib = torch.where(active, weights, torch.zeros_like(weights))  # [E]

        # Accumulate into post-synaptic current vector
        post_current = torch.zeros(n_post, device=weights.device, dtype=weights.dtype)
        post_current.scatter_add_(0, post_idx, contrib)

        return SynapseStepResult(
            post_current={Compartment.SOMA: post_current},
        )

    def state_tensors(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return state tensors (empty for static synapse)."""
        return state

    # ── math helpers (public for testing) ──────────────────────────

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
