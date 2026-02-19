"""Dale's-Law static synapse model.

Wraps the standard static-synapse propagation but applies effective
weights via the Dale reparameterisation::

    w_eff = |w_raw| × sign_pre[pre_idx]

This ensures excitatory neurons produce only non-negative post-synaptic
currents and inhibitory neurons produce only non-positive currents.

The ``sign_pre`` tensor is stored on the model instance at construction
time and must have shape ``[n_pre]``.
"""

from __future__ import annotations

from typing import Any

from neuroforge.contracts.synapses import (
    SynapseInputs,
    SynapseStepResult,
    SynapseTopology,
)
from neuroforge.contracts.types import Compartment

__all__ = ["DalesStaticSynapseModel"]


class DalesStaticSynapseModel:
    """Static synapse model with Dale's Law weight reparameterisation.

    Parameters
    ----------
    sign_pre:
        Tensor ``[n_pre]`` of ``+1.0`` / ``-1.0`` per pre-synaptic
        neuron.  If ``None``, defaults to all-excitatory on first use.
    """

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

    # ── ISynapseModel implementation ────────────────────────────────

    def init_state(
        self,
        topology: SynapseTopology,
        device: str,
        dtype: str,
    ) -> dict[str, Any]:
        """Dale synapse has no internal state — return empty dict."""
        return {}

    def step(
        self,
        state: dict[str, Any],
        topology: SynapseTopology,
        inputs: SynapseInputs,
        ctx: object,
    ) -> SynapseStepResult:
        """Propagate spikes through Dale-constrained weights.

        The effective weight for edge *e* is::

            w_eff[e] = |weights[e]| × sign_pre[pre_idx[e]]
        """
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()

        pre_spikes = inputs.pre_spikes  # [N_pre] bool or float
        weights = topology.weights  # [E]
        pre_idx = topology.pre_idx  # [E] int
        post_idx = topology.post_idx  # [E] int
        n_post = topology.n_post

        # Apply Dale reparameterisation: |w| × sign.
        if self._sign_pre is not None:
            sign_e = self._sign_pre[pre_idx]  # [E]
            w_eff = torch.abs(weights) * sign_e.to(weights.dtype)
        else:
            w_eff = weights

        # Spike contribution — float or bool.
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
                    contrib = w_eff[active_e]
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
            contrib = torch.where(spike_vals, w_eff, torch.zeros_like(w_eff))
        else:
            contrib = w_eff * spike_vals.to(w_eff.dtype)

        # Accumulate post-synaptic current.
        post_current = torch.zeros(n_post, device=weights.device, dtype=weights.dtype)
        post_current.scatter_add_(0, post_idx, contrib)

        return SynapseStepResult(
            post_current={Compartment.SOMA: post_current},
        )

    def state_tensors(self, state: dict[str, Any]) -> dict[str, Any]:
        """Return state tensors (empty for this model)."""
        return state
