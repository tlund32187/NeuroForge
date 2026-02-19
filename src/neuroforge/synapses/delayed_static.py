"""Delayed static synapse model using per-edge integer delays.

This model uses ``SynapseTopology.delays`` to route edge contributions
through a ring buffer and emit delayed post-synaptic current.

Phase 7 note:
    This implementation is currently intended for inference/benchmarks.
    If autograd is enabled, :meth:`step` raises ``NotImplementedError``.
    Training support for delayed propagation is planned for a later phase.
"""

from __future__ import annotations

from typing import Any

from neuroforge.contracts.synapses import (
    SynapseInputs,
    SynapseStepResult,
    SynapseTopology,
)
from neuroforge.contracts.types import Compartment

__all__ = ["DelayedStaticSynapseModel"]


class DelayedStaticSynapseModel:
    """Static synapse propagation with integer-step delays via ring buffer."""

    def __init__(
        self,
        compartment: str | Compartment = Compartment.SOMA,
        *,
        max_delay: int | None = None,
    ) -> None:
        if isinstance(compartment, Compartment):
            self._compartment = compartment
        else:
            self._compartment = Compartment(compartment)
        self._max_delay_override = int(max_delay) if max_delay is not None else None

    def init_state(
        self,
        topology: SynapseTopology,
        device: str,
        dtype: str,
    ) -> dict[str, Any]:
        """Initialise ring-buffer and compact delay-bucket metadata."""
        from neuroforge.core.torch_utils import require_torch

        del device, dtype
        torch = require_torch()

        weights = topology.weights
        delays = topology.delays.to(dtype=torch.int64)
        pre_idx = topology.pre_idx.to(dtype=torch.int64)
        post_idx = topology.post_idx.to(dtype=torch.int64)

        topo_max_delay = 0 if int(delays.numel()) == 0 else int(delays.max().item())
        if topo_max_delay < 0:
            msg = "Synapse delays must be non-negative"
            raise ValueError(msg)

        if self._max_delay_override is None:
            max_delay = topo_max_delay
        else:
            max_delay = int(self._max_delay_override)
            if max_delay < 0:
                msg = f"max_delay must be >= 0, got {max_delay!r}"
                raise ValueError(msg)
            if max_delay < topo_max_delay:
                msg = (
                    "max_delay override is smaller than topology delay max: "
                    f"{max_delay} < {topo_max_delay}"
                )
                raise ValueError(msg)

        ring_len = max(1, max_delay + 1)
        ring = torch.zeros(
            ring_len,
            topology.n_post,
            dtype=weights.dtype,
            device=weights.device,
        )
        head = torch.tensor(0, dtype=torch.int64, device=weights.device)

        if int(delays.numel()) > 0:
            # Stable sort by post_idx then delays -> grouped by delay with local post locality.
            order_post = torch.argsort(post_idx, stable=True)
            order = order_post[torch.argsort(delays[order_post], stable=True)]
            pre_idx_s = pre_idx[order]
            post_idx_s = post_idx[order]
            delays_s = delays[order]
            unique_delays, counts = torch.unique_consecutive(
                delays_s,
                return_counts=True,
            )
            delay_ptr = torch.cat(
                [
                    torch.zeros(1, dtype=torch.int64, device=weights.device),
                    counts.cumsum(0).to(dtype=torch.int64),
                ],
            )
        else:
            order = torch.zeros(0, dtype=torch.int64, device=weights.device)
            pre_idx_s = pre_idx
            post_idx_s = post_idx
            delays_s = delays
            unique_delays = torch.zeros(0, dtype=torch.int64, device=weights.device)
            delay_ptr = torch.zeros(1, dtype=torch.int64, device=weights.device)

        return {
            "ring": ring,
            "head": head,
            "edge_order": order,
            "pre_idx_s": pre_idx_s,
            "post_idx_s": post_idx_s,
            "delays_s": delays_s,
            "delay_vals": unique_delays,
            "delay_ptr": delay_ptr,
            "max_delay": torch.tensor(max_delay, dtype=torch.int64, device=weights.device),
        }

    def step(
        self,
        state: dict[str, Any],
        topology: SynapseTopology,
        inputs: SynapseInputs,
        ctx: object,
    ) -> SynapseStepResult:
        """Propagate spikes through delayed edge buckets."""
        from neuroforge.core.torch_utils import require_torch

        del ctx
        torch = require_torch()

        if torch.is_grad_enabled():
            msg = (
                "DelayedStaticSynapseModel does not support grad-enabled mode in Phase 7. "
                "Use torch.no_grad() for inference/benchmarks."
            )
            raise NotImplementedError(msg)

        ring = state["ring"]
        head = int(state["head"].item())
        ring_len = int(ring.shape[0])

        pre_spikes = inputs.pre_spikes
        edge_order = state["edge_order"]
        pre_idx_s = state["pre_idx_s"]
        post_idx_s = state["post_idx_s"]
        delay_vals = state["delay_vals"]
        delay_ptr = state["delay_ptr"]
        weights_s = topology.weights[edge_order]

        n_segments = int(delay_vals.numel())
        for seg_i in range(n_segments):
            start = int(delay_ptr[seg_i].item())
            end = int(delay_ptr[seg_i + 1].item())
            if end <= start:
                continue

            seg_pre = pre_idx_s[start:end]
            seg_post = post_idx_s[start:end]
            seg_weights = weights_s[start:end]
            seg_spikes = pre_spikes[seg_pre]

            if seg_spikes.dtype == torch.bool:
                contrib = torch.where(seg_spikes, seg_weights, torch.zeros_like(seg_weights))
            else:
                contrib = seg_weights * seg_spikes.to(seg_weights.dtype)

            delay_steps = int(delay_vals[seg_i].item())
            slot = (head + delay_steps) % ring_len
            ring[slot].scatter_add_(0, seg_post, contrib)

        out = ring[head].clone()
        ring[head].zero_()
        state["head"] = torch.tensor(
            (head + 1) % ring_len,
            dtype=torch.int64,
            device=ring.device,
        )

        return SynapseStepResult(post_current={self._compartment: out})

    def state_tensors(self, state: dict[str, Any]) -> dict[str, Any]:
        """Expose internal state tensors."""
        return state
