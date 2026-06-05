"""Delayed sparse static projection operator."""

from __future__ import annotations

from typing import Any

__all__ = ["delayed_sparse_static_step"]


def delayed_sparse_static_step(
    *,
    ring: Any,
    head: int,
    pre_spikes: Any,
    edge_order: Any,
    pre_idx_s: Any,
    post_idx_s: Any,
    delay_vals: Any,
    delay_ptr: Any,
    weights: Any,
) -> Any:
    """Accumulate delayed sparse edge contributions and return current slot output."""
    from neuroforge.kernel.torch_utils import require_torch

    torch = require_torch()

    ring_len = int(ring.shape[0])
    weights_s = weights[edge_order]
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
    return out
