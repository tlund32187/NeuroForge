"""Sparse static projection operator."""

from __future__ import annotations

from typing import Any

__all__ = ["sparse_static_current"]


def sparse_static_current(
    *,
    pre_spikes: Any,
    pre_idx: Any,
    post_idx: Any,
    weights: Any,
    n_post: int,
    use_active_edge_filter: bool = False,
    active_edge_max_fraction: float = 0.2,
) -> Any:
    """Return sparse post current using edge scatter-add."""
    from neuroforge.kernel.torch_utils import require_torch

    torch = require_torch()

    spike_vals = pre_spikes[pre_idx]
    if spike_vals.dtype == torch.bool:
        if use_active_edge_filter:
            active_e = spike_vals.nonzero(as_tuple=False).squeeze(1)
            n_active = int(active_e.numel())
            if n_active == 0:
                return torch.zeros(n_post, device=weights.device, dtype=weights.dtype)
            n_edges = int(spike_vals.numel())
            if n_edges > 0 and (n_active / n_edges) <= active_edge_max_fraction:
                post_current = torch.zeros(n_post, device=weights.device, dtype=weights.dtype)
                post_current.scatter_add_(0, post_idx[active_e], weights[active_e])
                return post_current
        contrib = torch.where(spike_vals, weights, torch.zeros_like(weights))
    else:
        contrib = weights * spike_vals.to(weights.dtype)

    post_current = torch.zeros(n_post, device=weights.device, dtype=weights.dtype)
    post_current.scatter_add_(0, post_idx, contrib)
    return post_current
