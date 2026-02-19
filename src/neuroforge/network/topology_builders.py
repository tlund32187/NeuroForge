"""Topology builder helpers for dense and sparse projection layouts.

These helpers are pure construction functions used by ``NetworkFactory``.
They perform no file I/O and do not depend on engine state.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from neuroforge.contracts.synapses import SynapseTopology

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "build_dense_topology",
    "build_sparse_random_topology",
    "build_sparse_fanout_topology",
    "build_sparse_fanin_topology",
    "build_block_sparse_topology",
]


def build_dense_topology(
    weight_matrix: Any,
    n_pre: int,
    n_post: int,
    dev: Any,
    torch: Any,
) -> SynapseTopology:
    """Build a dense edge-list topology from a weight matrix.

    The returned ``topology.weights`` is a live view of ``weight_matrix``.
    """
    pre_idx = torch.arange(
        n_pre,
        dtype=torch.int64,
        device=dev,
    ).repeat_interleave(n_post)
    post_idx = torch.arange(
        n_post,
        dtype=torch.int64,
        device=dev,
    ).repeat(n_pre)
    delays = torch.zeros(n_pre * n_post, dtype=torch.int64, device=dev)
    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        weights=weight_matrix.reshape(-1),
        delays=delays,
        n_pre=n_pre,
        n_post=n_post,
    )


def build_sparse_random_topology(
    n_pre: int,
    n_post: int,
    p_connect: float,
    *,
    init_cfg: dict[str, Any],
    delay_cfg: dict[str, Any],
    dev: Any,
    tdt: Any,
    torch: Any,
    rng: Any,
    sort: bool = True,
) -> tuple[SynapseTopology, Tensor]:
    """Sample sparse edges independently with probability ``p_connect``."""
    if p_connect < 0.0 or p_connect > 1.0:
        msg = f"p_connect must be in [0, 1], got {p_connect!r}"
        raise ValueError(msg)

    total_edges = n_pre * n_post
    pre_all = torch.arange(n_pre, dtype=torch.int64, device="cpu").repeat_interleave(n_post)
    post_all = torch.arange(n_post, dtype=torch.int64, device="cpu").repeat(n_pre)

    if total_edges == 0 or p_connect <= 0.0:
        keep = torch.zeros(total_edges, dtype=torch.bool, device="cpu")
    elif p_connect >= 1.0:
        keep = torch.ones(total_edges, dtype=torch.bool, device="cpu")
    else:
        keep = torch.rand(total_edges, generator=rng, device="cpu") < p_connect

    pre_idx_cpu = pre_all[keep]
    post_idx_cpu = post_all[keep]
    return _finalise_sparse_topology(
        pre_idx_cpu,
        post_idx_cpu,
        n_pre=n_pre,
        n_post=n_post,
        init_cfg=init_cfg,
        delay_cfg=delay_cfg,
        dev=dev,
        tdt=tdt,
        torch=torch,
        rng=rng,
        sort=sort,
    )


def build_sparse_fanout_topology(
    n_pre: int,
    n_post: int,
    fanout: int,
    *,
    init_cfg: dict[str, Any],
    delay_cfg: dict[str, Any],
    dev: Any,
    tdt: Any,
    torch: Any,
    rng: Any,
    sort: bool = True,
) -> tuple[SynapseTopology, Tensor]:
    """Build sparse edges with fixed fanout per pre-neuron."""
    if fanout < 0:
        msg = f"fanout must be >= 0, got {fanout!r}"
        raise ValueError(msg)

    fanout_eff = min(fanout, n_post)
    if n_pre <= 0 or n_post <= 0 or fanout_eff <= 0:
        pre_idx_cpu = torch.zeros(0, dtype=torch.int64, device="cpu")
        post_idx_cpu = torch.zeros(0, dtype=torch.int64, device="cpu")
    else:
        pre_parts: list[Any] = []
        post_parts: list[Any] = []
        all_post = torch.arange(n_post, dtype=torch.int64, device="cpu")
        for pre_neuron in range(n_pre):
            if fanout_eff >= n_post:
                chosen_post = all_post
            else:
                chosen_post = torch.randperm(n_post, generator=rng, device="cpu")[:fanout_eff]
            pre_rep = torch.full(
                (fanout_eff,),
                pre_neuron,
                dtype=torch.int64,
                device="cpu",
            )
            pre_parts.append(pre_rep)
            post_parts.append(chosen_post)
        pre_idx_cpu = torch.cat(pre_parts)
        post_idx_cpu = torch.cat(post_parts)

    return _finalise_sparse_topology(
        pre_idx_cpu,
        post_idx_cpu,
        n_pre=n_pre,
        n_post=n_post,
        init_cfg=init_cfg,
        delay_cfg=delay_cfg,
        dev=dev,
        tdt=tdt,
        torch=torch,
        rng=rng,
        sort=sort,
    )


def build_sparse_fanin_topology(
    n_pre: int,
    n_post: int,
    fanin: int,
    *,
    init_cfg: dict[str, Any],
    delay_cfg: dict[str, Any],
    dev: Any,
    tdt: Any,
    torch: Any,
    rng: Any,
    sort: bool = True,
) -> tuple[SynapseTopology, Tensor]:
    """Build sparse edges with fixed fanin per post-neuron."""
    if fanin < 0:
        msg = f"fanin must be >= 0, got {fanin!r}"
        raise ValueError(msg)

    fanin_eff = min(fanin, n_pre)
    if n_pre <= 0 or n_post <= 0 or fanin_eff <= 0:
        pre_idx_cpu = torch.zeros(0, dtype=torch.int64, device="cpu")
        post_idx_cpu = torch.zeros(0, dtype=torch.int64, device="cpu")
    else:
        pre_parts: list[Any] = []
        post_parts: list[Any] = []
        all_pre = torch.arange(n_pre, dtype=torch.int64, device="cpu")
        for post_neuron in range(n_post):
            if fanin_eff >= n_pre:
                chosen_pre = all_pre
            else:
                chosen_pre = torch.randperm(n_pre, generator=rng, device="cpu")[:fanin_eff]
            post_rep = torch.full(
                (fanin_eff,),
                post_neuron,
                dtype=torch.int64,
                device="cpu",
            )
            pre_parts.append(chosen_pre)
            post_parts.append(post_rep)
        pre_idx_cpu = torch.cat(pre_parts)
        post_idx_cpu = torch.cat(post_parts)

    return _finalise_sparse_topology(
        pre_idx_cpu,
        post_idx_cpu,
        n_pre=n_pre,
        n_post=n_post,
        init_cfg=init_cfg,
        delay_cfg=delay_cfg,
        dev=dev,
        tdt=tdt,
        torch=torch,
        rng=rng,
        sort=sort,
    )


def build_block_sparse_topology(
    n_pre: int,
    n_post: int,
    block_pre: int,
    block_post: int,
    p_block: float,
    *,
    init_cfg: dict[str, Any],
    delay_cfg: dict[str, Any],
    dev: Any,
    tdt: Any,
    torch: Any,
    rng: Any,
    sort: bool = True,
) -> tuple[SynapseTopology, Tensor]:
    """Build sparse topology by sampling dense pre/post blocks."""
    if block_pre <= 0 or block_post <= 0:
        msg = f"block_pre and block_post must be > 0, got {block_pre!r}/{block_post!r}"
        raise ValueError(msg)
    if p_block < 0.0 or p_block > 1.0:
        msg = f"p_block must be in [0, 1], got {p_block!r}"
        raise ValueError(msg)

    pre_parts: list[Any] = []
    post_parts: list[Any] = []
    if n_pre > 0 and n_post > 0 and p_block > 0.0:
        for pre_start in range(0, n_pre, block_pre):
            pre_stop = min(pre_start + block_pre, n_pre)
            pre_nodes = torch.arange(pre_start, pre_stop, dtype=torch.int64, device="cpu")
            for post_start in range(0, n_post, block_post):
                if p_block < 1.0:
                    keep_block = bool(torch.rand((), generator=rng, device="cpu").item() < p_block)
                    if not keep_block:
                        continue
                post_stop = min(post_start + block_post, n_post)
                post_nodes = torch.arange(post_start, post_stop, dtype=torch.int64, device="cpu")
                pre_rep = pre_nodes.repeat_interleave(post_nodes.numel())
                post_rep = post_nodes.repeat(pre_nodes.numel())
                pre_parts.append(pre_rep)
                post_parts.append(post_rep)

    if pre_parts:
        pre_idx_cpu = torch.cat(pre_parts)
        post_idx_cpu = torch.cat(post_parts)
    else:
        pre_idx_cpu = torch.zeros(0, dtype=torch.int64, device="cpu")
        post_idx_cpu = torch.zeros(0, dtype=torch.int64, device="cpu")

    return _finalise_sparse_topology(
        pre_idx_cpu,
        post_idx_cpu,
        n_pre=n_pre,
        n_post=n_post,
        init_cfg=init_cfg,
        delay_cfg=delay_cfg,
        dev=dev,
        tdt=tdt,
        torch=torch,
        rng=rng,
        sort=sort,
    )


def _finalise_sparse_topology(
    pre_idx_cpu: Any,
    post_idx_cpu: Any,
    *,
    n_pre: int,
    n_post: int,
    init_cfg: dict[str, Any],
    delay_cfg: dict[str, Any],
    dev: Any,
    tdt: Any,
    torch: Any,
    rng: Any,
    sort: bool,
) -> tuple[SynapseTopology, Tensor]:
    """Create delays/weights, optionally sort edges, and move tensors to target device."""
    n_edges = int(pre_idx_cpu.numel())
    if int(post_idx_cpu.numel()) != n_edges:
        msg = "pre_idx and post_idx must have the same edge count"
        raise ValueError(msg)

    weights_cpu = _init_weights_cpu(n_edges, init_cfg, tdt, torch, rng)
    delays_cpu = _init_delays_cpu(n_edges, delay_cfg, torch, rng)
    pre_idx_cpu, post_idx_cpu, weights_cpu, delays_cpu = _sort_edges_if_needed(
        pre_idx_cpu,
        post_idx_cpu,
        weights_cpu,
        delays_cpu,
        delay_cfg=delay_cfg,
        torch=torch,
        sort=sort,
    )

    pre_idx = pre_idx_cpu.to(device=dev, dtype=torch.int64)
    post_idx = post_idx_cpu.to(device=dev, dtype=torch.int64)
    delays = delays_cpu.to(device=dev, dtype=torch.int64)
    weights_vec = weights_cpu.to(device=dev, dtype=tdt)
    weights_vec.requires_grad_(True)

    topo = SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        weights=weights_vec,
        delays=delays,
        n_pre=n_pre,
        n_post=n_post,
    )
    return topo, weights_vec


def _init_weights_cpu(
    n_edges: int,
    init_cfg: dict[str, Any],
    tdt: Any,
    torch: Any,
    rng: Any,
) -> Any:
    init = str(init_cfg.get("init", "uniform"))
    if init == "uniform":
        low = float(init_cfg.get("low", -0.3))
        high = float(init_cfg.get("high", 0.3))
        weights = torch.empty(n_edges, dtype=tdt, device="cpu")
        if n_edges > 0:
            weights.uniform_(low, high, generator=rng)
        return weights
    if init == "zeros":
        return torch.zeros(n_edges, dtype=tdt, device="cpu")
    msg = f"Unsupported weight init: {init!r}"
    raise ValueError(msg)


def _init_delays_cpu(
    n_edges: int,
    delay_cfg: dict[str, Any],
    torch: Any,
    rng: Any,
) -> Any:
    mode = str(delay_cfg.get("mode", "zeros"))
    if mode == "zeros":
        return torch.zeros(n_edges, dtype=torch.int64, device="cpu")
    if mode == "uniform_int":
        max_delay = max(0, int(delay_cfg.get("max_delay", 0)))
        if n_edges == 0 or max_delay == 0:
            return torch.zeros(n_edges, dtype=torch.int64, device="cpu")
        return torch.randint(
            0,
            max_delay + 1,
            (n_edges,),
            dtype=torch.int64,
            device="cpu",
            generator=rng,
        )
    msg = f"Unsupported delay mode: {mode!r}"
    raise ValueError(msg)


def _sort_edges_if_needed(
    pre_idx_cpu: Any,
    post_idx_cpu: Any,
    weights_cpu: Any,
    delays_cpu: Any,
    *,
    delay_cfg: dict[str, Any],
    torch: Any,
    sort: bool,
) -> tuple[Any, Any, Any, Any]:
    if not sort or int(pre_idx_cpu.numel()) <= 1:
        return pre_idx_cpu, post_idx_cpu, weights_cpu, delays_cpu

    order = torch.argsort(post_idx_cpu, stable=True)
    delay_mode = str(delay_cfg.get("mode", "zeros"))
    if delay_mode == "uniform_int":
        delay_sorted = delays_cpu[order]
        delay_order = torch.argsort(delay_sorted, stable=True)
        order = order[delay_order]

    return (
        pre_idx_cpu[order],
        post_idx_cpu[order],
        weights_cpu[order],
        delays_cpu[order],
    )
