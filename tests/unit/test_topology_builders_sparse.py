"""Sparse topology builder tests."""

from __future__ import annotations

from typing import Any

import pytest

from neuroforge.core.torch_utils import require_torch
from neuroforge.network.topology_builders import (
    build_block_sparse_topology,
    build_sparse_fanin_topology,
    build_sparse_fanout_topology,
    build_sparse_random_topology,
)


def _rng(seed: int, torch: Any) -> Any:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return gen


def _assert_index_bounds(topo: Any, n_pre: int, n_post: int, torch: Any) -> None:
    assert topo.pre_idx.dtype == torch.int64
    assert topo.post_idx.dtype == torch.int64
    assert topo.delays.dtype == torch.int64

    n_edges = int(topo.pre_idx.numel())
    assert int(topo.post_idx.numel()) == n_edges
    assert int(topo.weights.numel()) == n_edges
    assert int(topo.delays.numel()) == n_edges

    if n_edges == 0:
        return
    assert int(topo.pre_idx.min().item()) >= 0
    assert int(topo.pre_idx.max().item()) < n_pre
    assert int(topo.post_idx.min().item()) >= 0
    assert int(topo.post_idx.max().item()) < n_post


@pytest.mark.unit
def test_sparse_random_bounds_and_edge_count() -> None:
    torch = require_torch()
    topo, weights = build_sparse_random_topology(
        7,
        5,
        0.35,
        init_cfg={"init": "uniform", "low": -0.1, "high": 0.2},
        delay_cfg={"mode": "uniform_int", "max_delay": 4},
        dev=torch.device("cpu"),
        tdt=torch.float32,
        torch=torch,
        rng=_rng(11, torch),
        sort=True,
    )
    n_edges = int(topo.pre_idx.numel())
    assert 0 <= n_edges <= 35
    _assert_index_bounds(topo, 7, 5, torch)
    assert weights.requires_grad
    assert topo.weights.requires_grad


@pytest.mark.unit
def test_sparse_fanout_edge_count_and_bounds() -> None:
    torch = require_torch()
    n_pre, n_post, fanout = 6, 4, 2
    topo, _weights = build_sparse_fanout_topology(
        n_pre,
        n_post,
        fanout,
        init_cfg={"init": "zeros"},
        delay_cfg={"mode": "zeros"},
        dev=torch.device("cpu"),
        tdt=torch.float32,
        torch=torch,
        rng=_rng(22, torch),
        sort=True,
    )
    assert int(topo.pre_idx.numel()) == n_pre * fanout
    _assert_index_bounds(topo, n_pre, n_post, torch)


@pytest.mark.unit
def test_sparse_fanin_edge_count_and_bounds() -> None:
    torch = require_torch()
    n_pre, n_post, fanin = 5, 7, 3
    topo, _weights = build_sparse_fanin_topology(
        n_pre,
        n_post,
        fanin,
        init_cfg={"init": "zeros"},
        delay_cfg={"mode": "zeros"},
        dev=torch.device("cpu"),
        tdt=torch.float32,
        torch=torch,
        rng=_rng(33, torch),
        sort=True,
    )
    assert int(topo.pre_idx.numel()) == n_post * fanin
    _assert_index_bounds(topo, n_pre, n_post, torch)


@pytest.mark.unit
def test_block_sparse_bounds_and_edge_count() -> None:
    torch = require_torch()
    topo, _weights = build_block_sparse_topology(
        8,
        6,
        2,
        3,
        0.5,
        init_cfg={"init": "uniform", "low": -0.3, "high": 0.3},
        delay_cfg={"mode": "uniform_int", "max_delay": 3},
        dev=torch.device("cpu"),
        tdt=torch.float32,
        torch=torch,
        rng=_rng(44, torch),
        sort=True,
    )
    n_edges = int(topo.pre_idx.numel())
    assert 0 <= n_edges <= 48
    _assert_index_bounds(topo, 8, 6, torch)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("builder", "kwargs"),
    [
        (
            build_sparse_random_topology,
            {"n_pre": 5, "n_post": 4, "p_connect": 0.4},
        ),
        (
            build_sparse_fanout_topology,
            {"n_pre": 5, "n_post": 4, "fanout": 2},
        ),
        (
            build_sparse_fanin_topology,
            {"n_pre": 5, "n_post": 4, "fanin": 2},
        ),
        (
            build_block_sparse_topology,
            {"n_pre": 6, "n_post": 5, "block_pre": 2, "block_post": 2, "p_block": 0.6},
        ),
    ],
)
def test_sparse_topology_is_deterministic(
    builder: Any,
    kwargs: dict[str, Any],
) -> None:
    torch = require_torch()
    common = {
        "init_cfg": {"init": "uniform", "low": -0.2, "high": 0.2},
        "delay_cfg": {"mode": "uniform_int", "max_delay": 3},
        "dev": torch.device("cpu"),
        "tdt": torch.float32,
        "torch": torch,
        "sort": True,
    }
    topo1, _w1 = builder(**kwargs, **common, rng=_rng(123, torch))
    topo2, _w2 = builder(**kwargs, **common, rng=_rng(123, torch))
    assert torch.equal(topo1.pre_idx, topo2.pre_idx)
    assert torch.equal(topo1.post_idx, topo2.post_idx)
    assert torch.equal(topo1.delays, topo2.delays)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("builder", "kwargs"),
    [
        (
            build_sparse_random_topology,
            {"n_pre": 4, "n_post": 3, "p_connect": 0.5},
        ),
        (
            build_sparse_fanout_topology,
            {"n_pre": 4, "n_post": 3, "fanout": 2},
        ),
        (
            build_sparse_fanin_topology,
            {"n_pre": 4, "n_post": 3, "fanin": 2},
        ),
        (
            build_block_sparse_topology,
            {"n_pre": 4, "n_post": 3, "block_pre": 2, "block_post": 2, "p_block": 0.8},
        ),
    ],
)
def test_sparse_weights_are_trainable_live_view(
    builder: Any,
    kwargs: dict[str, Any],
) -> None:
    torch = require_torch()
    topo, weights = builder(
        **kwargs,
        init_cfg={"init": "uniform", "low": -0.2, "high": 0.2},
        delay_cfg={"mode": "zeros"},
        dev=torch.device("cpu"),
        tdt=torch.float32,
        torch=torch,
        rng=_rng(999, torch),
        sort=True,
    )
    assert weights.requires_grad
    assert topo.weights.requires_grad
    assert topo.weights.data_ptr() == weights.data_ptr()
