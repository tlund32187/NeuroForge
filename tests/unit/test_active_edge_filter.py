"""Parity tests for active-edge filtering in static synapse models."""

from __future__ import annotations

from typing import Any, cast

import pytest
import torch

from neuroforge.contracts.synapses import SynapseInputs, SynapseTopology
from neuroforge.contracts.types import Compartment
from neuroforge.factories.hub import DEFAULT_HUB
from neuroforge.synapses.dales_static import DalesStaticSynapseModel
from neuroforge.synapses.static import StaticSynapseModel


def _build_dense_topology(
    n_pre: int,
    n_post: int,
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> SynapseTopology:
    pre_idx = torch.arange(n_pre, dtype=torch.int64, device=device).repeat_interleave(n_post)
    post_idx = torch.arange(n_post, dtype=torch.int64, device=device).repeat(n_pre)
    n_edges = n_pre * n_post
    # Deterministic non-trivial weights (includes negative values).
    weights = torch.linspace(-0.75, 0.75, steps=n_edges, dtype=dtype, device=device)
    delays = torch.zeros(n_edges, dtype=torch.int64, device=device)
    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        weights=weights,
        delays=delays,
        n_pre=n_pre,
        n_post=n_post,
    )


def _run_step(
    model: Any,
    topo: SynapseTopology,
    pre_spikes: torch.Tensor,
) -> torch.Tensor:
    post_spikes = torch.zeros(
        topo.n_post,
        dtype=torch.bool,
        device=pre_spikes.device,
    )
    out = model.step(
        {},
        topo,
        SynapseInputs(
            pre_spikes=pre_spikes,
            post_spikes=post_spikes,
        ),
        ctx=None,
    )
    return cast("torch.Tensor", out.post_current[Compartment.SOMA])


def _build_models(
    model_name: str,
    *,
    sign_pre: torch.Tensor | None = None,
    max_fraction: float = 0.2,
) -> tuple[Any, Any]:
    fast: Any
    base: Any
    if model_name == "static":
        fast = StaticSynapseModel(
            use_active_edge_filter=True,
            active_edge_max_fraction=max_fraction,
        )
        base = StaticSynapseModel(use_active_edge_filter=False)
        return fast, base
    if model_name == "dales":
        fast = DalesStaticSynapseModel(
            sign_pre=sign_pre,
            use_active_edge_filter=True,
            active_edge_max_fraction=max_fraction,
        )
        base = DalesStaticSynapseModel(sign_pre=sign_pre, use_active_edge_filter=False)
        return fast, base
    msg = f"Unsupported model_name: {model_name!r}"
    raise ValueError(msg)


@pytest.mark.unit
@pytest.mark.parametrize("model_name", ["static", "dales"])
def test_active_edge_filter_no_spikes_matches_baseline(model_name: str) -> None:
    dev = torch.device("cpu")
    topo = _build_dense_topology(6, 4, device=dev)
    sign_pre = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], device=dev)
    fast, base = _build_models(model_name, sign_pre=sign_pre, max_fraction=0.2)

    pre_spikes = torch.zeros(topo.n_pre, dtype=torch.bool, device=dev)
    got_fast = _run_step(fast, topo, pre_spikes)
    got_base = _run_step(base, topo, pre_spikes)
    assert torch.allclose(got_fast, got_base)
    assert torch.count_nonzero(got_fast).item() == 0


@pytest.mark.unit
def test_registry_accepts_active_edge_filter_kwargs() -> None:
    static_model = DEFAULT_HUB.synapses.create(
        "static",
        use_active_edge_filter=True,
        active_edge_max_fraction=0.15,
    )
    assert isinstance(static_model, StaticSynapseModel)

    sign_pre = torch.tensor([1.0, -1.0], device=torch.device("cpu"))
    dales_model = DEFAULT_HUB.synapses.create(
        "static_dales",
        sign_pre=sign_pre,
        use_active_edge_filter=True,
        active_edge_max_fraction=0.15,
    )
    assert isinstance(dales_model, DalesStaticSynapseModel)


@pytest.mark.unit
@pytest.mark.parametrize("model_name", ["static", "dales"])
def test_active_edge_filter_sparse_bool_matches_baseline(model_name: str) -> None:
    dev = torch.device("cpu")
    topo = _build_dense_topology(6, 4, device=dev)
    sign_pre = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], device=dev)
    # 1 active pre out of 6 -> active edges ratio = 4 / 24 = 0.166 <= 0.2.
    fast, base = _build_models(model_name, sign_pre=sign_pre, max_fraction=0.2)

    pre_spikes = torch.tensor([False, True, False, False, False, False], device=dev)
    got_fast = _run_step(fast, topo, pre_spikes)
    got_base = _run_step(base, topo, pre_spikes)
    assert torch.allclose(got_fast, got_base, atol=1e-7, rtol=0.0)


@pytest.mark.unit
@pytest.mark.parametrize("model_name", ["static", "dales"])
def test_active_edge_filter_dense_bool_matches_baseline(model_name: str) -> None:
    dev = torch.device("cpu")
    topo = _build_dense_topology(6, 4, device=dev)
    sign_pre = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0], device=dev)
    # All active -> active edges ratio = 1.0, forcing fallback path for fast model.
    fast, base = _build_models(model_name, sign_pre=sign_pre, max_fraction=0.2)

    pre_spikes = torch.ones(topo.n_pre, dtype=torch.bool, device=dev)
    got_fast = _run_step(fast, topo, pre_spikes)
    got_base = _run_step(base, topo, pre_spikes)
    assert torch.allclose(got_fast, got_base, atol=1e-7, rtol=0.0)


@pytest.mark.unit
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize("model_name", ["static", "dales"])
def test_active_edge_filter_cuda_matches_baseline(model_name: str) -> None:
    dev = torch.device("cuda")
    topo = _build_dense_topology(8, 5, device=dev)
    sign_pre = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], device=dev)
    fast, base = _build_models(model_name, sign_pre=sign_pre, max_fraction=0.25)

    pre_spikes = torch.tensor([False, True, False, False, False, False, False, False], device=dev)
    got_fast = _run_step(fast, topo, pre_spikes)
    got_base = _run_step(base, topo, pre_spikes)
    assert got_fast.device.type == "cuda"
    assert got_base.device.type == "cuda"
    assert torch.allclose(got_fast, got_base, atol=1e-6, rtol=0.0)
