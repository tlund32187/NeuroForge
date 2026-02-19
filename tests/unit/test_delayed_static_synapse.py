"""Tests for delayed static synapse propagation."""

from __future__ import annotations

import pytest
import torch

from neuroforge.contracts.synapses import SynapseInputs, SynapseTopology
from neuroforge.contracts.types import Compartment
from neuroforge.synapses.delayed_static import DelayedStaticSynapseModel
from neuroforge.synapses.registry import SYNAPSE_MODELS


def _build_topology(device: torch.device) -> SynapseTopology:
    # Edges:
    # pre0 -> post0  w=+1.0 d=0
    # pre1 -> post0  w=+2.0 d=1
    # pre2 -> post1  w=-1.5 d=2
    # pre0 -> post1  w=+0.5 d=2
    return SynapseTopology(
        pre_idx=torch.tensor([0, 1, 2, 0], dtype=torch.int64, device=device),
        post_idx=torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=device),
        weights=torch.tensor([1.0, 2.0, -1.5, 0.5], dtype=torch.float32, device=device),
        delays=torch.tensor([0, 1, 2, 2], dtype=torch.int64, device=device),
        n_pre=3,
        n_post=2,
    )


def _run_bool_pattern(device: torch.device) -> list[torch.Tensor]:
    topo = _build_topology(device)
    model = DelayedStaticSynapseModel()
    state = model.init_state(topo, device.type, "float32")
    post_zeros = torch.zeros(topo.n_post, dtype=torch.bool, device=device)

    pattern = [
        torch.tensor([True, False, False], dtype=torch.bool, device=device),
        torch.tensor([False, True, False], dtype=torch.bool, device=device),
        torch.tensor([False, False, True], dtype=torch.bool, device=device),
        torch.tensor([True, False, False], dtype=torch.bool, device=device),
        torch.tensor([False, False, False], dtype=torch.bool, device=device),
        torch.tensor([False, False, False], dtype=torch.bool, device=device),
    ]

    outs: list[torch.Tensor] = []
    with torch.no_grad():
        for pre_spikes in pattern:
            result = model.step(
                state,
                topo,
                SynapseInputs(pre_spikes=pre_spikes, post_spikes=post_zeros),
                ctx=None,
            )
            outs.append(result.post_current[Compartment.SOMA].detach().cpu())
    return outs


@pytest.mark.unit
def test_delayed_static_registered() -> None:
    assert SYNAPSE_MODELS.has("static_delayed")


@pytest.mark.unit
def test_delayed_static_bool_delays_cpu() -> None:
    outs = _run_bool_pattern(torch.device("cpu"))
    expected = [
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0], dtype=torch.float32),
        torch.tensor([2.0, 0.5], dtype=torch.float32),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, -1.5], dtype=torch.float32),
        torch.tensor([0.0, 0.5], dtype=torch.float32),
    ]
    for got, exp in zip(outs, expected, strict=True):
        assert torch.allclose(got, exp, atol=1e-6, rtol=0.0)


@pytest.mark.unit
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_delayed_static_bool_delays_cuda() -> None:
    outs = _run_bool_pattern(torch.device("cuda"))
    expected = [
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, 0.0], dtype=torch.float32),
        torch.tensor([2.0, 0.5], dtype=torch.float32),
        torch.tensor([1.0, 0.0], dtype=torch.float32),
        torch.tensor([0.0, -1.5], dtype=torch.float32),
        torch.tensor([0.0, 0.5], dtype=torch.float32),
    ]
    for got, exp in zip(outs, expected, strict=True):
        assert torch.allclose(got, exp, atol=1e-6, rtol=0.0)


@pytest.mark.unit
def test_delayed_static_float_spikes() -> None:
    device = torch.device("cpu")
    topo = SynapseTopology(
        pre_idx=torch.tensor([0], dtype=torch.int64, device=device),
        post_idx=torch.tensor([0], dtype=torch.int64, device=device),
        weights=torch.tensor([2.0], dtype=torch.float32, device=device),
        delays=torch.tensor([2], dtype=torch.int64, device=device),
        n_pre=1,
        n_post=1,
    )
    model = DelayedStaticSynapseModel()
    state = model.init_state(topo, "cpu", "float32")

    pattern = [
        torch.tensor([0.5], dtype=torch.float32, device=device),
        torch.tensor([0.0], dtype=torch.float32, device=device),
        torch.tensor([0.0], dtype=torch.float32, device=device),
        torch.tensor([1.0], dtype=torch.float32, device=device),
        torch.tensor([0.0], dtype=torch.float32, device=device),
        torch.tensor([0.0], dtype=torch.float32, device=device),
    ]
    expected = [0.0, 0.0, 1.0, 0.0, 0.0, 2.0]

    with torch.no_grad():
        for pre_spikes, exp in zip(pattern, expected, strict=True):
            result = model.step(
                state,
                topo,
                SynapseInputs(
                    pre_spikes=pre_spikes,
                    post_spikes=torch.zeros(1, dtype=torch.float32, device=device),
                ),
                ctx=None,
            )
            got = float(result.post_current[Compartment.SOMA][0].item())
            assert got == pytest.approx(exp, abs=1e-6)


@pytest.mark.unit
def test_delayed_static_raises_when_grad_enabled() -> None:
    device = torch.device("cpu")
    topo = _build_topology(device)
    model = DelayedStaticSynapseModel()
    state = model.init_state(topo, "cpu", "float32")
    inputs = SynapseInputs(
        pre_spikes=torch.tensor([True, False, False], dtype=torch.bool, device=device),
        post_spikes=torch.zeros(topo.n_post, dtype=torch.bool, device=device),
    )
    assert torch.is_grad_enabled()
    with pytest.raises(NotImplementedError, match="Phase 7"):
        _ = model.step(state, topo, inputs, ctx=None)
