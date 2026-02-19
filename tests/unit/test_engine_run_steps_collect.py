"""Tests for CoreEngine.run_steps collection modes."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from neuroforge.contracts.simulation import SimulationConfig
from neuroforge.contracts.synapses import SynapseTopology
from neuroforge.contracts.types import Compartment
from neuroforge.engine.core_engine import CoreEngine, Population, Projection
from neuroforge.neurons.lif.model import LIFModel
from neuroforge.synapses.static import StaticSynapseModel


def _cfg() -> SimulationConfig:
    return SimulationConfig(dt=1e-3, seed=123, device="cpu", dtype="float64")


def _single_edge_topology() -> SynapseTopology:
    return SynapseTopology(
        pre_idx=torch.tensor([0], dtype=torch.long),
        post_idx=torch.tensor([0], dtype=torch.long),
        weights=torch.tensor([15.0], dtype=torch.float64),
        delays=torch.zeros(1, dtype=torch.long),
        n_pre=1,
        n_post=1,
    )


def _build_engine() -> CoreEngine:
    engine = CoreEngine(_cfg())
    engine.add_population(Population(name="pre", model=LIFModel(), n=1))
    engine.add_population(Population(name="post", model=LIFModel(), n=1))
    engine.add_projection(
        Projection(
            name="pre_to_post",
            model=StaticSynapseModel(),
            source="pre",
            target="post",
            topology=_single_edge_topology(),
        )
    )
    engine.build()
    return engine


def _drive_fn(_step: int) -> dict[str, dict[Compartment, Any]]:
    return {
        "pre": {Compartment.SOMA: torch.tensor([100.0], dtype=torch.float64)},
    }


@pytest.mark.unit
def test_run_steps_collect_false_returns_none() -> None:
    engine = _build_engine()
    ret = engine.run_steps(8, _drive_fn, collect=False)
    assert ret is None

    # Next step should continue from step index 8.
    next_result = engine.step(external_drive=_drive_fn(8))
    assert next_result.step == 8


@pytest.mark.unit
def test_run_steps_collect_true_matches_run() -> None:
    engine_run = _build_engine()
    engine_steps = _build_engine()

    run_results = engine_run.run(12, _drive_fn)
    step_results = engine_steps.run_steps(12, _drive_fn, collect=True)
    assert step_results is not None
    assert len(run_results) == len(step_results)

    for a, b in zip(run_results, step_results, strict=True):
        assert a.step == b.step
        assert a.t == pytest.approx(b.t)
        assert set(a.spikes.keys()) == set(b.spikes.keys())
        for pop_name in a.spikes:
            assert torch.equal(a.spikes[pop_name], b.spikes[pop_name])
