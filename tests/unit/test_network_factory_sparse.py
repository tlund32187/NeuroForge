"""Sparse-path tests for NetworkFactory."""

from __future__ import annotations

import math
from typing import Any

import pytest

from neuroforge.network.factory import NetworkFactory
from neuroforge.network.specs import NetworkSpec, PopulationSpec, ProjectionSpec
from neuroforge.neurons.registry import NEURON_MODELS
from neuroforge.synapses.registry import SYNAPSE_MODELS


def _sparse_spec(topology: dict[str, Any]) -> NetworkSpec:
    return NetworkSpec(
        populations=[
            PopulationSpec("input", 40, "lif"),
            PopulationSpec("hidden", 30, "lif"),
            PopulationSpec("output", 2, "lif"),
        ],
        projections=[
            ProjectionSpec(
                "input_hidden",
                "input",
                "hidden",
                "static",
                topology=topology,
            ),
        ],
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("topology", "expected_edges"),
    [
        (
            {
                "type": "sparse_fanout",
                "fanout": 7,
                "init": "uniform",
                "low": -0.2,
                "high": 0.2,
                "delays": {"mode": "zeros"},
            },
            40 * 7,
        ),
        (
            {
                "type": "sparse_fanin",
                "fanin": 5,
                "init": "uniform",
                "low": -0.2,
                "high": 0.2,
                "delays": {"mode": "zeros"},
            },
            30 * 5,
        ),
        (
            {
                "type": "block_sparse",
                "block_pre": 8,
                "block_post": 6,
                "p_block": 1.0,
                "init": "uniform",
                "low": -0.2,
                "high": 0.2,
                "delays": {"mode": "uniform_int", "max_delay": 2},
            },
            40 * 30,
        ),
    ],
)
def test_sparse_projection_trainable_weight_vector_exact_edge_count(
    topology: dict[str, Any],
    expected_edges: int,
) -> None:
    factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
    engine = factory.build(_sparse_spec(topology), device="cpu", dtype="float32", seed=123)
    proj = engine.projections["input_hidden"]

    assert "weights" in proj.state
    assert "weight_matrix" not in proj.state
    assert proj.state["weights"].requires_grad
    assert proj.topology.weights.requires_grad
    assert proj.topology.weights.ndim == 1
    assert proj.topology.weights.data_ptr() == proj.state["weights"].data_ptr()
    assert int(proj.topology.weights.numel()) == expected_edges


@pytest.mark.unit
def test_sparse_random_projection_edge_count_is_near_expected() -> None:
    p_connect = 0.25
    topology = {
        "type": "sparse_random",
        "p_connect": p_connect,
        "init": "uniform",
        "low": -0.2,
        "high": 0.2,
        "delays": {"mode": "uniform_int", "max_delay": 2},
    }
    n_pre = 40
    n_post = 30
    n_total = n_pre * n_post

    factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
    engine = factory.build(_sparse_spec(topology), device="cpu", dtype="float32", seed=123)
    proj = engine.projections["input_hidden"]

    assert "weights" in proj.state
    assert "weight_matrix" not in proj.state
    assert proj.topology.weights.requires_grad
    assert proj.topology.weights.ndim == 1
    assert proj.topology.weights.data_ptr() == proj.state["weights"].data_ptr()

    observed = int(proj.topology.weights.numel())
    expected = n_total * p_connect
    sigma = math.sqrt(n_total * p_connect * (1.0 - p_connect))
    tolerance = max(20.0, 5.0 * sigma)
    assert abs(observed - expected) <= tolerance
