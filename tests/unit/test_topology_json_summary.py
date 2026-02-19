"""Tests for topology JSON summary behavior on large projections."""

from __future__ import annotations

import pytest

from neuroforge.network.factory import NetworkFactory, to_topology_json
from neuroforge.network.specs import NetworkSpec, PopulationSpec, ProjectionSpec
from neuroforge.neurons.registry import NEURON_MODELS
from neuroforge.synapses.registry import SYNAPSE_MODELS


def _large_dense_spec() -> NetworkSpec:
    return NetworkSpec(
        populations=[
            PopulationSpec("input", 96, "lif"),
            PopulationSpec("hidden", 96, "lif"),
            PopulationSpec("output", 1, "lif"),
        ],
        projections=[
            ProjectionSpec(
                "input_hidden",
                "input",
                "hidden",
                "static",
                topology={
                    "type": "dense",
                    "init": "uniform",
                    "low": -0.2,
                    "high": 0.2,
                },
            ),
        ],
    )


@pytest.mark.unit
def test_to_topology_json_uses_summary_for_large_projection() -> None:
    factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
    engine = factory.build(_large_dense_spec(), device="cpu", dtype="float32", seed=7)
    topo = to_topology_json(engine)

    assert "edges" in topo
    assert len(topo["edges"]) == 1

    edge = topo["edges"][0]
    assert edge["name"] == "input_hidden"
    assert edge["n_edges"] == 96 * 96
    assert "weight_stats" in edge
    assert "weights" not in edge
    assert "weights_sample" in edge
    assert "sample_size" in edge
    assert edge["sample_size"] <= 256
    assert len(edge["weights_sample"]) == edge["sample_size"]

    assert "projection_meta" in topo
    assert len(topo["projection_meta"]) == 1
    meta = topo["projection_meta"][0]
    assert meta["name"] == "input_hidden"
    assert meta["n_edges"] == 96 * 96
