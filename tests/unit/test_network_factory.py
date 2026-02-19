"""Unit tests for NetworkFactory + DTOs."""

from __future__ import annotations

import pytest

from neuroforge.network.factory import NetworkFactory, to_topology_json
from neuroforge.network.specs import NetworkSpec, PopulationSpec, ProjectionSpec
from neuroforge.neurons.registry import NEURON_MODELS
from neuroforge.synapses.registry import SYNAPSE_MODELS


def _small_spec() -> NetworkSpec:
    """Input(2) → Hidden(4) → Output(1)."""
    return NetworkSpec(
        populations=[
            PopulationSpec("input", 2, "lif"),
            PopulationSpec("hidden", 4, "lif"),
            PopulationSpec("output", 1, "lif"),
        ],
        projections=[
            ProjectionSpec(
                "input_hidden", "input", "hidden", "static",
                topology={
                    "type": "dense", "init": "uniform",
                    "low": -0.5, "high": 0.5, "bias": True,
                },
            ),
            ProjectionSpec(
                "hidden_output", "hidden", "output", "static",
                topology={
                    "type": "dense", "init": "uniform",
                    "low": -0.5, "high": 0.5, "bias": True,
                },
            ),
        ],
    )


# ── Build on CPU ────────────────────────────────────────────────────


@pytest.mark.unit
class TestNetworkFactoryCPU:
    """Build a tiny network on CPU and verify structure."""

    def test_populations_exist(self) -> None:
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        engine = factory.build(_small_spec(), device="cpu", dtype="float32", seed=42)
        pops = engine.populations

        assert set(pops.keys()) == {"input", "hidden", "output"}
        assert pops["input"].n == 2
        assert pops["hidden"].n == 4
        assert pops["output"].n == 1

    def test_population_state_device_and_shape(self) -> None:
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        engine = factory.build(_small_spec(), device="cpu", dtype="float32", seed=42)

        for pop in engine.populations.values():
            v = pop.state["v"]
            assert v.device.type == "cpu"
            assert v.shape == (pop.n,)

    def test_projections_exist(self) -> None:
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        engine = factory.build(_small_spec(), device="cpu", dtype="float32", seed=42)
        projs = engine.projections

        assert set(projs.keys()) == {"input_hidden", "hidden_output"}

    def test_weight_matrix_shapes(self) -> None:
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        engine = factory.build(_small_spec(), device="cpu", dtype="float32", seed=42)
        projs = engine.projections

        ih = projs["input_hidden"]
        assert ih.state["weight_matrix"].shape == (2, 4)
        assert ih.state["bias"].shape == (4,)

        ho = projs["hidden_output"]
        # n_post == 1 → weight stored as 1-D [n_pre]
        assert ho.state["weight_matrix"].shape == (4,)
        assert ho.state["bias"].shape == (1,)

    def test_weight_matrix_device(self) -> None:
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        engine = factory.build(_small_spec(), device="cpu", dtype="float32", seed=42)

        for proj in engine.projections.values():
            wm = proj.state["weight_matrix"]
            assert wm.device.type == "cpu"

    def test_engine_is_built(self) -> None:
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        engine = factory.build(_small_spec(), device="cpu", dtype="float32", seed=42)
        # Engine should be marked as built (can run step)
        assert engine._built  # noqa: SLF001


# ── Determinism ─────────────────────────────────────────────────────


@pytest.mark.unit
class TestNetworkFactoryDeterminism:
    """Same seed → identical weights."""

    def test_same_seed_same_weights(self) -> None:
        import torch

        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        e1 = factory.build(_small_spec(), device="cpu", dtype="float32", seed=42)
        e2 = factory.build(_small_spec(), device="cpu", dtype="float32", seed=42)

        w1 = e1.projections["input_hidden"].state["weight_matrix"]
        w2 = e2.projections["input_hidden"].state["weight_matrix"]
        assert torch.equal(w1, w2)

    def test_different_seed_different_weights(self) -> None:
        import torch

        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        e1 = factory.build(_small_spec(), device="cpu", dtype="float32", seed=42)
        e2 = factory.build(_small_spec(), device="cpu", dtype="float32", seed=99)

        w1 = e1.projections["input_hidden"].state["weight_matrix"]
        w2 = e2.projections["input_hidden"].state["weight_matrix"]
        assert not torch.equal(w1, w2)


# ── Topology JSON ───────────────────────────────────────────────────


@pytest.mark.unit
class TestTopologyJson:
    """to_topology_json produces the expected structure."""

    def test_layers_and_edges(self) -> None:
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        engine = factory.build(_small_spec(), device="cpu", dtype="float32", seed=42)
        topo = to_topology_json(engine)

        assert "layers" in topo
        assert "edges" in topo
        assert "projection_meta" in topo
        assert len(topo["layers"]) == 3
        assert len(topo["edges"]) == 2
        assert len(topo["projection_meta"]) == 2

    def test_edge_structure(self) -> None:
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        engine = factory.build(_small_spec(), device="cpu", dtype="float32", seed=42)
        topo = to_topology_json(engine)

        src_dst = [(e["src"], e["dst"]) for e in topo["edges"]]
        assert ("input", "hidden") in src_dst
        assert ("hidden", "output") in src_dst

        for edge in topo["edges"]:
            assert "weights" in edge

        for proj_meta in topo["projection_meta"]:
            assert isinstance(proj_meta["n_pre"], int)
            assert isinstance(proj_meta["n_post"], int)
            assert isinstance(proj_meta["n_edges"], int)


# ── Error handling ──────────────────────────────────────────────────


@pytest.mark.unit
class TestNetworkFactoryErrors:
    """Validation catches bad specs early."""

    def test_unknown_neuron_model(self) -> None:
        spec = NetworkSpec(
            populations=[PopulationSpec("x", 2, "does_not_exist")],
            projections=[],
        )
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        with pytest.raises(KeyError, match="does_not_exist"):
            factory.build(spec, device="cpu", dtype="float32")

    def test_unknown_projection_source(self) -> None:
        spec = NetworkSpec(
            populations=[PopulationSpec("a", 2, "lif")],
            projections=[
                ProjectionSpec("bad", "missing", "a", "static"),
            ],
        )
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        with pytest.raises(ValueError, match="missing"):
            factory.build(spec, device="cpu", dtype="float32")

    def test_unsupported_topology_type(self) -> None:
        spec = NetworkSpec(
            populations=[
                PopulationSpec("a", 2, "lif"),
                PopulationSpec("b", 2, "lif"),
            ],
            projections=[
                ProjectionSpec(
                    "ab", "a", "b", "static",
                    topology={"type": "star"},
                ),
            ],
        )
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        with pytest.raises(ValueError, match="star"):
            factory.build(spec, device="cpu", dtype="float32")


# ── Optional CUDA test ──────────────────────────────────────────────


def _has_cuda() -> bool:
    try:
        import torch

        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.unit
@pytest.mark.cuda
@pytest.mark.skipif(not _has_cuda(), reason="CUDA not available")
class TestNetworkFactoryCUDA:
    """Build on CUDA and verify tensor placement."""

    def test_tensors_on_cuda(self) -> None:
        factory = NetworkFactory(NEURON_MODELS, SYNAPSE_MODELS)
        engine = factory.build(_small_spec(), device="cuda", dtype="float32", seed=42)

        for pop in engine.populations.values():
            assert pop.state["v"].device.type == "cuda"

        for proj in engine.projections.values():
            assert proj.state["weight_matrix"].device.type == "cuda"
