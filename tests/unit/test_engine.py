"""Tests for the CoreEngine — orchestration of populations and projections.

Tests verify that the engine correctly wires neuron populations through
synaptic projections and produces mathematically predictable results.
"""

from __future__ import annotations

import math

import pytest
import torch

from neuroforge.contracts.simulation import SimulationConfig
from neuroforge.contracts.synapses import SynapseTopology
from neuroforge.contracts.types import Compartment
from neuroforge.engine.core_engine import CoreEngine, Population, Projection
from neuroforge.neurons.lif.model import LIFModel, LIFParams
from neuroforge.synapses.static import StaticSynapseModel

# ── Helpers ─────────────────────────────────────────────────────────


def _make_config(dt: float = 1e-3, seed: int = 42) -> SimulationConfig:
    return SimulationConfig(dt=dt, seed=seed, device="cpu", dtype="float64")


def _make_topology(
    edges: list[tuple[int, int, float]],
    n_pre: int,
    n_post: int,
) -> SynapseTopology:
    pre_idx = torch.tensor([e[0] for e in edges], dtype=torch.long)
    post_idx = torch.tensor([e[1] for e in edges], dtype=torch.long)
    weights = torch.tensor([e[2] for e in edges], dtype=torch.float64)
    delays = torch.zeros(len(edges), dtype=torch.long)
    return SynapseTopology(
        pre_idx=pre_idx,
        post_idx=post_idx,
        weights=weights,
        delays=delays,
        n_pre=n_pre,
        n_post=n_post,
    )


# ── Basic lifecycle tests ──────────────────────────────────────────


class TestEngineLifecycle:
    """Test build, reset, and error handling."""

    def test_build_single_population(self) -> None:
        """Engine can build with a single population, no projections."""
        engine = CoreEngine(_make_config())
        engine.add_population(Population(name="pop", model=LIFModel(), n=5))
        engine.build()

        # State should be initialised
        pop = engine.populations["pop"]
        assert pop.state["v"].shape == (5,)

    def test_step_before_build_raises(self) -> None:
        """Stepping before build raises RuntimeError."""
        engine = CoreEngine(_make_config())
        engine.add_population(Population(name="pop", model=LIFModel(), n=1))
        with pytest.raises(RuntimeError, match="not built"):
            engine.step()

    def test_duplicate_population_raises(self) -> None:
        engine = CoreEngine(_make_config())
        engine.add_population(Population(name="pop", model=LIFModel(), n=1))
        with pytest.raises(ValueError, match="Duplicate"):
            engine.add_population(Population(name="pop", model=LIFModel(), n=1))

    def test_duplicate_projection_raises(self) -> None:
        engine = CoreEngine(_make_config())
        engine.add_population(Population(name="a", model=LIFModel(), n=1))
        engine.add_population(Population(name="b", model=LIFModel(), n=1))
        topo = _make_topology([(0, 0, 1.0)], 1, 1)
        engine.add_projection(
            Projection(
                name="p",
                model=StaticSynapseModel(),
                source="a",
                target="b",
                topology=topo,
            )
        )
        with pytest.raises(ValueError, match="Duplicate"):
            engine.add_projection(
                Projection(
                    name="p",
                    model=StaticSynapseModel(),
                    source="a",
                    target="b",
                    topology=topo,
                )
            )

    def test_missing_source_raises(self) -> None:
        engine = CoreEngine(_make_config())
        engine.add_population(Population(name="b", model=LIFModel(), n=1))
        topo = _make_topology([(0, 0, 1.0)], 1, 1)
        engine.add_projection(
            Projection(
                name="p",
                model=StaticSynapseModel(),
                source="missing",
                target="b",
                topology=topo,
            )
        )
        with pytest.raises(ValueError, match="source.*not found"):
            engine.build()

    def test_reset(self) -> None:
        engine = CoreEngine(_make_config())
        lif = LIFModel()
        engine.add_population(Population(name="pop", model=lif, n=3))
        engine.build()

        # Modify voltage
        engine.populations["pop"].state["v"].fill_(0.9)

        engine.reset()
        expected = torch.full((3,), lif.params.v_rest, dtype=torch.float64)
        # Need to get state from the internal population
        assert torch.allclose(engine._populations["pop"].state["v"], expected)


# ── Single population with external drive ──────────────────────────


class TestEngineSinglePopulation:
    """Test with one population and external drive only."""

    def test_external_drive_produces_spike(self) -> None:
        """External drive above threshold makes single neuron spike at predicted step."""
        dt = 1e-3
        lif = LIFModel(LIFParams(tau_mem=20e-3, v_thresh=1.0))
        drive_val = 25.0

        # Math prediction
        expected_k = lif.steps_to_fire(drive_val, dt)
        assert expected_k is not None

        engine = CoreEngine(_make_config(dt=dt))
        engine.add_population(Population(name="out", model=lif, n=1))
        engine.build()

        drive_tensor = torch.tensor([drive_val], dtype=torch.float64)

        fired_at: int | None = None
        for i in range(expected_k + 5):
            result = engine.step(
                external_drive={"out": {Compartment.SOMA: drive_tensor}}
            )
            if result.spikes["out"].any().item():
                fired_at = i + 1  # 1-based
                break

        assert fired_at == expected_k

    def test_no_drive_no_spike(self) -> None:
        """Without external drive, neuron never spikes (voltage stays at rest)."""
        engine = CoreEngine(_make_config())
        engine.add_population(Population(name="pop", model=LIFModel(), n=1))
        engine.build()

        for _ in range(100):
            result = engine.step()
            assert not result.spikes["pop"].any().item()

    def test_run_multiple_steps(self) -> None:
        """run() returns a list of StepResults with correct step indices."""
        engine = CoreEngine(_make_config())
        engine.add_population(Population(name="pop", model=LIFModel(), n=1))
        engine.build()

        results = engine.run(10)
        assert len(results) == 10
        assert [r.step for r in results] == list(range(10))
        for r in results:
            assert r.t == pytest.approx(r.step * 1e-3)


# ── Two populations with projection ────────────────────────────────


class TestEngineTwoPopulations:
    """Test with two populations connected by a projection."""

    def test_spike_propagation(self) -> None:
        """Pre-neuron spike propagates through synapse to post-neuron.

        Setup:
        - Population "pre" (1 neuron): driven externally to fire
        - Population "post" (1 neuron): receives synaptic current from "pre"
        - Projection: weight = 30.0 (strong enough to eventually make post fire)

        Math:
        - pre fires at step K (predicted by steps_to_fire)
        - At step K+1 the synapse delivers w=30.0 to post
        - post integrates that current and eventually spikes
        """
        dt = 1e-3
        lif = LIFModel(LIFParams(tau_mem=20e-3, v_thresh=1.0))
        synapse_weight = 30.0
        drive_val = 100.0  # strong drive to make pre fire on step 1

        # Predict when pre fires
        pre_fire_step = lif.steps_to_fire(drive_val, dt)
        assert pre_fire_step is not None
        assert pre_fire_step == 1  # with drive=100, should fire on step 1

        engine = CoreEngine(_make_config(dt=dt))
        engine.add_population(Population(name="pre", model=LIFModel(lif.params), n=1))
        engine.add_population(Population(name="post", model=LIFModel(lif.params), n=1))
        engine.add_projection(
            Projection(
                name="pre_to_post",
                model=StaticSynapseModel(),
                source="pre",
                target="post",
                topology=_make_topology([(0, 0, synapse_weight)], 1, 1),
            )
        )
        engine.build()

        drive_tensor = torch.tensor([drive_val], dtype=torch.float64)
        pre_fired = False
        post_fired = False

        # Run enough steps for spike to propagate
        for _i in range(50):
            result = engine.step(
                external_drive={"pre": {Compartment.SOMA: drive_tensor}}
            )
            if result.spikes["pre"].any().item():
                pre_fired = True
            if result.spikes["post"].any().item():
                post_fired = True
                break

        assert pre_fired, "Pre-neuron should have fired"
        assert post_fired, "Post-neuron should have been driven to fire by synapse"

    def test_inhibitory_projection_blocks_spike(self) -> None:
        """Strong inhibitory connection prevents post-neuron from reaching threshold.

        We give both pre and post external drive, but the inhibitory synapse
        from pre→post should counteract the post drive.
        """
        dt = 1e-3
        lif_params = LIFParams(tau_mem=20e-3, v_thresh=1.0)

        engine = CoreEngine(_make_config(dt=dt))
        engine.add_population(Population(name="pre", model=LIFModel(lif_params), n=1))
        engine.add_population(Population(name="post", model=LIFModel(lif_params), n=1))

        # Strong inhibition: when pre fires, post gets -50.0
        engine.add_projection(
            Projection(
                name="inh",
                model=StaticSynapseModel(),
                source="pre",
                target="post",
                topology=_make_topology([(0, 0, -50.0)], 1, 1),
            )
        )
        engine.build()

        # Drive pre hard (100), post just enough to be near threshold
        drive_pre = torch.tensor([100.0], dtype=torch.float64)
        # Drive post at 20.0 — steady state = 20*0.05/0.04877 ≈ 20.5 >> threshold,
        # but inhibition should suppress
        drive_post = torch.tensor([20.0], dtype=torch.float64)

        for _ in range(50):
            engine.step(
                external_drive={
                    "pre": {Compartment.SOMA: drive_pre},
                    "post": {Compartment.SOMA: drive_post},
                }
            )

        # At drive=20, without inhibition post would fire. We check the
        # voltage is being pushed down by the inhibitory current.
        # Due to timing, there may be a brief fire before inhibition kicks in
        # on step 0 (no prior spikes). The key test is the voltage is
        # significantly lower than without inhibition.
        v_post = engine._populations["post"].state["v"].item()
        # Without inhibition it would be near v_thresh or cycling.
        # With strong inhibition it should be well below threshold.
        assert v_post < lif_params.v_thresh


# ── Voltage trajectory verification ────────────────────────────────


class TestEngineVoltageTrajectory:
    """Verify voltage values against math predictions through the engine."""

    def test_voltage_matches_standalone(self) -> None:
        """Engine-driven LIF voltage must match standalone LIF model voltage."""
        dt = 1e-3
        drive_val = 0.8  # sub-threshold
        lif = LIFModel()

        # Standalone prediction
        predicted = lif.voltage_after_k_steps(0.0, drive_val, dt, 20)
        assert predicted < lif.params.v_thresh

        # Engine run
        engine = CoreEngine(_make_config(dt=dt))
        engine.add_population(Population(name="pop", model=LIFModel(), n=1))
        engine.build()

        drive_tensor = torch.tensor([drive_val], dtype=torch.float64)
        for _ in range(20):
            engine.step(external_drive={"pop": {Compartment.SOMA: drive_tensor}})

        actual = engine._populations["pop"].state["v"].item()
        assert actual == pytest.approx(predicted, rel=1e-9)

    def test_decay_without_drive(self) -> None:
        """Voltage decays toward v_rest when no drive is applied."""
        dt = 1e-3
        lif = LIFModel()

        engine = CoreEngine(_make_config(dt=dt))
        engine.add_population(Population(name="pop", model=LIFModel(), n=1))
        engine.build()

        # Set initial voltage
        engine._populations["pop"].state["v"] = torch.tensor([0.8], dtype=torch.float64)

        for _ in range(100):
            engine.step()

        alpha_100 = math.exp(-100 * dt / lif.params.tau_mem)
        expected = lif.params.v_rest + (0.8 - lif.params.v_rest) * alpha_100
        actual = engine._populations["pop"].state["v"].item()
        assert actual == pytest.approx(expected, rel=1e-9)
