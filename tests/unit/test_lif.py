"""Math-predictive tests for the LIF neuron model.

Every test computes the expected result analytically FIRST, then
verifies that the model produces the same result.  No "trust the code"
tests — only "trust the math" tests.
"""

from __future__ import annotations

import math

import pytest
import torch

from neuroforge.contracts.neurons import NeuronInputs, StepContext
from neuroforge.contracts.types import Compartment
from neuroforge.neurons.lif.model import LIFModel, LIFParams


def _ctx(dt: float, step: int) -> StepContext:
    """Create a StepContext with t computed from step and dt."""
    return StepContext(dt=dt, step=step, t=step * dt)


# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def default_params() -> LIFParams:
    return LIFParams()


@pytest.fixture
def lif(default_params: LIFParams) -> LIFModel:
    return LIFModel(default_params)


@pytest.fixture
def dt() -> float:
    return 1e-3  # 1 ms


# ── Pure-math helper tests ──────────────────────────────────────────


class TestLIFMathHelpers:
    """Verify the analytical helper methods on LIFModel itself."""

    def test_decay_factor(self, lif: LIFModel, dt: float) -> None:
        """alpha = exp(-dt / tau_mem)."""
        expected = math.exp(-dt / lif.params.tau_mem)
        assert lif.decay_factor(dt) == pytest.approx(expected)

    def test_drive_factor(self, lif: LIFModel, dt: float) -> None:
        """R = dt / tau_mem."""
        expected = dt / lif.params.tau_mem
        assert lif.drive_factor(dt) == pytest.approx(expected)

    def test_steady_state(self, lif: LIFModel, dt: float) -> None:
        """v_ss = v_rest + drive * R / (1 - alpha)."""
        drive = 1.0
        alpha = lif.decay_factor(dt)
        r = lif.drive_factor(dt)
        expected = lif.params.v_rest + drive * r / (1.0 - alpha)
        assert lif.steady_state_voltage(drive, dt) == pytest.approx(expected)

    def test_voltage_after_k_steps(self, lif: LIFModel, dt: float) -> None:
        """Iterative recurrence must match closed-form for k steps."""
        drive = 0.5
        v0 = 0.0
        alpha = lif.decay_factor(dt)
        r = lif.drive_factor(dt)
        v_rest = lif.params.v_rest

        # Iterate manually
        v = v0
        for _ in range(10):
            v = v_rest + (v - v_rest) * alpha + drive * r

        predicted = lif.voltage_after_k_steps(v0, drive, dt, 10)
        assert predicted == pytest.approx(v, rel=1e-10)

    def test_steps_to_fire(self, lif: LIFModel, dt: float) -> None:
        """Predict the exact step on which the neuron fires."""
        drive = 25.0  # strong drive → should fire quickly
        k = lif.steps_to_fire(drive, dt)
        assert k is not None

        # One step before should be below threshold
        v_before = lif.voltage_after_k_steps(0.0, drive, dt, k - 1)
        assert v_before < lif.params.v_thresh

        # At step k should be at or above threshold
        v_at = lif.voltage_after_k_steps(0.0, drive, dt, k)
        assert v_at >= lif.params.v_thresh

    def test_steps_to_fire_weak_drive_returns_none(
        self, lif: LIFModel, dt: float
    ) -> None:
        """If steady-state < threshold, steps_to_fire returns None."""
        # steady-state with drive=0.5: v_ss = 0.5 * 0.05/0.04877 ≈ 0.5125 < 1.0
        drive = 0.5
        assert lif.steps_to_fire(drive, dt) is None


# ── Tensor-level simulation tests ──────────────────────────────────


class TestLIFSimulation:
    """Run the actual step() method and compare to math predictions."""

    def test_single_step_decay(self, lif: LIFModel, dt: float) -> None:
        """With zero drive, voltage should decay: v1 = v_rest + (v0 - v_rest) * alpha."""
        state = lif.init_state(1, "cpu", "float64")
        v0 = 0.8
        state["v"] = torch.tensor([v0], dtype=torch.float64)

        inputs = NeuronInputs(
            drive={Compartment.SOMA: torch.tensor([0.0], dtype=torch.float64)}
        )
        ctx = _ctx(dt, 0)

        result = lif.step(state, inputs, ctx)

        alpha = lif.decay_factor(dt)
        expected_v = lif.params.v_rest + (v0 - lif.params.v_rest) * alpha
        assert state["v"].item() == pytest.approx(expected_v, rel=1e-10)
        assert result.spikes.any().item() is False

    def test_single_step_with_drive(self, lif: LIFModel, dt: float) -> None:
        """v1 = v_rest + (v0 - v_rest) * alpha + drive * R."""
        state = lif.init_state(1, "cpu", "float64")
        drive_val = 5.0
        state["v"] = torch.tensor([0.0], dtype=torch.float64)

        inputs = NeuronInputs(
            drive={Compartment.SOMA: torch.tensor([drive_val], dtype=torch.float64)}
        )
        ctx = _ctx(dt, 0)

        lif.step(state, inputs, ctx)

        alpha = lif.decay_factor(dt)
        r = lif.drive_factor(dt)
        expected = lif.params.v_rest + 0.0 * alpha + drive_val * r
        assert state["v"].item() == pytest.approx(expected, rel=1e-10)

    def test_multi_step_trajectory(self, lif: LIFModel, dt: float) -> None:
        """Run k steps and compare final voltage to closed-form prediction."""
        drive_val = 0.8  # low enough → no spike
        k = 50

        # Math prediction (no spike)
        predicted = lif.voltage_after_k_steps(0.0, drive_val, dt, k)
        assert predicted < lif.params.v_thresh, "Drive too high — would spike"

        state = lif.init_state(1, "cpu", "float64")
        drive_tensor = torch.tensor([drive_val], dtype=torch.float64)

        for step_i in range(k):
            inputs = NeuronInputs(drive={Compartment.SOMA: drive_tensor})
            ctx = _ctx(dt, step_i)
            lif.step(state, inputs, ctx)

        assert state["v"].item() == pytest.approx(predicted, rel=1e-9)

    def test_spike_at_predicted_step(self, lif: LIFModel, dt: float) -> None:
        """Neuron must spike on the exact step predicted by steps_to_fire."""
        drive_val = 25.0
        expected_k = lif.steps_to_fire(drive_val, dt)
        assert expected_k is not None

        state = lif.init_state(1, "cpu", "float64")
        drive_tensor = torch.tensor([drive_val], dtype=torch.float64)

        fired_at: int | None = None
        for step_i in range(expected_k + 5):
            inputs = NeuronInputs(drive={Compartment.SOMA: drive_tensor})
            ctx = _ctx(dt, step_i)
            result = lif.step(state, inputs, ctx)
            if result.spikes.any().item():
                fired_at = step_i + 1  # step_i is 0-based, k is 1-based
                break

        assert fired_at == expected_k

    def test_reset_after_spike(self, lif: LIFModel, dt: float) -> None:
        """After spiking, voltage should be v_reset."""
        state = lif.init_state(1, "cpu", "float64")
        # Put voltage just above threshold
        state["v"] = torch.tensor([lif.params.v_thresh + 0.1], dtype=torch.float64)

        # Any drive — voltage is already above threshold from last step's update
        # Actually we need to step so the model detects the spike
        drive_val = 100.0  # strong drive to guarantee spike
        inputs = NeuronInputs(
            drive={Compartment.SOMA: torch.tensor([drive_val], dtype=torch.float64)}
        )
        ctx = _ctx(dt, 0)
        result = lif.step(state, inputs, ctx)

        assert result.spikes.all().item() is True
        assert state["v"].item() == pytest.approx(lif.params.v_reset)

    def test_sub_threshold_no_spike(self, lif: LIFModel, dt: float) -> None:
        """With weak drive, neuron should never spike even after many steps."""
        drive_val = 0.3
        assert lif.steps_to_fire(drive_val, dt) is None

        state = lif.init_state(1, "cpu", "float64")
        drive_tensor = torch.tensor([drive_val], dtype=torch.float64)

        for step_i in range(1000):
            inputs = NeuronInputs(drive={Compartment.SOMA: drive_tensor})
            ctx = _ctx(dt, step_i)
            result = lif.step(state, inputs, ctx)
            assert result.spikes.any().item() is False

    def test_population_batch(self, lif: LIFModel, dt: float) -> None:
        """Multiple neurons with varying drives should spike independently."""
        n = 4
        state = lif.init_state(n, "cpu", "float64")
        # Drives: [0.3, 25.0, 50.0, 0.0]
        drives = torch.tensor([0.3, 25.0, 50.0, 0.0], dtype=torch.float64)

        # Predict which ones will eventually spike
        will_fire = [lif.steps_to_fire(d, dt) is not None for d in drives.tolist()]
        assert will_fire == [False, True, True, False]

        # Run enough steps for the fast ones to fire
        max_steps = 200
        ever_spiked = torch.zeros(n, dtype=torch.bool)
        for step_i in range(max_steps):
            inputs = NeuronInputs(drive={Compartment.SOMA: drives})
            ctx = _ctx(dt, step_i)
            result = lif.step(state, inputs, ctx)
            ever_spiked |= result.spikes

        assert ever_spiked[0].item() is False
        assert ever_spiked[1].item() is True
        assert ever_spiked[2].item() is True
        assert ever_spiked[3].item() is False

    def test_reset_state(self, lif: LIFModel) -> None:
        """reset_state should set voltage back to v_rest."""
        state = lif.init_state(5, "cpu", "float64")
        state["v"].fill_(0.9)

        lif.reset_state(state)

        expected = torch.full((5,), lif.params.v_rest, dtype=torch.float64)
        assert torch.allclose(state["v"], expected)

    def test_determinism(self, lif: LIFModel, dt: float) -> None:
        """Two runs with same initial state and inputs must produce identical results."""
        drive_tensor = torch.tensor([15.0], dtype=torch.float64)
        results = []

        for _ in range(2):
            state = lif.init_state(1, "cpu", "float64")
            voltages: list[float] = []
            for step_i in range(100):
                inputs = NeuronInputs(drive={Compartment.SOMA: drive_tensor.clone()})
                ctx = _ctx(dt, step_i)
                lif.step(state, inputs, ctx)
                voltages.append(state["v"].item())
            results.append(voltages)

        for v1, v2 in zip(results[0], results[1], strict=True):
            assert v1 == v2

    def test_zero_drive_decay_to_rest(self, lif: LIFModel, dt: float) -> None:
        """With zero drive and v0 > v_rest, voltage should decay toward v_rest."""
        state = lif.init_state(1, "cpu", "float64")
        state["v"] = torch.tensor([0.5], dtype=torch.float64)

        for step_i in range(500):
            inputs = NeuronInputs(
                drive={Compartment.SOMA: torch.tensor([0.0], dtype=torch.float64)}
            )
            ctx = _ctx(dt, step_i)
            lif.step(state, inputs, ctx)

        # After 500 ms of decay, should be very close to v_rest
        expected = lif.voltage_after_k_steps(0.5, 0.0, dt, 500)
        assert state["v"].item() == pytest.approx(expected, abs=1e-12)
        assert abs(state["v"].item() - lif.params.v_rest) < 1e-10


# ── Registry tests ──────────────────────────────────────────────────


class TestNeuronRegistry:
    """Verify the neuron registry has LIF registered."""

    def test_lif_registered(self) -> None:
        from neuroforge.neurons.registry import NEURON_MODELS

        assert NEURON_MODELS.has("lif")

    def test_create_lif(self) -> None:
        from neuroforge.neurons.registry import create_neuron_model

        model = create_neuron_model("lif")
        assert isinstance(model, LIFModel)

    def test_list_keys(self) -> None:
        from neuroforge.neurons.registry import NEURON_MODELS

        keys = NEURON_MODELS.list_keys()
        assert "lif" in keys

    def test_compartments(self) -> None:
        lif = LIFModel()
        assert lif.compartments() == (Compartment.SOMA,)
