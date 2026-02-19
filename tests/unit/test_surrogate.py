"""Tests for surrogate-gradient LIF and autograd compatibility."""

from __future__ import annotations

import pytest
import torch

from neuroforge.contracts.neurons import NeuronInputs, StepContext
from neuroforge.contracts.synapses import SynapseInputs, SynapseTopology
from neuroforge.contracts.types import Compartment
from neuroforge.core.surrogate import surrogate_spike
from neuroforge.engine.core_engine import CoreEngine, Population, Projection
from neuroforge.neurons.lif.surrogate import SurrogateLIFModel
from neuroforge.synapses.static import StaticSynapseModel

# ── surrogate_spike unit tests ──────────────────────────────────────


class TestSurrogateSpike:
    """Verify forward (hard step) and backward (smooth gradient)."""

    def test_forward_hard_step(self) -> None:
        x = torch.tensor([-1.0, 0.0, 0.5, 1.0], requires_grad=True)
        s = surrogate_spike(x)
        assert s.tolist() == [0.0, 1.0, 1.0, 1.0]

    def test_backward_nonzero(self) -> None:
        x = torch.tensor([0.1], requires_grad=True)
        s = surrogate_spike(x)
        s.backward()
        assert x.grad is not None
        assert x.grad.item() != 0.0

    def test_backward_at_zero(self) -> None:
        x = torch.tensor([0.0], requires_grad=True)
        s = surrogate_spike(x, beta=5.0)
        s.backward()
        assert x.grad is not None
        # At x=0 the fast-sigmoid derivative is beta/2.
        assert abs(x.grad.item() - 2.5) < 1e-5

    def test_negative_input_grad(self) -> None:
        x = torch.tensor([-0.5], requires_grad=True)
        s = surrogate_spike(x, beta=5.0)
        s.backward()
        assert x.grad is not None
        # Gradient should be positive but small.
        assert x.grad.item() > 0.0


# ── SurrogateLIFModel unit tests ───────────────────────────────────


class TestSurrogateLIFModel:
    """Basic integration of the surrogate LIF neuron model."""

    def test_float_spikes(self) -> None:
        model = SurrogateLIFModel()
        state = model.init_state(4, "cpu", "float32")
        # Drive all neurons well above threshold.
        inp = NeuronInputs(drive={Compartment.SOMA: torch.full((4,), 100.0)})
        ctx = StepContext(dt=1e-3, step=0, t=0.0)
        result = model.step(state, inp, ctx)
        assert result.spikes.dtype == torch.float32
        assert result.spikes.sum().item() == 4.0

    def test_no_spikes_below_threshold(self) -> None:
        model = SurrogateLIFModel()
        state = model.init_state(2, "cpu", "float32")
        # Tiny drive — voltage stays below threshold.
        inp = NeuronInputs(drive={Compartment.SOMA: torch.full((2,), 0.01)})
        ctx = StepContext(dt=1e-3, step=0, t=0.0)
        result = model.step(state, inp, ctx)
        assert result.spikes.sum().item() == 0.0

    def test_reset_uses_detached_spikes(self) -> None:
        """Voltage after spiking should not carry spike gradient."""
        model = SurrogateLIFModel()
        state = model.init_state(1, "cpu", "float32")
        # Force spike.
        inp = NeuronInputs(drive={Compartment.SOMA: torch.full((1,), 100.0)})
        ctx = StepContext(dt=1e-3, step=0, t=0.0)
        result = model.step(state, inp, ctx)
        # Voltage should be at v_reset and not require grad.
        v = result.voltage[Compartment.SOMA]
        assert v.item() == pytest.approx(0.0, abs=1e-6)


# ── StaticSynapseModel float-spike tests ───────────────────────────


class TestStaticSynapseFloat:
    """Verify synapse handles float spikes while preserving bool behavior."""

    def _make_topology(self, w: torch.Tensor) -> SynapseTopology:
        """Build a 2-pre → 1-post fully-connected topology."""
        pre_idx = torch.tensor([0, 1], dtype=torch.long)
        post_idx = torch.tensor([0, 0], dtype=torch.long)
        delays = torch.zeros(2, dtype=torch.long)
        return SynapseTopology(
            pre_idx=pre_idx, post_idx=post_idx, weights=w,
            delays=delays, n_pre=2, n_post=1,
        )

    def test_bool_spikes_unchanged(self) -> None:
        w = torch.tensor([0.5, 0.3])
        topo = self._make_topology(w)
        model = StaticSynapseModel()
        state = model.init_state(topo, "cpu", "float32")
        inp = SynapseInputs(
            pre_spikes=torch.tensor([True, False]),
            post_spikes=torch.tensor([False]),
        )
        ctx_stub: object = None
        result = model.step(state, topo, inp, ctx_stub)
        current = result.post_current[Compartment.SOMA]
        assert current.item() == pytest.approx(0.5, abs=1e-6)

    def test_float_spikes(self) -> None:
        w = torch.tensor([0.5, 0.3])
        topo = self._make_topology(w)
        model = StaticSynapseModel()
        state = model.init_state(topo, "cpu", "float32")
        inp = SynapseInputs(
            pre_spikes=torch.tensor([1.0, 0.0]),
            post_spikes=torch.tensor([0.0]),
        )
        ctx_stub: object = None
        result = model.step(state, topo, inp, ctx_stub)
        current = result.post_current[Compartment.SOMA]
        assert current.item() == pytest.approx(0.5, abs=1e-6)

    def test_float_spikes_partial(self) -> None:
        """A float spike of 0.5 should contribute half the weight."""
        w = torch.tensor([1.0, 1.0])
        topo = self._make_topology(w)
        model = StaticSynapseModel()
        state = model.init_state(topo, "cpu", "float32")
        inp = SynapseInputs(
            pre_spikes=torch.tensor([0.5, 0.0]),
            post_spikes=torch.tensor([0.0]),
        )
        ctx_stub: object = None
        result = model.step(state, topo, inp, ctx_stub)
        current = result.post_current[Compartment.SOMA]
        assert current.item() == pytest.approx(0.5, abs=1e-6)


# ── End-to-end gradient flow test ──────────────────────────────────


class TestSurrogateGradientFlow:
    """Build a tiny engine network, run forward, backward, check grad."""

    def test_weight_grad_nonzero(self) -> None:
        """A trainable synapse weight must receive nonzero gradient."""
        from neuroforge.contracts.simulation import SimulationConfig

        # 1-neuron input pop → 1-neuron output pop via one synapse.
        w = torch.tensor([0.8], requires_grad=True)
        topo = SynapseTopology(
            pre_idx=torch.tensor([0], dtype=torch.long),
            post_idx=torch.tensor([0], dtype=torch.long),
            weights=w,
            delays=torch.tensor([0], dtype=torch.long),
            n_pre=1,
            n_post=1,
        )

        engine = CoreEngine(SimulationConfig(dt=1e-3, seed=42, device="cpu", dtype="float32"))
        engine.add_population(Population("inp", SurrogateLIFModel(), 1))
        engine.add_population(Population("out", SurrogateLIFModel(), 1))
        engine.add_projection(Projection(
            "inp_out", StaticSynapseModel(), "inp", "out", topo,
        ))
        engine.build()

        # Drive the input population hard so it spikes each step.
        strong_drive = {Compartment.SOMA: torch.full((1,), 100.0)}
        steps = 10
        spike_sum = torch.tensor(0.0)
        for _ in range(steps):
            result = engine.step(external_drive={"inp": strong_drive})
            spike_sum = spike_sum + result.spikes["out"].sum()

        loss = (spike_sum - 5.0) ** 2
        loss.backward()  # type: ignore[no-untyped-call]

        assert w.grad is not None, "weight should have received a gradient"
        assert w.grad.abs().item() > 0.0, "gradient should be nonzero"


# ── Registry tests ─────────────────────────────────────────────────


class TestSurrogateRegistry:
    """Ensure surrogate LIF is discoverable via the factory hub."""

    def test_lif_surr_registered(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB

        model = DEFAULT_HUB.neurons.create("lif_surr")
        assert isinstance(model, SurrogateLIFModel)

    def test_lif_surrogate_alias(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB

        model = DEFAULT_HUB.neurons.create("lif_surrogate")
        assert isinstance(model, SurrogateLIFModel)
