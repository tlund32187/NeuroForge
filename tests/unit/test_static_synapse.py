"""Math-predictive tests for the static synapse model.

Each test computes the expected post-synaptic current analytically,
then verifies the model produces the same result.
"""

from __future__ import annotations

import pytest
import torch

from neuroforge.contracts.synapses import (
    SynapseInputs,
    SynapseStepResult,
    SynapseTopology,
)
from neuroforge.contracts.types import Compartment
from neuroforge.synapses.static import StaticSynapseModel

# ── Fixtures ────────────────────────────────────────────────────────


@pytest.fixture
def synapse() -> StaticSynapseModel:
    return StaticSynapseModel()


def _make_topology(
    edges: list[tuple[int, int, float]],
    n_pre: int,
    n_post: int,
) -> SynapseTopology:
    """Helper to build a SynapseTopology from an edge list."""
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


# ── Single edge tests ──────────────────────────────────────────────


class TestStaticSynapseSingleEdge:
    """Test with a single edge: pre=0 → post=0, w=0.5."""

    def test_fire_produces_current(self, synapse: StaticSynapseModel) -> None:
        """When pre fires, post current = weight."""
        edges = [(0, 0, 0.5)]
        topo = _make_topology(edges, n_pre=1, n_post=1)
        state = synapse.init_state(topo, "cpu", "float64")

        # Math prediction
        expected = StaticSynapseModel.predict_post_current(
            pre_fired=[0], edges=edges, n_post=1
        )
        assert expected == [0.5]

        # Model
        inputs = SynapseInputs(
            pre_spikes=torch.tensor([True]),
            post_spikes=torch.tensor([False]),
        )
        result = synapse.step(state, topo, inputs, None)
        assert isinstance(result, SynapseStepResult)
        actual = result.post_current[Compartment.SOMA]
        assert actual.item() == pytest.approx(expected[0])

    def test_no_fire_no_current(self, synapse: StaticSynapseModel) -> None:
        """When pre doesn't fire, post current = 0."""
        edges = [(0, 0, 0.5)]
        topo = _make_topology(edges, n_pre=1, n_post=1)
        state = synapse.init_state(topo, "cpu", "float64")

        expected = StaticSynapseModel.predict_post_current(
            pre_fired=[], edges=edges, n_post=1
        )
        assert expected == [0.0]

        inputs = SynapseInputs(
            pre_spikes=torch.tensor([False]),
            post_spikes=torch.tensor([False]),
        )
        result = synapse.step(state, topo, inputs, None)
        assert result.post_current[Compartment.SOMA].item() == pytest.approx(0.0)


# ── Multi-edge tests ───────────────────────────────────────────────


class TestStaticSynapseMultiEdge:
    """Test with multiple edges demonstrating summation."""

    def test_two_pre_one_post_both_fire(self, synapse: StaticSynapseModel) -> None:
        """Two pre-neurons both fire → post gets sum of weights."""
        edges = [(0, 0, 0.3), (1, 0, 0.7)]
        topo = _make_topology(edges, n_pre=2, n_post=1)
        state = synapse.init_state(topo, "cpu", "float64")

        expected = StaticSynapseModel.predict_post_current(
            pre_fired=[0, 1], edges=edges, n_post=1
        )
        assert expected == [pytest.approx(1.0)]

        inputs = SynapseInputs(
            pre_spikes=torch.tensor([True, True]),
            post_spikes=torch.tensor([False]),
        )
        result = synapse.step(state, topo, inputs, None)
        assert result.post_current[Compartment.SOMA].item() == pytest.approx(1.0)

    def test_two_pre_one_post_one_fires(self, synapse: StaticSynapseModel) -> None:
        """Only neuron 1 fires → post gets only w_1."""
        edges = [(0, 0, 0.3), (1, 0, 0.7)]
        topo = _make_topology(edges, n_pre=2, n_post=1)
        state = synapse.init_state(topo, "cpu", "float64")

        expected = StaticSynapseModel.predict_post_current(
            pre_fired=[1], edges=edges, n_post=1
        )
        assert expected == [pytest.approx(0.7)]

        inputs = SynapseInputs(
            pre_spikes=torch.tensor([False, True]),
            post_spikes=torch.tensor([False]),
        )
        result = synapse.step(state, topo, inputs, None)
        assert result.post_current[Compartment.SOMA].item() == pytest.approx(0.7)

    def test_fan_out(self, synapse: StaticSynapseModel) -> None:
        """One pre-neuron connects to multiple post-neurons."""
        # pre=0 → post=0 (w=0.4), pre=0 → post=1 (w=0.6)
        edges = [(0, 0, 0.4), (0, 1, 0.6)]
        topo = _make_topology(edges, n_pre=1, n_post=2)
        state = synapse.init_state(topo, "cpu", "float64")

        expected = StaticSynapseModel.predict_post_current(
            pre_fired=[0], edges=edges, n_post=2
        )
        assert expected == [pytest.approx(0.4), pytest.approx(0.6)]

        inputs = SynapseInputs(
            pre_spikes=torch.tensor([True]),
            post_spikes=torch.tensor([False, False]),
        )
        result = synapse.step(state, topo, inputs, None)
        actual = result.post_current[Compartment.SOMA]
        assert actual[0].item() == pytest.approx(0.4)
        assert actual[1].item() == pytest.approx(0.6)


# ── Inhibitory weights ─────────────────────────────────────────────


class TestStaticSynapseInhibitory:
    """Negative weights model inhibition."""

    def test_inhibitory_current(self, synapse: StaticSynapseModel) -> None:
        """Negative weight → negative post-synaptic current."""
        edges = [(0, 0, -0.8)]
        topo = _make_topology(edges, n_pre=1, n_post=1)
        state = synapse.init_state(topo, "cpu", "float64")

        expected = StaticSynapseModel.predict_post_current(
            pre_fired=[0], edges=edges, n_post=1
        )
        assert expected == [pytest.approx(-0.8)]

        inputs = SynapseInputs(
            pre_spikes=torch.tensor([True]),
            post_spikes=torch.tensor([False]),
        )
        result = synapse.step(state, topo, inputs, None)
        assert result.post_current[Compartment.SOMA].item() == pytest.approx(-0.8)

    def test_mixed_excitatory_inhibitory(self, synapse: StaticSynapseModel) -> None:
        """Mixed excitatory/inhibitory: net current is sum of all weights."""
        # pre=0 → post=0 (w=+0.5), pre=1 → post=0 (w=-0.3)
        edges = [(0, 0, 0.5), (1, 0, -0.3)]
        topo = _make_topology(edges, n_pre=2, n_post=1)
        state = synapse.init_state(topo, "cpu", "float64")

        expected = StaticSynapseModel.predict_post_current(
            pre_fired=[0, 1], edges=edges, n_post=1
        )
        assert expected == [pytest.approx(0.2)]

        inputs = SynapseInputs(
            pre_spikes=torch.tensor([True, True]),
            post_spikes=torch.tensor([False]),
        )
        result = synapse.step(state, topo, inputs, None)
        assert result.post_current[Compartment.SOMA].item() == pytest.approx(0.2)


# ── State & registry tests ─────────────────────────────────────────


class TestStaticSynapseRegistry:
    """Verify registry and state management."""

    def test_init_state_empty(self, synapse: StaticSynapseModel) -> None:
        """Static synapse has no internal state."""
        edges = [(0, 0, 1.0)]
        topo = _make_topology(edges, n_pre=1, n_post=1)
        state = synapse.init_state(topo, "cpu", "float64")
        assert state == {}

    def test_state_tensors_empty(self, synapse: StaticSynapseModel) -> None:
        """state_tensors returns empty dict for static synapse."""
        assert synapse.state_tensors({}) == {}

    def test_registered(self) -> None:
        from neuroforge.synapses.registry import SYNAPSE_MODELS

        assert SYNAPSE_MODELS.has("static")

    def test_create_static(self) -> None:
        from neuroforge.synapses.registry import create_synapse_model

        model = create_synapse_model("static")
        assert isinstance(model, StaticSynapseModel)

    def test_determinism(self, synapse: StaticSynapseModel) -> None:
        """Same inputs → same outputs (determinism check)."""
        edges = [(0, 0, 0.3), (1, 0, 0.7), (0, 1, 0.2)]
        topo = _make_topology(edges, n_pre=2, n_post=2)
        state = synapse.init_state(topo, "cpu", "float64")

        inputs = SynapseInputs(
            pre_spikes=torch.tensor([True, False]),
            post_spikes=torch.tensor([False, False]),
        )

        r1 = synapse.step(state, topo, inputs, None)
        r2 = synapse.step(state, topo, inputs, None)

        c1 = r1.post_current[Compartment.SOMA]
        c2 = r2.post_current[Compartment.SOMA]
        assert torch.equal(c1, c2)
