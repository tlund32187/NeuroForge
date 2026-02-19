"""Tests for gate-network builder, Dale's synapse, readout, and losses."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from neuroforge.contracts.encoding import ILoss, IReadout
from neuroforge.contracts.synapses import SynapseInputs, SynapseTopology
from neuroforge.contracts.types import Compartment
from neuroforge.encoding.losses import BceLogitsLoss, MseCountLoss
from neuroforge.encoding.readout import ReadoutResult, SpikeCountReadout
from neuroforge.factories.hub import DEFAULT_HUB, build_default_hub
from neuroforge.network.gate_builder import (
    GateNetwork,
    build_dale_signs,
    build_gate_network,
    build_projection,
    init_projection_weights,
)
from neuroforge.network.specs import GateNetworkSpec
from neuroforge.synapses.dales_static import DalesStaticSynapseModel

# ── Hub registry look-up tests ─────────────────────────────────────


class TestHubRegistrations:
    """Verify that new components are discoverable in the default hub."""

    def test_static_dales_registered(self) -> None:
        assert DEFAULT_HUB.synapses.has("static_dales")

    def test_spike_count_readout_registered(self) -> None:
        assert DEFAULT_HUB.readouts.has("spike_count")

    def test_mse_count_loss_registered(self) -> None:
        assert DEFAULT_HUB.losses.has("mse_count")

    def test_bce_logits_loss_registered(self) -> None:
        assert DEFAULT_HUB.losses.has("bce_logits")

    def test_create_spike_count_readout(self) -> None:
        readout = DEFAULT_HUB.readouts.create("spike_count", threshold=2.0)
        assert isinstance(readout, SpikeCountReadout)

    def test_create_mse_count_loss(self) -> None:
        loss_fn = DEFAULT_HUB.losses.create("mse_count")
        assert isinstance(loss_fn, MseCountLoss)

    def test_create_bce_logits_loss(self) -> None:
        loss_fn = DEFAULT_HUB.losses.create("bce_logits")
        assert isinstance(loss_fn, BceLogitsLoss)

    def test_create_dales_synapse(self) -> None:
        sign = torch.ones(3)
        syn = DEFAULT_HUB.synapses.create("static_dales", sign_pre=sign)
        assert isinstance(syn, DalesStaticSynapseModel)


# ── DalesStaticSynapseModel unit tests ─────────────────────────────


class TestDalesStaticSynapseModel:
    """Verify Dale's-Law weight reparameterisation."""

    @pytest.fixture()
    def simple_setup(self) -> dict[str, Any]:
        """2-pre, 2-post dense topology with mixed signs."""
        n_pre, n_post = 2, 2
        sign_pre = torch.tensor([1.0, -1.0])
        model = DalesStaticSynapseModel(sign_pre=sign_pre)

        # Edges: (0→0), (0→1), (1→0), (1→1)
        pre_idx = torch.tensor([0, 0, 1, 1])
        post_idx = torch.tensor([0, 1, 0, 1])
        weights = torch.tensor([0.5, 0.3, -0.4, 0.2])
        delays = torch.zeros(4, dtype=torch.int64)

        topo = SynapseTopology(
            pre_idx=pre_idx,
            post_idx=post_idx,
            weights=weights,
            delays=delays,
            n_pre=n_pre,
            n_post=n_post,
        )
        return {"model": model, "topo": topo, "n_post": n_post}

    def test_excitatory_positive(self, simple_setup: dict[str, Any]) -> None:
        model = simple_setup["model"]
        topo = simple_setup["topo"]

        spikes = torch.tensor([True, False])
        inputs = SynapseInputs(pre_spikes=spikes, post_spikes=torch.tensor([False, False]))
        result = model.step({}, topo, inputs, ctx=None)
        psc = result.post_current[Compartment.SOMA]
        # Only neuron 0 spiked (sign +1), edges 0→0 and 0→1 fire.
        assert psc[0].item() == pytest.approx(0.5, abs=1e-6)
        assert psc[1].item() == pytest.approx(0.3, abs=1e-6)

    def test_inhibitory_negative(self, simple_setup: dict[str, Any]) -> None:
        model = simple_setup["model"]
        topo = simple_setup["topo"]

        spikes = torch.tensor([False, True])
        inputs = SynapseInputs(pre_spikes=spikes, post_spikes=torch.tensor([False, False]))
        result = model.step({}, topo, inputs, ctx=None)
        psc = result.post_current[Compartment.SOMA]
        # Only neuron 1 spiked (sign −1).  w_eff = |w| × (−1).
        assert psc[0].item() == pytest.approx(-0.4, abs=1e-6)
        assert psc[1].item() == pytest.approx(-0.2, abs=1e-6)

    def test_float_spikes_propagation(self, simple_setup: dict[str, Any]) -> None:
        model = simple_setup["model"]
        topo = simple_setup["topo"]

        spikes = torch.tensor([0.8, 0.0])
        inputs = SynapseInputs(pre_spikes=spikes, post_spikes=torch.tensor([0.0, 0.0]))
        result = model.step({}, topo, inputs, ctx=None)
        psc = result.post_current[Compartment.SOMA]
        # 0.8 × |0.5| × +1 = 0.4, 0.8 × |0.3| × +1 = 0.24
        assert psc[0].item() == pytest.approx(0.4, abs=1e-5)
        assert psc[1].item() == pytest.approx(0.24, abs=1e-5)

    def test_no_sign_acts_as_identity(self) -> None:
        """When sign_pre is None, w_eff = weights (no reparameterisation)."""
        model = DalesStaticSynapseModel(sign_pre=None)
        pre_idx = torch.tensor([0, 1])
        post_idx = torch.tensor([0, 0])
        weights = torch.tensor([0.5, -0.3])
        delays = torch.zeros(2, dtype=torch.int64)
        topo = SynapseTopology(pre_idx=pre_idx, post_idx=post_idx, weights=weights,
                               delays=delays, n_pre=2, n_post=1)
        spikes = torch.tensor([True, True])
        inputs = SynapseInputs(pre_spikes=spikes, post_spikes=torch.tensor([False]))
        result = model.step({}, topo, inputs, ctx=None)
        assert result.post_current[Compartment.SOMA][0].item() == pytest.approx(0.2, abs=1e-6)

    def test_init_state_empty(self) -> None:
        model = DalesStaticSynapseModel()
        topo = SynapseTopology(
            pre_idx=torch.tensor([0]), post_idx=torch.tensor([0]),
            weights=torch.tensor([1.0]), delays=torch.zeros(1, dtype=torch.int64),
            n_pre=1, n_post=1,
        )
        state = model.init_state(topo, "cpu", "float32")
        assert state == {}


# ── SpikeCountReadout unit tests ───────────────────────────────────


class TestSpikeCountReadout:
    """Verify spike-count to logit conversion."""

    def test_basic_count(self) -> None:
        readout = SpikeCountReadout(threshold=3.0)
        spikes = torch.ones(5, 2)  # 5 timesteps, 2 output neurons → count = 5 each
        result = readout(spikes)
        assert isinstance(result, ReadoutResult)
        assert result.count.tolist() == pytest.approx([5.0, 5.0])
        assert result.logits.tolist() == pytest.approx([2.0, 2.0])  # 5−3

    def test_zero_spikes(self) -> None:
        readout = SpikeCountReadout(threshold=3.0)
        spikes = torch.zeros(10, 1)
        result = readout(spikes)
        assert result.count.item() == pytest.approx(0.0)
        assert result.logits.item() == pytest.approx(-3.0)

    def test_custom_threshold(self) -> None:
        readout = SpikeCountReadout(threshold=1.0)
        spikes = torch.tensor([[1.0], [1.0], [0.0]])
        result = readout(spikes)
        assert result.count.item() == pytest.approx(2.0)
        assert result.logits.item() == pytest.approx(1.0)

    def test_preserves_gradient(self) -> None:
        readout = SpikeCountReadout(threshold=3.0)
        spikes = torch.randn(4, 1, requires_grad=True)
        result = readout(spikes)
        result.logits.sum().backward()
        assert spikes.grad is not None


# ── Loss function unit tests ───────────────────────────────────────


class TestMseCountLoss:
    """Verify MSE spike-count loss."""

    def test_zero_loss_when_exact(self) -> None:
        loss_fn = MseCountLoss()
        c = torch.tensor([5.0])
        assert loss_fn(c, c).item() == pytest.approx(0.0)

    def test_positive_loss(self) -> None:
        loss_fn = MseCountLoss()
        c = torch.tensor([4.0])
        t = torch.tensor([2.0])
        assert loss_fn(c, t).item() == pytest.approx(4.0)

    def test_reduction_sum(self) -> None:
        loss_fn = MseCountLoss(reduction="sum")
        c = torch.tensor([4.0, 6.0])
        t = torch.tensor([2.0, 3.0])
        # (4−2)^2 + (6−3)^2 = 4 + 9 = 13
        assert loss_fn(c, t).item() == pytest.approx(13.0)

    def test_gradient_flows(self) -> None:
        loss_fn = MseCountLoss()
        c = torch.tensor([3.0], requires_grad=True)
        t = torch.tensor([1.0])
        loss_fn(c, t).backward()
        assert c.grad is not None and c.grad.item() != 0.0


class TestBceLogitsLoss:
    """Verify BCE-with-logits wrapper."""

    def test_positive_loss(self) -> None:
        loss_fn = BceLogitsLoss()
        logits = torch.tensor([0.0])
        target = torch.tensor([1.0])
        loss = loss_fn(logits, target)
        # −log(σ(0)) = −log(0.5) ≈ 0.6931
        assert loss.item() == pytest.approx(0.6931, abs=1e-3)

    def test_gradient_flows(self) -> None:
        loss_fn = BceLogitsLoss()
        logits = torch.tensor([1.0], requires_grad=True)
        target = torch.tensor([0.0])
        loss_fn(logits, target).backward()
        assert logits.grad is not None and logits.grad.item() != 0.0


# ── build_gate_network tests ───────────────────────────────────────


class TestBuildGateNetwork:
    """Verify end-to-end gate-network construction."""

    def test_default_spec_produces_gate_network(self) -> None:
        gn = build_gate_network(GateNetworkSpec())
        assert isinstance(gn, GateNetwork)
        assert gn.engine._built  # noqa: SLF001

    def test_populations_present(self) -> None:
        gn = build_gate_network(GateNetworkSpec())
        pop_names = set(gn.engine.populations.keys())
        assert pop_names == {"input", "hidden", "output"}

    def test_projections_present(self) -> None:
        gn = build_gate_network(GateNetworkSpec())
        proj_names = set(gn.engine.projections.keys())
        assert proj_names == {"in_to_hidden", "hidden_to_out"}

    def test_trainable_keys(self) -> None:
        gn = build_gate_network(GateNetworkSpec())
        expected = {
            "raw_w_in_to_hidden", "bias_in_to_hidden",
            "raw_w_hidden_to_out", "bias_hidden_to_out",
        }
        assert set(gn.trainables.keys()) == expected

    def test_trainables_require_grad(self) -> None:
        gn = build_gate_network(GateNetworkSpec())
        for name, t in gn.trainables.items():
            assert t.requires_grad, f"{name} should require grad"

    def test_meta_contains_sizes(self) -> None:
        gn = build_gate_network(GateNetworkSpec())
        assert gn.meta["input_size"] == 2
        assert gn.meta["hidden_size"] == 6
        assert gn.meta["output_size"] == 1

    def test_dale_sign_in_meta(self) -> None:
        gn = build_gate_network(GateNetworkSpec(hidden_size=6, n_inhibitory_hidden=2))
        sh = gn.meta["sign_hidden"]
        assert sh.shape == (6,)
        assert (sh[:4] == 1.0).all()
        assert (sh[4:] == -1.0).all()

    def test_no_hidden_layer(self) -> None:
        spec = GateNetworkSpec(hidden_size=0)
        gn = build_gate_network(spec)
        assert set(gn.engine.populations.keys()) == {"input", "output"}
        assert set(gn.engine.projections.keys()) == {"in_to_out"}
        assert "sign_hidden" not in gn.meta

    def test_sparse_connectivity(self) -> None:
        spec = GateNetworkSpec(hidden_size=10, p_connect=0.5, seed=99)
        gn = build_gate_network(spec)
        # With p_connect=0.5, at least some edges should be pruned.
        proj = gn.engine.projections["in_to_hidden"]
        n_full = spec.input_size * spec.hidden_size
        n_actual = proj.topology.pre_idx.shape[0]
        assert n_actual < n_full

    def test_reproducible_weights(self) -> None:
        spec = GateNetworkSpec(seed=123)
        gn1 = build_gate_network(spec)
        gn2 = build_gate_network(spec)
        w1 = gn1.trainables["raw_w_in_to_hidden"]
        w2 = gn2.trainables["raw_w_in_to_hidden"]
        assert torch.equal(w1, w2)

    def test_custom_init_scale(self) -> None:
        spec = GateNetworkSpec(init_scale=0.01, hidden_size=0)
        gn = build_gate_network(spec)
        w = gn.trainables["raw_w_in_to_out"]
        # All weights should be within [−0.01, 0.01].
        assert w.abs().max().item() <= 0.01 + 1e-7

    def test_output_size_respected(self) -> None:
        spec = GateNetworkSpec(output_size=3, hidden_size=0)
        gn = build_gate_network(spec)
        assert gn.engine.populations["output"].n == 3

    def test_explicit_hub(self) -> None:
        hub = build_default_hub()
        gn = build_gate_network(GateNetworkSpec(), hub=hub)
        assert isinstance(gn, GateNetwork)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
    def test_cuda_device(self) -> None:
        spec = GateNetworkSpec(device="cuda", dtype="float32")
        gn = build_gate_network(spec)
        w = gn.trainables["raw_w_in_to_hidden"]
        assert w.device.type == "cuda"


# ── End-to-end gradient flow test ──────────────────────────────────


class TestGateNetworkGradientFlow:
    """Verify gradients flow from loss through the full pipeline."""

    def test_mse_loss_grads_flow(self) -> None:
        spec = GateNetworkSpec(hidden_size=0, seed=42)
        gn = build_gate_network(spec)
        readout = SpikeCountReadout(threshold=3.0)
        loss_fn = MseCountLoss()

        # Fake output spikes (simulate T=5 timesteps, 1 output neuron).
        fake_spikes = gn.trainables["raw_w_in_to_out"].unsqueeze(0).expand(5, -1)
        result = readout(fake_spikes)
        loss = loss_fn(result.count, torch.tensor([3.0, 3.0]))
        loss.backward()

        assert gn.trainables["raw_w_in_to_out"].grad is not None

    def test_bce_loss_grads_flow(self) -> None:
        spec = GateNetworkSpec(hidden_size=0, seed=42)
        gn = build_gate_network(spec)
        readout = SpikeCountReadout(threshold=3.0)
        loss_fn = BceLogitsLoss()

        fake_spikes = gn.trainables["raw_w_in_to_out"].unsqueeze(0).expand(5, -1)
        result = readout(fake_spikes)
        target = torch.tensor([1.0, 1.0])
        loss = loss_fn(result.logits, target)
        loss.backward()

        assert gn.trainables["raw_w_in_to_out"].grad is not None


# ── Protocol conformance tests ─────────────────────────────────────


class TestProtocolConformance:
    """Verify that concrete readouts and losses satisfy their protocols."""

    def test_spike_count_readout_is_ireadout(self) -> None:
        assert isinstance(SpikeCountReadout(), IReadout)

    def test_mse_count_loss_is_iloss(self) -> None:
        assert isinstance(MseCountLoss(), ILoss)

    def test_bce_logits_loss_is_iloss(self) -> None:
        assert isinstance(BceLogitsLoss(), ILoss)


# ── GateNetworkSpec model key tests ────────────────────────────────


class TestGateNetworkSpecModelKeys:
    """Verify neuron_model / synapse_model fields on GateNetworkSpec."""

    def test_default_neuron_model(self) -> None:
        assert GateNetworkSpec().neuron_model == "lif_surr"

    def test_default_synapse_model(self) -> None:
        assert GateNetworkSpec().synapse_model == "static_dales"

    def test_custom_neuron_model_accepted(self) -> None:
        spec = GateNetworkSpec(neuron_model="lif_surrogate")
        gn = build_gate_network(spec)
        assert isinstance(gn, GateNetwork)

    def test_custom_synapse_model_accepted(self) -> None:
        """Use 'static' (no Dale's) — sign_pre should be ignored."""
        spec = GateNetworkSpec(synapse_model="static", hidden_size=0)
        gn = build_gate_network(spec)
        assert isinstance(gn, GateNetwork)


# ── build_dale_signs unit tests ────────────────────────────────────


class TestBuildDaleSigns:
    """Verify Dale sign-mask construction in isolation."""

    def test_input_always_excitatory(self) -> None:
        signs = build_dale_signs(3, 0, 0, "cpu", torch.float64, torch)
        assert signs["sign_input"].tolist() == [1.0, 1.0, 1.0]

    def test_no_sign_hidden_when_no_hidden(self) -> None:
        signs = build_dale_signs(2, 0, 0, "cpu", torch.float64, torch)
        assert "sign_hidden" not in signs

    def test_mixed_excitatory_inhibitory(self) -> None:
        signs = build_dale_signs(2, 6, 2, "cpu", torch.float64, torch)
        sh = signs["sign_hidden"]
        assert sh.shape == (6,)
        assert (sh[:4] == 1.0).all()
        assert (sh[4:] == -1.0).all()

    def test_all_excitatory_hidden(self) -> None:
        signs = build_dale_signs(2, 4, 0, "cpu", torch.float64, torch)
        assert (signs["sign_hidden"] == 1.0).all()

    def test_all_inhibitory_hidden(self) -> None:
        signs = build_dale_signs(1, 3, 3, "cpu", torch.float64, torch)
        assert (signs["sign_hidden"] == -1.0).all()


# ── init_projection_weights unit tests ─────────────────────────────


class TestInitProjectionWeights:
    """Verify trainable weight/bias allocation in isolation."""

    def test_shapes_2d(self) -> None:
        raw_w, bias = init_projection_weights(3, 4, 0.3, torch.float64, "cpu", torch)
        assert raw_w.shape == (3, 4)
        assert bias.shape == (4,)

    def test_shapes_1d_squeeze(self) -> None:
        raw_w, bias = init_projection_weights(5, 1, 0.3, torch.float64, "cpu", torch)
        assert raw_w.shape == (5,)
        assert bias.shape == (1,)

    def test_requires_grad(self) -> None:
        raw_w, bias = init_projection_weights(2, 2, 0.1, torch.float64, "cpu", torch)
        assert raw_w.requires_grad
        assert bias.requires_grad

    def test_within_init_scale(self) -> None:
        raw_w, _ = init_projection_weights(50, 50, 0.05, torch.float64, "cpu", torch)
        assert raw_w.abs().max().item() <= 0.05 + 1e-7

    def test_bias_initialised_to_zero(self) -> None:
        _, bias = init_projection_weights(3, 3, 0.3, torch.float64, "cpu", torch)
        assert (bias == 0.0).all()


# ── build_projection unit tests ────────────────────────────────────


class TestBuildProjection:
    """Verify single-projection assembly in isolation."""

    @pytest.fixture()
    def proj_kwargs(self) -> dict[str, Any]:
        """Common keyword arguments for build_projection."""
        rng = torch.Generator()
        rng.manual_seed(42)
        return {
            "synapse_key": "static_dales",
            "p_connect": 1.0,
            "init_scale": 0.3,
            "device": "cpu",
            "dtype": "float64",
            "hub": DEFAULT_HUB,
            "dev": torch.device("cpu"),
            "tdt": torch.float64,
            "rng": rng,
            "torch_mod": torch,
        }

    def test_returns_projection_and_trainables(
        self, proj_kwargs: dict[str, Any],
    ) -> None:
        sign = torch.ones(2)
        proj, trainables = build_projection(
            "test_proj", "src", "tgt", 2, 3, sign, **proj_kwargs,
        )
        assert proj.name == "test_proj"
        assert proj.source == "src"
        assert proj.target == "tgt"
        assert "raw_w_test_proj" in trainables
        assert "bias_test_proj" in trainables

    def test_trainables_require_grad(
        self, proj_kwargs: dict[str, Any],
    ) -> None:
        sign = torch.ones(2)
        _, trainables = build_projection(
            "p", "s", "t", 2, 3, sign, **proj_kwargs,
        )
        for t in trainables.values():
            assert t.requires_grad

    def test_topology_edge_count_dense(
        self, proj_kwargs: dict[str, Any],
    ) -> None:
        sign = torch.ones(3)
        proj, _ = build_projection(
            "p", "s", "t", 3, 4, sign, **proj_kwargs,
        )
        assert proj.topology.pre_idx.shape[0] == 12  # 3×4

    def test_sparse_fewer_edges(
        self, proj_kwargs: dict[str, Any],
    ) -> None:
        proj_kwargs["p_connect"] = 0.5
        sign = torch.ones(10)
        proj, _ = build_projection(
            "p", "s", "t", 10, 10, sign, **proj_kwargs,
        )
        assert proj.topology.pre_idx.shape[0] < 100
