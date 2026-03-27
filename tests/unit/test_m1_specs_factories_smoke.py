"""FactoryHub smoke tests — DEFAULT_HUB, per-domain registries, backward compat.

Verifies that:
- ``FactoryHub`` / ``build_default_hub`` can be imported and used.
- ``DEFAULT_HUB`` contains the expected built-in registrations.
- Thin per-domain wrappers (``NEURON_MODELS``, ``SYNAPSE_MODELS``, etc.)
  point at the same underlying registry instances.
- ``create_neuron_model`` / ``create_synapse_model`` still work.
"""

from __future__ import annotations

import pytest  # noqa: TC002

# ── Hub construction ────────────────────────────────────────────────


@pytest.mark.unit
class TestFactoryHubConstruction:
    """FactoryHub can be built fresh and inspected."""

    def test_build_default_hub_returns_hub(self) -> None:
        from neuroforge.factories.hub import FactoryHub, build_default_hub

        hub = build_default_hub()
        assert isinstance(hub, FactoryHub)

    def test_default_hub_has_all_registries(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB

        for attr in (
            "neurons",
            "synapses",
            "encoders",
            "readouts",
            "vision_backbones",
            "losses",
        ):
            reg = getattr(DEFAULT_HUB, attr)
            assert reg is not None
            assert len(reg.list_keys()) > 0, f"{attr} registry is empty"


# ── Built-in registrations ──────────────────────────────────────────


@pytest.mark.unit
class TestHubBuiltins:
    """DEFAULT_HUB ships with all expected built-in keys."""

    def test_neurons_has_lif(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB

        assert DEFAULT_HUB.neurons.has("lif")

    def test_synapses_has_static(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB

        assert DEFAULT_HUB.synapses.has("static")

    def test_encoders_has_rate(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB

        assert DEFAULT_HUB.encoders.has("rate")

    def test_readouts_has_rate_decoder(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB

        assert DEFAULT_HUB.readouts.has("rate_decoder")

    def test_losses_has_mse_count(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB

        assert DEFAULT_HUB.losses.has("mse_count")

    def test_losses_has_bce_logits(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB

        assert DEFAULT_HUB.losses.has("bce_logits")

    def test_vision_backbones_has_lif_convnet_v1(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB

        assert DEFAULT_HUB.vision_backbones.has("lif_convnet_v1")


# ── create() round-trips ────────────────────────────────────────────


@pytest.mark.unit
class TestHubCreation:
    """DEFAULT_HUB.create() returns correct objects for real impls."""

    def test_create_lif_neuron(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB
        from neuroforge.neurons.lif.model import LIFModel

        model = DEFAULT_HUB.neurons.create("lif")
        assert isinstance(model, LIFModel)

    def test_create_static_synapse(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB
        from neuroforge.synapses.static import StaticSynapseModel

        model = DEFAULT_HUB.synapses.create("static")
        assert isinstance(model, StaticSynapseModel)

    def test_create_rate_encoder(self) -> None:
        from neuroforge.encoding.rate import RateEncoder
        from neuroforge.factories.hub import DEFAULT_HUB

        enc = DEFAULT_HUB.encoders.create("rate")
        assert isinstance(enc, RateEncoder)

    def test_create_rate_decoder(self) -> None:
        from neuroforge.encoding.decode import RateDecoder
        from neuroforge.factories.hub import DEFAULT_HUB

        dec = DEFAULT_HUB.readouts.create("rate_decoder")
        assert isinstance(dec, RateDecoder)

    def test_loss_mse_count_instantiates(self) -> None:
        from neuroforge.encoding.losses import MseCountLoss
        from neuroforge.factories.hub import DEFAULT_HUB

        obj = DEFAULT_HUB.losses.create("mse_count")
        assert isinstance(obj, MseCountLoss)

    def test_readout_spike_count_instantiates(self) -> None:
        from neuroforge.encoding.readout import SpikeCountReadout
        from neuroforge.factories.hub import DEFAULT_HUB

        obj = DEFAULT_HUB.readouts.create("spike_count")
        assert isinstance(obj, SpikeCountReadout)

    def test_vision_backbone_factory_instantiates(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB
        from neuroforge.network.specs import VisionBackboneSpec, VisionBlockSpec, VisionInputSpec
        from neuroforge.vision.factory import LIFConvNetV1BackboneFactory

        spec = VisionBackboneSpec(
            type="lif_convnet_v1",
            input=VisionInputSpec(channels=1, height=8, width=8),
            time_steps=8,
            encoding_mode="poisson",
            blocks=[VisionBlockSpec(type="conv", params={"out_channels": 8, "kernel_size": 3})],
            output_dim=32,
        )
        obj = DEFAULT_HUB.vision_backbones.create(spec.type, spec=spec)
        assert isinstance(obj, LIFConvNetV1BackboneFactory)

    def test_unknown_key_raises(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB

        with pytest.raises(KeyError, match="no_such_neuron"):
            DEFAULT_HUB.neurons.create("no_such_neuron")


# ── Backward-compatible wrappers ────────────────────────────────────


@pytest.mark.unit
class TestBackwardCompat:
    """Old import paths still work and point at the hub's registries."""

    def test_neuron_models_is_hub_neurons(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB
        from neuroforge.neurons.registry import NEURON_MODELS

        assert NEURON_MODELS is DEFAULT_HUB.neurons

    def test_synapse_models_is_hub_synapses(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB
        from neuroforge.synapses.registry import SYNAPSE_MODELS

        assert SYNAPSE_MODELS is DEFAULT_HUB.synapses

    def test_create_neuron_model_still_works(self) -> None:
        from neuroforge.neurons.lif.model import LIFModel
        from neuroforge.neurons.registry import create_neuron_model

        model = create_neuron_model("lif")
        assert isinstance(model, LIFModel)

    def test_create_synapse_model_still_works(self) -> None:
        from neuroforge.synapses.registry import create_synapse_model
        from neuroforge.synapses.static import StaticSynapseModel

        model = create_synapse_model("static")
        assert isinstance(model, StaticSynapseModel)

    def test_encoders_registry_is_hub_encoders(self) -> None:
        from neuroforge.encoding.registry import ENCODERS
        from neuroforge.factories.hub import DEFAULT_HUB

        assert ENCODERS is DEFAULT_HUB.encoders

    def test_readouts_registry_is_hub_readouts(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB
        from neuroforge.readout.registry import READOUTS

        assert READOUTS is DEFAULT_HUB.readouts

    def test_losses_registry_is_hub_losses(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB
        from neuroforge.losses.registry import LOSSES

        assert LOSSES is DEFAULT_HUB.losses

    def test_vision_registry_is_hub_vision_backbones(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB
        from neuroforge.vision.registry import VISION_BACKBONES

        assert VISION_BACKBONES is DEFAULT_HUB.vision_backbones


# ── Fresh hub isolation ─────────────────────────────────────────────


@pytest.mark.unit
class TestFreshHub:
    """build_default_hub() returns an independent hub each time."""

    def test_fresh_hub_has_lif(self) -> None:
        from neuroforge.factories.hub import build_default_hub

        hub = build_default_hub()
        assert hub.neurons.has("lif")

    def test_fresh_hub_is_distinct(self) -> None:
        from neuroforge.factories.hub import DEFAULT_HUB, build_default_hub

        hub2 = build_default_hub()
        assert hub2 is not DEFAULT_HUB
        assert hub2.neurons is not DEFAULT_HUB.neurons
