"""FactoryHub smoke tests for the construction registry bundle."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.unit
def test_construction_tree_contains_expected_modules() -> None:
    root = Path(__file__).parents[2] / "src" / "neuroforge" / "construction"
    expected = {
        "__init__.py",
        "registry.py",
        "hub.py",
        "composition_root.py",
        "network_factory.py",
        "task_factory.py",
        "monitor_factory.py",
        "registrations/__init__.py",
        "registrations/biology.py",
        "registrations/synapses.py",
        "registrations/plasticity.py",
        "registrations/perception.py",
        "registrations/learning.py",
        "registrations/applications.py",
    }

    missing = [path for path in sorted(expected) if not (root / path).is_file()]
    assert missing == []


@pytest.mark.unit
def test_build_default_hub_returns_independent_hub() -> None:
    from neuroforge.construction.composition_root import DEFAULT_HUB, build_default_hub
    from neuroforge.construction.hub import FactoryHub

    hub = build_default_hub()

    assert isinstance(hub, FactoryHub)
    assert hub is not DEFAULT_HUB
    assert hub.neurons is not DEFAULT_HUB.neurons


@pytest.mark.unit
def test_default_hub_has_all_registry_families() -> None:
    from neuroforge.construction.composition_root import DEFAULT_HUB

    for attr in (
        "neurons",
        "synapses",
        "encoders",
        "readouts",
        "vision_backbones",
        "losses",
        "learning_rules",
        "monitors",
    ):
        registry = getattr(DEFAULT_HUB, attr)
        assert registry is not None
        assert len(registry.list_keys()) > 0, f"{attr} registry is empty"


@pytest.mark.unit
def test_default_hub_has_expected_builtin_keys() -> None:
    from neuroforge.construction.composition_root import DEFAULT_HUB

    assert DEFAULT_HUB.neurons.has("lif")
    assert DEFAULT_HUB.neurons.has("lif_surr")
    assert DEFAULT_HUB.synapses.has("static")
    assert DEFAULT_HUB.synapses.has("static_dales")
    assert DEFAULT_HUB.synapses.has("static_delayed")
    assert DEFAULT_HUB.encoders.has("rate")
    assert DEFAULT_HUB.readouts.has("rate_decoder")
    assert DEFAULT_HUB.readouts.has("spike_count")
    assert DEFAULT_HUB.vision_backbones.has("lif_convnet_v1")
    assert DEFAULT_HUB.losses.has("mse_count")
    assert DEFAULT_HUB.losses.has("bce_logits")
    assert DEFAULT_HUB.learning_rules.has("rstdp")
    assert DEFAULT_HUB.monitors.has("training")


@pytest.mark.unit
def test_default_hub_creates_core_builtins() -> None:
    from neuroforge.biology.neurons.models.lif.model import LIFModel
    from neuroforge.biology.plasticity.rules.rstdp import RSTDPRule
    from neuroforge.biology.synapses.models.static import StaticSynapseModel
    from neuroforge.construction.composition_root import DEFAULT_HUB
    from neuroforge.learning.encoders.rate import RateEncoder
    from neuroforge.learning.losses import MseCountLoss
    from neuroforge.learning.readouts.rate_decoder import RateDecoder
    from neuroforge.learning.readouts.spike_count import SpikeCountReadout

    assert isinstance(DEFAULT_HUB.neurons.create("lif"), LIFModel)
    assert isinstance(DEFAULT_HUB.synapses.create("static"), StaticSynapseModel)
    assert isinstance(DEFAULT_HUB.encoders.create("rate"), RateEncoder)
    assert isinstance(DEFAULT_HUB.readouts.create("rate_decoder"), RateDecoder)
    assert isinstance(DEFAULT_HUB.readouts.create("spike_count"), SpikeCountReadout)
    assert isinstance(DEFAULT_HUB.losses.create("mse_count"), MseCountLoss)
    assert isinstance(DEFAULT_HUB.learning_rules.create("rstdp"), RSTDPRule)


@pytest.mark.unit
def test_default_hub_creates_vision_backbone_factory() -> None:
    from neuroforge.construction.composition_root import DEFAULT_HUB
    from neuroforge.perception.vision.factory import LIFConvNetV1BackboneFactory
    from neuroforge.simulation.topology.specs import (
        VisionBackboneSpec,
        VisionBlockSpec,
        VisionInputSpec,
    )

    spec = VisionBackboneSpec(
        type="lif_convnet_v1",
        input=VisionInputSpec(channels=1, height=8, width=8),
        time_steps=8,
        encoding_mode="poisson",
        blocks=[
            VisionBlockSpec(type="conv", params={"out_channels": 8, "kernel_size": 3})
        ],
        output_dim=32,
    )

    factory = DEFAULT_HUB.vision_backbones.create(spec.type, spec=spec)

    assert isinstance(factory, LIFConvNetV1BackboneFactory)


@pytest.mark.unit
def test_task_factory_creates_registered_task_specs() -> None:
    from neuroforge.construction.task_factory import build_task_factory

    factory = build_task_factory()

    assert "logic_gate" in factory.list_keys()
    assert factory.get("logic_gate") is not None


@pytest.mark.unit
def test_monitor_factory_creates_registered_monitors() -> None:
    from neuroforge.construction.monitor_factory import build_monitor_factory
    from neuroforge.observability.monitors.training_monitor import TrainingMonitor

    factory = build_monitor_factory()

    assert "training" in factory.list_keys()
    assert isinstance(factory.create("training"), TrainingMonitor)


@pytest.mark.unit
def test_unknown_key_raises() -> None:
    from neuroforge.construction.composition_root import DEFAULT_HUB

    with pytest.raises(KeyError, match="no_such_neuron"):
        DEFAULT_HUB.neurons.create("no_such_neuron")
