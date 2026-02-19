"""FactoryHub — injectable registry bundle for NeuroForge components.

Provides a single object that owns :class:`Registry` instances for every
component kind (neurons, synapses, encoders, readouts, losses).  The
module-level :data:`DEFAULT_HUB` is the global singleton that existing
registries delegate to.

Design
------
- **Pure construction** — no file I/O, no UI imports.
- **Dependency Inversion** — callers accept a ``FactoryHub`` instead of
  reaching for module-level singletons, making alternative wiring trivial
  in tests or alternative runtimes.
- **Backward-compatible** — ``DEFAULT_HUB`` is populated with the same
  built-ins that the old per-module registries registered, and the old
  ``NEURON_MODELS`` / ``SYNAPSE_MODELS`` globals now point at the hub's
  registries.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from neuroforge.core.registry import Registry

__all__ = ["FactoryHub", "build_default_hub", "DEFAULT_HUB"]


@dataclass
class FactoryHub:
    """Bundle of typed registries — one per component kind.

    Attributes
    ----------
    neurons:
        Neuron model constructors (e.g. ``"lif"``).
    synapses:
        Synapse model constructors (e.g. ``"static"``).
    encoders:
        Input encoder constructors (e.g. ``"rate"``).
    readouts:
        Output readout constructors (e.g. ``"rate_decoder"``).
    losses:
        Loss function constructors (e.g. ``"bce_logits"``).
    """

    neurons: Registry = field(default_factory=lambda: Registry("neurons"))
    synapses: Registry = field(default_factory=lambda: Registry("synapses"))
    encoders: Registry = field(default_factory=lambda: Registry("encoders"))
    readouts: Registry = field(default_factory=lambda: Registry("readouts"))
    losses: Registry = field(default_factory=lambda: Registry("losses"))


# ── Built-in registration ──────────────────────────────────────────


def _register_neurons(hub: FactoryHub) -> None:
    from neuroforge.neurons.lif.model import LIFModel
    from neuroforge.neurons.lif.surrogate import SurrogateLIFModel

    hub.neurons.register("lif", LIFModel)
    hub.neurons.register("lif_surr", SurrogateLIFModel)
    hub.neurons.register("lif_surrogate", SurrogateLIFModel)


def _register_synapses(hub: FactoryHub) -> None:
    from neuroforge.synapses.dales_static import DalesStaticSynapseModel
    from neuroforge.synapses.delayed_static import DelayedStaticSynapseModel
    from neuroforge.synapses.static import StaticSynapseModel

    hub.synapses.register("static", StaticSynapseModel)
    hub.synapses.register("static_dales", DalesStaticSynapseModel)
    hub.synapses.register("static_delayed", DelayedStaticSynapseModel)


def _register_encoders(hub: FactoryHub) -> None:
    from neuroforge.encoding.rate import RateEncoder

    hub.encoders.register("rate", RateEncoder)


def _register_readouts(hub: FactoryHub) -> None:
    from neuroforge.encoding.decode import RateDecoder
    from neuroforge.encoding.readout import SpikeCountReadout

    hub.readouts.register("rate_decoder", RateDecoder)
    hub.readouts.register("spike_count", SpikeCountReadout)


def _register_losses(hub: FactoryHub) -> None:
    from neuroforge.encoding.losses import BceLogitsLoss, MseCountLoss

    hub.losses.register("mse_count", MseCountLoss)
    hub.losses.register("bce_logits", BceLogitsLoss)


def build_default_hub() -> FactoryHub:
    """Create a :class:`FactoryHub` populated with all built-in components."""
    hub = FactoryHub()
    _register_neurons(hub)
    _register_synapses(hub)
    _register_encoders(hub)
    _register_readouts(hub)
    _register_losses(hub)
    return hub


DEFAULT_HUB: FactoryHub = build_default_hub()
