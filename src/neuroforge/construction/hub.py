"""Registry bundle used by the construction composition root."""

from __future__ import annotations

from dataclasses import dataclass, field

from neuroforge.construction.registry import Registry

__all__ = ["FactoryHub"]


@dataclass
class FactoryHub:
    """Bundle of registries used to compose NeuroForge systems."""

    neurons: Registry = field(default_factory=lambda: Registry("neurons"))
    synapses: Registry = field(default_factory=lambda: Registry("synapses"))
    encoders: Registry = field(default_factory=lambda: Registry("encoders"))
    readouts: Registry = field(default_factory=lambda: Registry("readouts"))
    vision_backbones: Registry = field(
        default_factory=lambda: Registry("vision_backbones")
    )
    losses: Registry = field(default_factory=lambda: Registry("losses"))
    learning_rules: Registry = field(default_factory=lambda: Registry("learning_rules"))
    monitors: Registry = field(default_factory=lambda: Registry("monitors"))
