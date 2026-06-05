"""Vision component registries."""

from __future__ import annotations

from neuroforge.construction.registry import Registry
from neuroforge.perception.vision.factory import LIFConvNetV1BackboneFactory

__all__ = ["build_vision_backbone_registry"]


def build_vision_backbone_registry() -> Registry:
    """Build a registry populated with built-in vision backbones."""
    registry = Registry("vision_backbones")
    registry.register("lif_convnet_v1", LIFConvNetV1BackboneFactory)
    return registry
