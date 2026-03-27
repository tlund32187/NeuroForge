"""Vision backbone registry - thin wrapper around ``DEFAULT_HUB.vision_backbones``."""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.factories.hub import DEFAULT_HUB
from neuroforge.vision.factory import ResolvedVisionBackbone, VisionBackboneFactory

if TYPE_CHECKING:
    from neuroforge.core.registry import Registry
    from neuroforge.network.specs import VisionBackboneSpec

__all__ = ["VISION_BACKBONES", "create_vision_backbone"]

VISION_BACKBONES: Registry = DEFAULT_HUB.vision_backbones


def create_vision_backbone(spec: VisionBackboneSpec) -> ResolvedVisionBackbone:
    """Resolve and build a vision backbone from config via registry key."""
    factory_obj = VISION_BACKBONES.create(spec.type, spec=spec)
    if not isinstance(factory_obj, VisionBackboneFactory):
        msg = (
            "vision_backbones registry must return a VisionBackboneFactory; "
            f"got {type(factory_obj).__name__}"
        )
        raise TypeError(msg)
    return factory_obj.build()

