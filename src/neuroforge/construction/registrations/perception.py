"""Register built-in perception components."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.construction.hub import FactoryHub

__all__ = ["register"]


def register(hub: FactoryHub) -> None:
    """Register built-in perception components."""
    from neuroforge.perception.vision.registry import build_vision_backbone_registry

    hub.vision_backbones = build_vision_backbone_registry()
