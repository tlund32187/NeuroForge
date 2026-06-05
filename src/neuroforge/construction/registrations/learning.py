"""Register built-in learning encoders, readouts, and losses."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.construction.hub import FactoryHub

__all__ = ["register"]


def register(hub: FactoryHub) -> None:
    """Register built-in learning components."""
    from neuroforge.learning.encoders.factory import build_encoder_registry
    from neuroforge.learning.losses.registry import build_loss_registry
    from neuroforge.learning.readouts.factory import build_readout_registry

    hub.encoders = build_encoder_registry()
    hub.readouts = build_readout_registry()
    hub.losses = build_loss_registry()
