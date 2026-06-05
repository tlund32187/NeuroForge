"""Register built-in biological plasticity rules."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.construction.hub import FactoryHub

__all__ = ["register"]


def register(hub: FactoryHub) -> None:
    """Register built-in plasticity rules."""
    from neuroforge.biology.plasticity.rules.rstdp import RSTDPRule

    hub.learning_rules.register("rstdp", RSTDPRule)
