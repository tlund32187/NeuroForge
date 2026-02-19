"""Loss registry — thin wrapper around :data:`DEFAULT_HUB.losses`.

Mirrors the style of ``neuroforge.neurons.registry`` and
``neuroforge.synapses.registry``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.factories.hub import DEFAULT_HUB

if TYPE_CHECKING:
    from neuroforge.core.registry import Registry

__all__ = ["LOSSES", "create_loss"]

LOSSES: Registry = DEFAULT_HUB.losses


def create_loss(key: str, **kwargs: object) -> object:
    """Create a loss function by registry key.

    Parameters
    ----------
    key:
        Loss key (e.g. ``"mse_count"``, ``"bce_logits"``).
    **kwargs:
        Forwarded to the loss constructor.
    """
    return LOSSES.create(key, **kwargs)
