"""GABA-B receptor model."""

from __future__ import annotations

from neuroforge.biology.receptors.base import ReceptorModelBase

__all__ = ["GABABReceptor"]


class GABABReceptor(ReceptorModelBase):
    """Inhibitory GABA-B receptor placeholder."""

    def current(self, activation: object) -> object:
        """Return activation as current."""
        return activation
