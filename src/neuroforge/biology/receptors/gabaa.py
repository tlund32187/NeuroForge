"""GABA-A receptor model."""

from __future__ import annotations

from neuroforge.biology.receptors.base import ReceptorModelBase

__all__ = ["GABAAReceptor"]


class GABAAReceptor(ReceptorModelBase):
    """Inhibitory GABA-A receptor placeholder."""

    def current(self, activation: object) -> object:
        """Return activation as current."""
        return activation
