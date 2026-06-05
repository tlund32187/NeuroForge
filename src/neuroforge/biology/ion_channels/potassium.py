"""Potassium ion-channel model."""

from __future__ import annotations

from neuroforge.biology.ion_channels.base import IonChannelBase

__all__ = ["PotassiumChannel"]


class PotassiumChannel(IonChannelBase):
    """Zero-current potassium channel placeholder."""

    def current(self, voltage: object) -> float:
        """Return zero current."""
        _ = voltage
        return 0.0
