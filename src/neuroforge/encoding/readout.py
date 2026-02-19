"""Spike-count readout — differentiable output decoding for SNN training.

Converts a window of output spikes ``[T, N_out]`` into scalar counts
and logits that loss functions can consume.
"""

from __future__ import annotations

from typing import Any

from neuroforge.contracts.encoding import ReadoutResult

__all__ = ["SpikeCountReadout", "ReadoutResult"]


class SpikeCountReadout:
    """Differentiable spike-count readout.

    Sums float (surrogate) spikes over the time axis and produces both
    a raw count and a logit value ``count − threshold``.

    Parameters
    ----------
    threshold:
        Spike-count threshold for the logit computation.
    """

    def __init__(self, threshold: float = 3.0) -> None:
        self.threshold = threshold

    def __call__(self, spikes: Any) -> ReadoutResult:
        """Compute readout from a spike tensor ``[T, N_out]``.

        Parameters
        ----------
        spikes:
            Float tensor of shape ``[T, N_out]`` containing 0/1 spike
            values (or fractional values if using partial spikes).

        Returns
        -------
        ReadoutResult:
            Count and logits for downstream loss computation.
        """
        count = spikes.sum(dim=0).float()  # [N_out]
        logits = count - self.threshold
        return ReadoutResult(count=count, logits=logits)
