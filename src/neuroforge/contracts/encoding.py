"""Readout and loss contracts.

Defines the protocols that readout decoders and loss functions must
implement, plus the frozen DTO for readout results.  Follows the same
zero-runtime-dependency pattern as the rest of :mod:`contracts`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

__all__ = [
    "ILoss",
    "IReadout",
    "ReadoutResult",
]


@dataclass(frozen=True, slots=True)
class ReadoutResult:
    """Result of a readout operation.

    Attributes
    ----------
    count:
        Float spike count summed over the time dimension (shape ``[N_out]``).
    logits:
        ``count - threshold`` as a float tensor (shape ``[N_out]``),
        suitable for sigmoid / BCE-logits losses.
    """

    count: Any  # Tensor [N_out]
    logits: Any  # Tensor [N_out]


@runtime_checkable
class IReadout(Protocol):
    """Protocol for spike readout decoders.

    A readout converts a window of output spikes ``[T, N_out]`` into
    scalar counts and logits consumed by loss functions.

    Implementors must be callable:

        result = readout(spikes)  # -> ReadoutResult
    """

    def __call__(self, spikes: Any) -> ReadoutResult:
        """Decode spikes into a :class:`ReadoutResult`.

        Parameters
        ----------
        spikes:
            Float tensor of shape ``[T, N_out]``.

        Returns
        -------
        ReadoutResult:
            Aggregated count and logits.
        """
        ...  # pragma: no cover


@runtime_checkable
class ILoss(Protocol):
    """Protocol for SNN training loss functions.

    A loss takes a prediction tensor and a target tensor and returns a
    scalar loss value.

    Implementors must be callable:

        loss = loss_fn(prediction, target)  # -> Tensor
    """

    def __call__(self, prediction: Any, target: Any) -> Any:
        """Compute scalar loss.

        Parameters
        ----------
        prediction:
            Model output tensor (counts, logits, etc.).
        target:
            Ground-truth tensor, same shape as *prediction*.

        Returns
        -------
        Tensor:
            Scalar loss value.
        """
        ...  # pragma: no cover
