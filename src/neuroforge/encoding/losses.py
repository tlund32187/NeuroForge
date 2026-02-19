"""Loss functions for spike-based SNN training.

Provides MSE-count and BCE-logits losses that operate on the outputs of
:class:`~neuroforge.encoding.readout.SpikeCountReadout`.
"""

from __future__ import annotations

from typing import Any

__all__ = ["MseCountLoss", "BceLogitsLoss"]


class MseCountLoss:
    """MSE between spike count and a target count.

    Parameters
    ----------
    reduction:
        ``"mean"`` (default) or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, count: Any, target_count: Any) -> Any:
        """Compute MSE loss.

        Parameters
        ----------
        count:
            Predicted spike count (float tensor).
        target_count:
            Target spike count (float tensor, same shape).

        Returns
        -------
        Tensor:
            Scalar loss value.
        """
        diff = count - target_count
        sq = diff * diff
        if self.reduction == "sum":
            return sq.sum()
        return sq.mean()


class BceLogitsLoss:
    """Binary cross-entropy with logits for spike-based binary outputs.

    Wraps ``torch.nn.functional.binary_cross_entropy_with_logits`` so
    that callers need not import ``torch.nn`` directly.

    Parameters
    ----------
    reduction:
        ``"mean"`` (default) or ``"sum"``.
    """

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, logits: Any, target: Any) -> Any:
        """Compute BCE-with-logits loss.

        Parameters
        ----------
        logits:
            Predicted logit tensor (e.g. ``count − threshold``).
        target:
            Target binary tensor (0.0 or 1.0), same shape.

        Returns
        -------
        Tensor:
            Scalar loss value.
        """
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction=self.reduction,
        )
