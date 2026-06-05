"""Binary-cross-entropy losses for spike-count learning."""

from __future__ import annotations

from typing import Any

__all__ = ["BceLogitsLoss"]


class BceLogitsLoss:
    """Binary cross-entropy with logits for spike-based binary outputs."""

    def __init__(self, reduction: str = "mean") -> None:
        self.reduction = reduction

    def __call__(self, logits: Any, target: Any) -> Any:
        """Compute BCE-with-logits loss."""
        from neuroforge.kernel.torch_utils import require_torch

        torch = require_torch()
        return torch.nn.functional.binary_cross_entropy_with_logits(
            logits, target, reduction=self.reduction,
        )
