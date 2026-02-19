"""Surrogate spike functions for autograd-compatible SNN training.

The straight-through estimator uses a hard step in the forward pass
(preserving exact 0/1 spike semantics) and a smooth surrogate gradient
in the backward pass so that ``torch.autograd`` can propagate useful
gradients through the spiking non-linearity.
"""

from __future__ import annotations

from typing import Any

__all__ = ["surrogate_spike"]


def surrogate_spike(v_minus_thresh: Any, beta: float = 5.0) -> Any:
    """Apply a surrogate spike function to ``v - v_thresh``.

    Forward:  ``(v_minus_thresh >= 0).float()``  (hard Heaviside step).
    Backward: fast-sigmoid derivative  ``beta / (2 * (1 + beta*|x|)^2)``.

    Parameters
    ----------
    v_minus_thresh:
        Tensor of (voltage − threshold) values.
    beta:
        Steepness of the surrogate gradient (default 5.0).

    Returns
    -------
    Tensor:
        Float tensor of 0.0 / 1.0 spikes, with surrogate gradient
        attached for autograd.
    """
    return _fast_sigmoid_spike(v_minus_thresh, beta)


def _fast_sigmoid_spike(v_minus_thresh: Any, beta: float) -> Any:
    """Dispatch to the custom autograd Function (lazy-built on first call)."""
    global _FastSigmoidSpike  # noqa: PLW0603
    if _FastSigmoidSpike is None:
        _FastSigmoidSpike = _build_autograd_fn()
    return _FastSigmoidSpike.apply(v_minus_thresh, beta)


# Lazily populated by ``_build_autograd_fn`` on first use.
_FastSigmoidSpike: Any = None


def _build_autograd_fn() -> Any:
    """Create the custom ``torch.autograd.Function`` subclass.

    Built inside a function so ``torch`` is only imported when needed.
    """
    from neuroforge.core.torch_utils import require_torch

    torch = require_torch()

    class FastSigmoidSpike(torch.autograd.Function):  # type: ignore[misc,name-defined]
        """Custom autograd function — hard step fwd, fast-sigmoid bwd."""

        @staticmethod
        def forward(ctx: Any, v_minus_thresh: Any, beta: float) -> Any:  # noqa: ANN401
            spikes = (v_minus_thresh >= 0).to(v_minus_thresh.dtype)
            ctx.save_for_backward(v_minus_thresh)
            ctx.beta = beta
            return spikes

        @staticmethod
        def backward(ctx: Any, grad_output: Any) -> tuple[Any, None]:  # noqa: ANN401
            (v_minus_thresh,) = ctx.saved_tensors
            beta: float = ctx.beta
            grad = grad_output * beta / (2.0 * (1.0 + beta * v_minus_thresh.abs()) ** 2)
            return grad, None

    return FastSigmoidSpike
