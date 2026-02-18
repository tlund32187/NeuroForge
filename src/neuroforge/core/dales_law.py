"""Dale's Law enforcement — constraining neurons to pure E/I identity.

Dale's Law (Dale's Principle) states that a neuron releases the same
set of neurotransmitters at all of its synapses.  In simplified SNN
models this means every outgoing synapse from a given neuron carries
the same sign:

- **Excitatory** neurons → all outgoing weights ≥ 0
- **Inhibitory** neurons → all outgoing weights ≤ 0

Implementation uses an absolute-value reparameterization::

    w_eff = |w_raw| × sign_mask

so that optimisers (gradient descent, R-STDP) operate on unconstrained
raw weights while the effective weights always satisfy the constraint.
Gradients flow through ``|·|`` via the ``sign()`` subgradient.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Any

__all__ = ["DaleSign", "DalesMask", "make_dale_mask", "apply_dales_constraint"]


class DaleSign(IntEnum):
    """Neuron type under Dale's Law."""

    EXCITATORY = 1
    INHIBITORY = -1


@dataclass(frozen=True, slots=True)
class DalesMask:
    """Per-neuron sign mask for Dale's Law enforcement.

    Attributes
    ----------
    signs:
        1-D tensor of ``+1.0`` (excitatory) / ``-1.0`` (inhibitory),
        one entry per neuron in the population.
    """

    signs: Any  # Tensor[N] of +1.0 / -1.0

    @property
    def n_excitatory(self) -> int:
        """Number of excitatory neurons."""
        return int((self.signs > 0).sum().item())

    @property
    def n_inhibitory(self) -> int:
        """Number of inhibitory neurons."""
        return int((self.signs < 0).sum().item())

    @property
    def size(self) -> int:
        """Total number of neurons in the population."""
        return int(self.signs.shape[0])


def make_dale_mask(
    n_excitatory: int,
    n_inhibitory: int,
    *,
    device: str = "cpu",
    dtype: str = "float64",
) -> DalesMask:
    """Create a Dale's Law mask (first *E* excitatory, then *I* inhibitory).

    Parameters
    ----------
    n_excitatory:
        Number of excitatory neurons (sign = +1).
    n_inhibitory:
        Number of inhibitory neurons (sign = −1).
    device:
        Torch device string.
    dtype:
        Torch dtype string.

    Returns
    -------
    DalesMask:
        Frozen mask with ``signs`` tensor of shape ``[E + I]``.

    Raises
    ------
    ValueError:
        If either count is negative.
    """
    if n_excitatory < 0 or n_inhibitory < 0:
        msg = (
            f"Neuron counts must be non-negative, "
            f"got n_excitatory={n_excitatory}, n_inhibitory={n_inhibitory}"
        )
        raise ValueError(msg)

    from neuroforge.core.torch_utils import require_torch, resolve_device_dtype

    torch = require_torch()
    dev, dt = resolve_device_dtype(device, dtype)
    signs = torch.ones(n_excitatory + n_inhibitory, device=dev, dtype=dt)
    signs[n_excitatory:] = -1.0
    return DalesMask(signs=signs)


def apply_dales_constraint(raw_weights: Any, signs: Any) -> Any:
    """Apply Dale's Law via ``|w| × sign`` reparameterization.

    Parameters
    ----------
    raw_weights:
        Unconstrained weight tensor (any shape).
    signs:
        Sign tensor (``+1.0`` / ``-1.0``), must be broadcastable
        with *raw_weights*.

    Returns
    -------
    Tensor:
        Effective weights ``|raw_weights| × signs`` that satisfy
        Dale's constraint.
    """
    from neuroforge.core.torch_utils import require_torch

    torch = require_torch()
    return torch.abs(raw_weights) * signs
