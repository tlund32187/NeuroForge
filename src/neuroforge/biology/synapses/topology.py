"""Synapse topology DTOs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from neuroforge.contracts.tensors import Tensor

__all__ = ["SynapseTopology"]


@dataclass(frozen=True, slots=True)
class SynapseTopology:
    """Immutable description of a projection's wiring."""

    pre_idx: Tensor
    post_idx: Tensor
    weights: Tensor
    delays: Tensor
    n_pre: int
    n_post: int
