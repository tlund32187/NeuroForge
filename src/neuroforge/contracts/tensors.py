"""Tensor type alias.

At type-check time this resolves to ``torch.Tensor`` so that IDEs and
type-checkers understand shapes/methods.  At runtime it falls back to
``Any`` so the contracts package never imports torch eagerly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from torch import Tensor as Tensor  # re-export for consumers
else:
    Tensor = Any

__all__ = ["Tensor"]
