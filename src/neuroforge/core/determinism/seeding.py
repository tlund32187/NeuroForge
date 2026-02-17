"""Seed management for reproducibility.

Sets seeds for torch, CUDA, and Python's random module to ensure
deterministic behaviour across runs.
"""

from __future__ import annotations

__all__ = ["set_global_seed"]


def set_global_seed(seed: int) -> None:
    """Set the global random seed for reproducibility.

    Sets seeds for:
    - ``torch.manual_seed``
    - ``torch.cuda.manual_seed_all`` (if CUDA available)
    - Python's ``random.seed``

    Parameters
    ----------
    seed:
        Integer seed value.
    """
    import random

    from neuroforge.core.torch_utils import require_torch

    random.seed(seed)

    torch = require_torch()
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
