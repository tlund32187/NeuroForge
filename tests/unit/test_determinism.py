"""Tests for deterministic-mode entrypoint."""

from __future__ import annotations

import pytest

from neuroforge.core.determinism.mode import DeterminismConfig, apply_determinism
from neuroforge.core.torch_utils import require_torch


def test_torch_repeatability() -> None:
    """Applying the same seed twice should reproduce torch random draws."""
    torch = require_torch()

    apply_determinism(DeterminismConfig(seed=1234, deterministic=True, warn_only=True))
    a = torch.randn(8)

    apply_determinism(DeterminismConfig(seed=1234, deterministic=True, warn_only=True))
    b = torch.randn(8)

    assert torch.equal(a, b)


def test_numpy_seed_if_available() -> None:
    """NumPy random sequence should be reproducible when NumPy is installed."""
    np = pytest.importorskip("numpy")

    apply_determinism(DeterminismConfig(seed=999, deterministic=False))
    a = np.random.randn(5)

    apply_determinism(DeterminismConfig(seed=999, deterministic=False))
    b = np.random.randn(5)

    assert (a == b).all()


def test_cuda_repeatability_if_available() -> None:
    """CUDA RNG should be reproducible when CUDA is available."""
    torch = require_torch()
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    apply_determinism(DeterminismConfig(seed=17, deterministic=True, warn_only=True))
    a = torch.randn(6, device="cuda")

    apply_determinism(DeterminismConfig(seed=17, deterministic=True, warn_only=True))
    b = torch.randn(6, device="cuda")

    assert torch.equal(a, b)
