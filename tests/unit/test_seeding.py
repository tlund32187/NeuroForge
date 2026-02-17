"""Tests for seed management — same seed produces same results."""

from __future__ import annotations

import pytest


@pytest.mark.unit
class TestSeeding:
    def test_same_seed_same_randn(self) -> None:
        """Setting the same seed twice must produce identical random tensors."""
        from neuroforge.core.determinism.seeding import set_global_seed
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()

        set_global_seed(123)
        a = torch.randn(10)

        set_global_seed(123)
        b = torch.randn(10)

        assert torch.equal(a, b), "Same seed must produce identical tensors"

    def test_different_seed_different_randn(self) -> None:
        """Different seeds should (almost certainly) produce different tensors."""
        from neuroforge.core.determinism.seeding import set_global_seed
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()

        set_global_seed(1)
        a = torch.randn(100)

        set_global_seed(2)
        b = torch.randn(100)

        assert not torch.equal(a, b), "Different seeds should differ"

    def test_seed_is_reproducible_across_calls(self) -> None:
        """Multiple randn calls after seeding reproduce the full sequence."""
        from neuroforge.core.determinism.seeding import set_global_seed
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()

        set_global_seed(42)
        seq1 = [torch.randn(5) for _ in range(3)]

        set_global_seed(42)
        seq2 = [torch.randn(5) for _ in range(3)]

        for t1, t2 in zip(seq1, seq2, strict=True):
            assert torch.equal(t1, t2)
