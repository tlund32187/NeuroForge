"""Acceptance test: multi-seed stability harness regression guard."""

from __future__ import annotations

from typing import Any

import pytest

from neuroforge.runners.stability_harness import run_multi_seed


@pytest.mark.acceptance
def test_multi_seed_stability_harness() -> None:
    """Harness should keep a high pass rate with no critical instability flags."""
    summary: dict[str, Any] = run_multi_seed(
        task_name="multi_gate",
        seeds=[1, 2, 3, 4, 5],
        base_config={
            "device": "cpu",
            "max_epochs": 220,
            "n_hidden": 32,
            "n_inhibitory": 8,
            "lr": 2e-2,
            "stability_every": 5,
        },
        deterministic=True,
    )

    pass_rate = float(summary["pass_rate"])
    p90_converge = summary["p90_converge"]
    successful_flags = summary["successful_flag_counts"]

    assert pass_rate >= 0.8, f"pass_rate too low: {pass_rate:.3f}"
    assert p90_converge is not None, "Expected at least one successful convergence sample"
    assert float(p90_converge) <= 200.0, f"p90 convergence too high: {p90_converge}"
    assert int(successful_flags["stab_nan_inf"]) == 0
    assert int(successful_flags["stab_weight_explode"]) == 0
