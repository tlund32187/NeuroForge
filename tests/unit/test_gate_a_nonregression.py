"""Non-regression: existing Gate A (single-gate LogicGateTask) still works.

Runs a short training pass on a simple gate to confirm the original
single-gate SNN path is unaffected by the vision pipeline additions.
"""

from __future__ import annotations

import pytest

from neuroforge.tasks.logic_gates import LogicGateConfig, LogicGateTask


@pytest.mark.parametrize("gate", ["AND", "OR"])
def test_gate_a_still_runs(gate: str) -> None:
    """Single-gate task completes without errors (not testing convergence)."""
    config = LogicGateConfig(
        gate=gate,
        max_trials=50,
        window_steps=10,
        dt=1e-3,
        convergence_streak=5,
        seed=42,
        lr=5e-3,
    )
    task = LogicGateTask(config)
    result = task.run()
    assert result.trials > 0
    assert len(result.accuracy_history) > 0


def test_multi_gate_still_runs() -> None:
    """Multi-gate task completes without errors (not testing convergence)."""
    from neuroforge.tasks.multi_gate import MultiGateConfig, MultiGateTask

    config = MultiGateConfig(
        max_epochs=1,
        seed=42,
        device="cpu",
    )
    task = MultiGateTask(config)
    result = task.run()
    assert result.trials > 0
    assert result.epochs >= 1
