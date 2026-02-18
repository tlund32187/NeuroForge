"""Acceptance tests: multi-gate task (all 6 gates, one brain).

The SNN must learn AND, OR, NAND, NOR, XOR, XNOR simultaneously
using fully shared weights, trained with surrogate gradients + Adam
+ Dale's Law.

Budget: ≤1500 epochs (36 000 trials) per seed.
"""

from __future__ import annotations

import pytest

from neuroforge.tasks.multi_gate import ALL_GATES, MultiGateConfig, MultiGateTask


@pytest.mark.acceptance
@pytest.mark.slow
def test_multi_gate_default_seed() -> None:
    """All 6 gates must converge with default config (seed=42)."""
    task = MultiGateTask(MultiGateConfig())
    result = task.run()
    assert result.converged, (
        f"Multi-gate did not converge after {result.trials} trials "
        f"({result.epochs} epochs). "
        f"Per-gate: {result.per_gate_converged}"
    )
    for gate in ALL_GATES:
        assert result.per_gate_converged[gate], f"{gate} did not converge"


@pytest.mark.acceptance
@pytest.mark.slow
@pytest.mark.parametrize("seed", [123, 7, 999])
def test_multi_gate_seeds(seed: int) -> None:
    """Multi-gate must converge across several seeds."""
    cfg = MultiGateConfig(seed=seed)
    task = MultiGateTask(cfg)
    result = task.run()
    assert result.converged, (
        f"seed={seed}: not converged after {result.trials} trials "
        f"({result.epochs} epochs). "
        f"Per-gate: {result.per_gate_converged}"
    )


@pytest.mark.acceptance
def test_multi_gate_result_structure() -> None:
    """Result dataclass has correct fields and types."""
    cfg = MultiGateConfig(max_epochs=5)  # just a few epochs
    task = MultiGateTask(cfg)
    result = task.run()

    assert result.gates == ALL_GATES
    assert isinstance(result.trials, int)
    assert isinstance(result.epochs, int)
    assert isinstance(result.per_gate_converged, dict)
    assert set(result.per_gate_converged.keys()) == set(ALL_GATES)
    assert isinstance(result.final_weights, dict)
    assert "w_ih" in result.final_weights
    assert "w_ho" in result.final_weights
    assert "b_h" in result.final_weights
    assert "b_o" in result.final_weights
    assert isinstance(result.accuracy_history, list)
