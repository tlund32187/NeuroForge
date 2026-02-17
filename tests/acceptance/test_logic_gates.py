"""Acceptance tests: all 6 logic gates must converge.

This is the Phase 8 milestone — the SNN must learn every logic gate
using R-STDP.  Simple gates (AND, OR, NAND, NOR) are linearly separable.
XOR and XNOR require a hidden layer.

Budget:
    - Simple gates: ≤5000 trials
    - XOR/XNOR: ≤20000 trials
"""

from __future__ import annotations

import pytest
from neuroforge.tasks.logic_gates import GATE_TABLES, LogicGateConfig, LogicGateTask

# ── Parametrised test for simple gates ──────────────────────────────


@pytest.mark.acceptance
@pytest.mark.parametrize("gate", ["AND", "OR", "NAND", "NOR"])
def test_simple_gate(gate: str) -> None:
    """Simple gate must converge within 5000 trials."""
    config = LogicGateConfig(
        gate=gate,
        max_trials=5000,
        window_steps=50,
        dt=1e-3,
        convergence_streak=20,
        amplitude=50.0,
        spike_threshold=3,
        seed=42,
        lr=5e-3,
    )
    task = LogicGateTask(config)
    result = task.run()
    assert result.converged, (
        f"{gate} did not converge after {result.trials} trials. "
        f"Final accuracy: {result.accuracy_history[-1]:.2f}"
    )


@pytest.mark.acceptance
@pytest.mark.parametrize("gate", ["XOR", "XNOR"])
def test_complex_gate(gate: str) -> None:
    """XOR/XNOR must converge within 20000 trials."""
    config = LogicGateConfig(
        gate=gate,
        max_trials=20000,
        window_steps=50,
        dt=1e-3,
        convergence_streak=20,
        n_hidden=6,
        amplitude=50.0,
        spike_threshold=3,
        seed=42,
        lr=5e-3,
    )
    task = LogicGateTask(config)
    result = task.run()
    assert result.converged, (
        f"{gate} did not converge after {result.trials} trials. "
        f"Final accuracy: {result.accuracy_history[-1]:.2f}"
    )


# ── Truth table verification (non-training) ────────────────────────


class TestGateTables:
    """Verify truth tables are correct."""

    def test_and(self) -> None:
        assert GATE_TABLES["AND"] == {(0, 0): 0, (0, 1): 0, (1, 0): 0, (1, 1): 1}

    def test_or(self) -> None:
        assert GATE_TABLES["OR"] == {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 1}

    def test_nand(self) -> None:
        assert GATE_TABLES["NAND"] == {(0, 0): 1, (0, 1): 1, (1, 0): 1, (1, 1): 0}

    def test_nor(self) -> None:
        assert GATE_TABLES["NOR"] == {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 0}

    def test_xor(self) -> None:
        assert GATE_TABLES["XOR"] == {(0, 0): 0, (0, 1): 1, (1, 0): 1, (1, 1): 0}

    def test_xnor(self) -> None:
        assert GATE_TABLES["XNOR"] == {(0, 0): 1, (0, 1): 0, (1, 0): 0, (1, 1): 1}

    def test_all_gates_have_four_entries(self) -> None:
        for gate_name, table in GATE_TABLES.items():
            assert len(table) == 4, f"{gate_name} has {len(table)} entries"
