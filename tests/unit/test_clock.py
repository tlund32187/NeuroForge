"""Tests for SimulationClock — advance, reset, time computation."""

from __future__ import annotations

import pytest

from neuroforge.core.time.clock import SimulationClock


@pytest.mark.unit
class TestSimulationClock:
    def test_initial_state(self) -> None:
        clock = SimulationClock(dt=1e-3)
        assert clock.step == 0
        assert clock.t == 0.0
        assert clock.dt == 1e-3

    def test_advance_increments_step(self) -> None:
        clock = SimulationClock(dt=1e-3)
        clock.advance()
        assert clock.step == 1

    def test_advance_increments_time(self) -> None:
        clock = SimulationClock(dt=0.01)
        clock.advance()
        assert abs(clock.t - 0.01) < 1e-12

    def test_multiple_advances(self) -> None:
        clock = SimulationClock(dt=1e-3)
        for _ in range(100):
            clock.advance()
        assert clock.step == 100
        assert abs(clock.t - 0.1) < 1e-10

    def test_reset_clears_step(self) -> None:
        clock = SimulationClock(dt=1e-3)
        for _ in range(50):
            clock.advance()
        clock.reset()
        assert clock.step == 0
        assert clock.t == 0.0

    def test_dt_preserved_after_reset(self) -> None:
        clock = SimulationClock(dt=0.005)
        clock.advance()
        clock.reset()
        assert clock.dt == 0.005

    def test_invalid_dt_raises(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            SimulationClock(dt=0.0)
        with pytest.raises(ValueError, match="positive"):
            SimulationClock(dt=-1.0)

    def test_repr(self) -> None:
        clock = SimulationClock(dt=1e-3)
        clock.advance()
        r = repr(clock)
        assert "step=1" in r
        assert "dt=0.001" in r
