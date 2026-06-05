"""Simulation clock — tracks step index and simulation time."""

from __future__ import annotations

__all__ = ["SimulationClock"]


class SimulationClock:
    """Tracks the current simulation step and time.

    Parameters
    ----------
    dt:
        Time-step size in seconds.

    Example
    -------
    >>> clock = SimulationClock(dt=1e-3)
    >>> clock.advance()
    >>> clock.step  # 1
    >>> clock.t     # 0.001
    """

    __slots__ = ("_dt", "_step")

    def __init__(self, dt: float) -> None:
        if dt <= 0:
            msg = f"dt must be positive, got {dt}"
            raise ValueError(msg)
        self._dt = dt
        self._step = 0

    @property
    def dt(self) -> float:
        """Time-step size in seconds."""
        return self._dt

    @property
    def step(self) -> int:
        """Current step index (0-based)."""
        return self._step

    @property
    def t(self) -> float:
        """Current simulation time in seconds."""
        return self._step * self._dt

    def advance(self) -> None:
        """Advance the clock by one step."""
        self._step += 1

    def reset(self) -> None:
        """Reset the clock to step 0."""
        self._step = 0

    def __repr__(self) -> str:
        return f"SimulationClock(dt={self._dt}, step={self._step}, t={self.t:.6f})"
