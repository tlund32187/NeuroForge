# pyright: basic
"""Stateful real-time inference over a spiking policy network.

The central abstraction for live play: hold one frame's drive constant for a
short window of engine ticks and read out the motor layer's firing — but
crucially **without resetting between frames**, so membrane potentials and
recurrent activity persist across frames (short-horizon memory). This is what
the stateless batch vision backbone cannot do; here we step a ``CoreEngine``
directly.

``decide`` returns per-button firing rates in [0, 1]; an optional injected
noise current on the motor layer gives stochastic, R-STDP-friendly exploration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from neuroforge.contracts.types import Compartment

if TYPE_CHECKING:
    from collections.abc import Callable

    from neuroforge.engine.core_engine import CoreEngine

__all__ = ["CoreEnginePolicyEngine", "IStatefulPolicyEngine", "PolicyDecision"]


@dataclass(frozen=True, slots=True)
class PolicyDecision:
    """Result of one ``decide`` call."""

    motor_rates: Any  # Tensor [n_buttons], each in [0, 1]
    ticks: int


@runtime_checkable
class IStatefulPolicyEngine(Protocol):
    """A brain that turns a drive vector into per-button firing rates."""

    def decide(self, drive: Any, *, ticks: int) -> PolicyDecision:
        """Run *ticks* engine steps with *drive* held, return motor rates."""
        ...

    def reset(self) -> None:
        """Clear neural state (call at episode start)."""
        ...


class CoreEnginePolicyEngine:
    """Drive a :class:`CoreEngine` statefully and read out motor firing rates."""

    def __init__(
        self,
        engine: CoreEngine,
        *,
        motor_pop: str,
        motor_per_button: int,
        n_buttons: int,
        input_pop: str = "input",
        noise_amp: float = 0.0,
        seed: int = 0,
        on_tick: Callable[[], None] | None = None,
    ) -> None:
        from neuroforge.core.torch_utils import require_torch

        self._engine = engine
        self._motor_pop = motor_pop
        self._k = int(motor_per_button)
        self._n_buttons = int(n_buttons)
        self._input_pop = input_pop
        self._noise_amp = float(noise_amp)
        self._n_motor = self._k * self._n_buttons
        self._torch = require_torch()
        self._rng = self._torch.Generator(device="cpu")
        self._rng.manual_seed(int(seed))
        self._motor_population = engine.populations[motor_pop]
        self._on_tick = on_tick

    @property
    def engine(self) -> CoreEngine:
        return self._engine

    def set_on_tick(self, on_tick: Callable[[], None] | None) -> None:
        """Register a callback invoked after every engine tick within ``decide``.

        Used by the learning loop to advance eligibility traces per tick.
        """
        self._on_tick = on_tick

    def reset(self) -> None:
        self._engine.reset()

    def decide(self, drive: Any, *, ticks: int) -> PolicyDecision:
        if ticks < 1:
            msg = "ticks must be >= 1"
            raise ValueError(msg)
        torch = self._torch
        ref = self._motor_population.state["v"]
        accum = torch.zeros(self._n_motor, device=ref.device, dtype=ref.dtype)

        for _ in range(ticks):
            external: dict[str, dict[Compartment, Any]] = {
                self._input_pop: {Compartment.SOMA: drive},
            }
            if self._noise_amp > 0.0:
                noise = torch.randn(
                    self._n_motor, generator=self._rng, dtype=ref.dtype,
                ) * self._noise_amp
                external[self._motor_pop] = {Compartment.SOMA: noise.to(ref.device)}
            result = self._engine.step(external)
            accum = accum + result.spikes[self._motor_pop].to(ref.dtype)
            if self._on_tick is not None:
                self._on_tick()

        rates = accum.reshape(self._n_buttons, self._k).mean(dim=1) / float(ticks)
        return PolicyDecision(motor_rates=rates, ticks=ticks)
