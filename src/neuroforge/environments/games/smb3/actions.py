"""Human-like controller energy for game training.

The model is intentionally small and stateful: it discourages frantic,
frame-by-frame button churn without forbidding exploration. Holding a useful
action has a small ongoing cost; changing many buttons at once is more costly;
score/progress can refill the bucket, echoing the "metabolism" idea without
turning it into a separate game objective.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from neuroforge.contracts.applications.games import NINTENDO_BUTTONS

if TYPE_CHECKING:
    from neuroforge.contracts.applications.games import ControllerAction, GameObservation

SMB3_ACTION_BUTTONS: tuple[str, ...] = NINTENDO_BUTTONS

__all__ = [
    "ActionEnergyConfig",
    "ActionEnergyModel",
    "ActionEnergyStep",
    "SMB3_ACTION_BUTTONS",
]


@dataclass(frozen=True, slots=True)
class ActionEnergyConfig:
    """Configuration for :class:`ActionEnergyModel`."""

    enabled: bool = False
    capacity: float = 10.0
    recover_per_frame: float = 0.035
    button_cost: float = 0.020
    change_cost: float = 0.060
    cost_penalty_scale: float = 0.20
    shortage_penalty_scale: float = 1.00
    progress_refill_scale: float = 5.0
    score_refill_scale: float = 0.001
    count_start_select: bool = False

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            msg = "ActionEnergyConfig.capacity must be > 0"
            raise ValueError(msg)
        for name in (
            "recover_per_frame",
            "button_cost",
            "change_cost",
            "cost_penalty_scale",
            "shortage_penalty_scale",
            "progress_refill_scale",
            "score_refill_scale",
        ):
            if float(getattr(self, name)) < 0.0:
                msg = f"ActionEnergyConfig.{name} must be >= 0"
                raise ValueError(msg)


@dataclass(frozen=True, slots=True)
class ActionEnergyStep:
    """One energy update and reward adjustment."""

    reward_delta: float
    energy: float
    cost: float
    shortage: float
    button_count: int
    change_count: int
    refill: float


class ActionEnergyModel:
    """Track a small action-energy bucket and return reward adjustments."""

    def __init__(self, config: ActionEnergyConfig | None = None) -> None:
        self._cfg = config or ActionEnergyConfig()
        self._energy = float(self._cfg.capacity)
        self._previous: tuple[bool, ...] | None = None

    def begin_episode(self) -> None:
        """Restore energy and forget previous action state."""
        self._energy = float(self._cfg.capacity)
        self._previous = None

    def score(
        self,
        action: ControllerAction,
        previous: GameObservation,
        current: GameObservation,
    ) -> ActionEnergyStep:
        """Apply one action cost/recovery update."""
        cfg = self._cfg
        bits = _effective_bits(action, count_start_select=cfg.count_start_select)
        button_count = sum(1 for bit in bits if bit)
        change_count = 0
        if self._previous is not None:
            change_count = sum(
                1
                for old, new in zip(self._previous, bits, strict=True)
                if old != new
            )
        self._previous = bits

        refill = self._refill(previous, current)
        self._energy = min(cfg.capacity, self._energy + refill)

        cost = button_count * cfg.button_cost + change_count * cfg.change_cost
        shortage = max(0.0, cost - self._energy)
        self._energy = max(0.0, self._energy - cost)
        reward_delta = -(
            cost * cfg.cost_penalty_scale
            + shortage * cfg.shortage_penalty_scale
        )
        return ActionEnergyStep(
            reward_delta=float(reward_delta),
            energy=float(self._energy),
            cost=float(cost),
            shortage=float(shortage),
            button_count=button_count,
            change_count=change_count,
            refill=float(refill),
        )

    def _refill(self, previous: GameObservation, current: GameObservation) -> float:
        cfg = self._cfg
        progress = _positive_delta(previous.metrics.x_progress, current.metrics.x_progress)
        score = _positive_delta(previous.metrics.score, current.metrics.score)
        return (
            cfg.recover_per_frame
            + progress * cfg.progress_refill_scale
            + score * cfg.score_refill_scale
        )


def _effective_bits(action: ControllerAction, *, count_start_select: bool) -> tuple[bool, ...]:
    bits = action.as_dense_tuple()
    if count_start_select:
        return bits
    return bits[: NINTENDO_BUTTONS.index("Start")]


def _positive_delta(
    previous: int | float | None,
    current: int | float | None,
) -> float:
    if previous is None or current is None:
        return 0.0
    return max(0.0, float(current) - float(previous))
