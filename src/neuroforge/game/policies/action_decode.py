# pyright: basic
"""Decode per-button motor firing rates into a :class:`ControllerAction`.

Each NES button has its own channel (multi-label, not argmax), so combos like
Right+B+A emerge naturally. Exploration is built in (threshold / Bernoulli /
epsilon). The d-pad exclusivity the contract enforces (no Left+Right, no
Up+Down) is resolved here in favour of the stronger-firing direction before the
action is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from neuroforge.contracts.game import NINTENDO_BUTTONS, ControllerAction

__all__ = ["ActionDecodeConfig", "ActionDecoder"]

_FIELD: dict[str, str] = {
    "Up": "up", "Down": "down", "Left": "left", "Right": "right",
    "A": "a", "B": "b", "Start": "start", "Select": "select",
}
_VALID_MODES = frozenset({"threshold", "bernoulli", "epsilon"})
_UP, _DOWN, _LEFT, _RIGHT = 0, 1, 2, 3
_START, _SELECT = 6, 7


@dataclass(frozen=True, slots=True)
class ActionDecodeConfig:
    """Configuration for :class:`ActionDecoder`."""

    threshold: float = 0.35
    mode: str = "threshold"     # "threshold" | "bernoulli" | "epsilon"
    temperature: float = 0.1
    epsilon: float = 0.0
    allow_start: bool = False    # keep Start/Select masked during play by default
    allow_select: bool = False
    # Optional human-like button budget. After d-pad conflicts and Start/Select
    # masking, keep only the strongest N pressed buttons; useful for preventing
    # random early policies from spamming every button every frame.
    max_pressed: int | None = None
    # In the stochastic modes, resolve a Left/Right (or Up/Down) clash by sampling
    # the kept direction in proportion to firing rate, with this additive floor so
    # the weaker direction still gets explored. Without it a random-but-fixed bias
    # toward one direction is *never* escaped (the classic "only ever goes left"
    # trap), and the policy can never discover the reward for the other direction.
    dpad_explore_floor: float = 0.1

    def __post_init__(self) -> None:
        if self.mode not in _VALID_MODES:
            msg = f"ActionDecodeConfig.mode must be one of {sorted(_VALID_MODES)}"
            raise ValueError(msg)
        if self.temperature <= 0.0:
            msg = "ActionDecodeConfig.temperature must be > 0"
            raise ValueError(msg)
        if self.dpad_explore_floor < 0.0:
            msg = "ActionDecodeConfig.dpad_explore_floor must be >= 0"
            raise ValueError(msg)
        if self.max_pressed is not None and self.max_pressed < 1:
            msg = "ActionDecodeConfig.max_pressed must be >= 1 when set"
            raise ValueError(msg)


class ActionDecoder:
    """Turn an 8-vector of button firing rates into a controller action."""

    def __init__(self, config: ActionDecodeConfig | None = None, *, seed: int = 0) -> None:
        from neuroforge.core.torch_utils import require_torch

        self._cfg = config or ActionDecodeConfig()
        self._torch = require_torch()
        self._rng = self._torch.Generator(device="cpu")
        self._rng.manual_seed(int(seed))

    def decode(self, motor_rates: Any) -> ControllerAction:
        """Decode ``motor_rates`` ([n_buttons] tensor in [0, 1]) to an action."""
        torch = self._torch
        cfg = self._cfg
        rates = motor_rates.detach().to("cpu").reshape(-1)

        if cfg.mode == "threshold":
            pressed = rates > cfg.threshold
        elif cfg.mode == "bernoulli":
            prob = torch.sigmoid((rates - cfg.threshold) / cfg.temperature)
            pressed = torch.rand(rates.shape, generator=self._rng) < prob
        else:  # epsilon
            base = rates > cfg.threshold
            explore = torch.rand(rates.shape, generator=self._rng) < cfg.epsilon
            coin = torch.rand(rates.shape, generator=self._rng) < 0.5
            pressed = torch.where(explore, coin, base)

        bits = [bool(b) for b in pressed.tolist()]
        rate_list = [float(r) for r in rates.tolist()]

        if cfg.mode == "threshold":
            _resolve_conflict(bits, rate_list, _LEFT, _RIGHT)
            _resolve_conflict(bits, rate_list, _UP, _DOWN)
        else:
            self._resolve_conflict_stochastic(bits, rate_list, _LEFT, _RIGHT)
            self._resolve_conflict_stochastic(bits, rate_list, _UP, _DOWN)
        if not cfg.allow_start:
            bits[_START] = False
        if not cfg.allow_select:
            bits[_SELECT] = False
        _limit_pressed(bits, rate_list, cfg.max_pressed)

        kwargs = {_FIELD[name]: bits[i] for i, name in enumerate(NINTENDO_BUTTONS)}
        return ControllerAction(**kwargs)

    def _resolve_conflict_stochastic(
        self, bits: list[bool], rates: list[float], a: int, b: int,
    ) -> None:
        """If both *a* and *b* are pressed, keep one sampled by (floored) rate.

        Proportional sampling with a floor means the weaker direction is still
        chosen sometimes, so a fixed init bias toward one direction is escapable.
        """
        if not (bits[a] and bits[b]):
            return
        floor = self._cfg.dpad_explore_floor
        ra, rb = rates[a] + floor, rates[b] + floor
        total = ra + rb
        keep_a = total <= 0.0 or float(self._torch.rand((), generator=self._rng)) < ra / total
        bits[b if keep_a else a] = False


def _resolve_conflict(bits: list[bool], rates: list[float], a: int, b: int) -> None:
    """If both *a* and *b* are pressed, keep the one with the higher rate."""
    if bits[a] and bits[b]:
        if rates[a] >= rates[b]:
            bits[b] = False
        else:
            bits[a] = False


def _limit_pressed(bits: list[bool], rates: list[float], max_pressed: int | None) -> None:
    """Keep only the strongest pressed buttons when a button budget is configured."""
    if max_pressed is None:
        return
    pressed = [index for index, bit in enumerate(bits) if bit]
    if len(pressed) <= max_pressed:
        return
    keep = set(sorted(pressed, key=lambda index: rates[index], reverse=True)[:max_pressed])
    for index in pressed:
        bits[index] = index in keep
