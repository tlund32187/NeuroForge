"""Temporal action commitment: hold a chosen direction to build run momentum.

The action-side analog of A2's trace rule (temporal smoothing of behaviour). A
per-frame stochastic policy *dithers* — sampling Left one frame and Right the
next nets ~zero displacement ("legs moving but runs in place"). Committing the
d-pad **direction** for a short window lets a side-scroller build run speed and
actually traverse, while jump / run / vertical buttons still respond every frame
(so it can react and jump while committed to a heading).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from neuroforge.contracts.applications.games import ControllerAction

__all__ = ["IActionCommitment", "TemporalCommitment", "TemporalCommitmentConfig"]


@runtime_checkable
class IActionCommitment(Protocol):
    """Maintains committed actions across repeated environment frames."""

    def apply(self, action: ControllerAction) -> ControllerAction:
        """Return the action to apply now."""
        ...


@dataclass(frozen=True, slots=True)
class TemporalCommitmentConfig:
    """Configuration for :class:`TemporalCommitment`."""

    commit_frames: int = 8   # frames to hold a chosen intent
    commit_all_buttons: bool = False

    def __post_init__(self) -> None:
        if self.commit_frames < 1:
            msg = "TemporalCommitmentConfig.commit_frames must be >= 1"
            raise ValueError(msg)


class TemporalCommitment:
    """Hold the decoded Left/Right heading for a window; pass other buttons through."""

    def __init__(self, config: TemporalCommitmentConfig | None = None) -> None:
        self._cfg = config or TemporalCommitmentConfig()
        self._direction = 0   # -1 = Left, +1 = Right, 0 = none
        self._frames_left = 0
        self._action: ControllerAction | None = None

    def reset(self) -> None:
        """Clear the active commitment (call at episode start)."""
        self._direction = 0
        self._frames_left = 0
        self._action = None

    def apply(self, action: ControllerAction) -> ControllerAction:
        """Return *action* with its horizontal heading held to the commitment."""
        if self._cfg.commit_all_buttons:
            return self._apply_full_action(action)

        if self._frames_left <= 0:
            self._direction = -1 if action.left else (1 if action.right else 0)
            self._frames_left = self._cfg.commit_frames if self._direction != 0 else 0

        if self._direction == 0:
            left, right = action.left, action.right
        else:
            left, right = self._direction < 0, self._direction > 0

        if self._frames_left > 0:
            self._frames_left -= 1

        return ControllerAction(
            up=action.up,
            down=action.down,
            left=left,
            right=right,
            a=action.a,
            b=action.b,
            start=action.start,
            select=action.select,
        )

    def _apply_full_action(self, action: ControllerAction) -> ControllerAction:
        """Hold the full decoded action for the configured commitment window."""
        if self._frames_left <= 0 or self._action is None:
            self._action = action
            self._frames_left = self._cfg.commit_frames
        committed = self._action
        self._frames_left -= 1
        return committed
