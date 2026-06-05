"""Deterministic in-memory game clients (no emulator required).

These are first-class :class:`IGameClient` implementations — not test doubles —
used to develop and exercise the loop, policies, and the learning task before a
real emulator is wired in, and to keep CI fully deterministic and offline.

* :class:`ScriptedGameClient` synthesises frames procedurally.
* :class:`ReplayGameClient` replays pre-recorded :class:`ScreenFrame`s.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from neuroforge.contracts.applications.games import (
    GameClientStep,
    GameObservation,
    ScreenFrame,
    VisionGameMetrics,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from neuroforge.contracts.applications.games import ControllerAction

__all__ = ["ActionProgressGameClient", "ReplayGameClient", "ScriptedGameClient"]


def _default_frame_bytes(step: int, n: int) -> bytes:
    """A cheap, deterministic, step-varying pixel pattern."""
    base = step & 0xFF
    return bytes((base + i) & 0xFF for i in range(n))


class ScriptedGameClient:
    """Synthesise frames procedurally; record actions for assertions."""

    def __init__(
        self,
        *,
        width: int = 84,
        height: int = 84,
        channels: int = 1,
        max_steps: int = 0,
        terminate_at: int = 0,
        frame_factory: Callable[[int], bytes] | None = None,
        fps: float = 60.0,
    ) -> None:
        self._w = int(width)
        self._h = int(height)
        self._c = int(channels)
        self._n = self._w * self._h * self._c
        self._max_steps = int(max_steps)
        self._terminate_at = int(terminate_at)
        self._frame_factory = frame_factory
        self._fps = float(fps)
        self._step_count = 0
        self.actions: list[ControllerAction] = []
        self.closed = False

    def _frame(self, step: int) -> ScreenFrame:
        data = (
            self._frame_factory(step)
            if self._frame_factory is not None
            else _default_frame_bytes(step, self._n)
        )
        return ScreenFrame(
            width=self._w,
            height=self._h,
            channels=self._c,
            data=data,
            frame_id=step,
            t=step / self._fps,
        )

    def _observation(self, step: int) -> GameObservation:
        frame = self._frame(step)
        return GameObservation(step=step, t=frame.t, frame=frame)

    def reset(self) -> GameObservation:
        self._step_count = 0
        return self._observation(0)

    def step(self, action: ControllerAction) -> GameClientStep:
        self.actions.append(action)
        self._step_count += 1
        terminated = self._terminate_at > 0 and self._step_count >= self._terminate_at
        truncated = self._max_steps > 0 and self._step_count >= self._max_steps
        return GameClientStep(
            self._observation(self._step_count),
            terminated=terminated,
            truncated=truncated,
        )

    def close(self) -> None:
        self.closed = True


class ReplayGameClient:
    """Replay a fixed sequence of recorded frames."""

    def __init__(self, frames: Sequence[ScreenFrame], *, loop: bool = False) -> None:
        if len(frames) == 0:
            msg = "ReplayGameClient requires at least one frame"
            raise ValueError(msg)
        self._frames = list(frames)
        self._loop = bool(loop)
        self._idx = 0
        self.actions: list[ControllerAction] = []
        self.closed = False

    def _observation(self, idx: int) -> GameObservation:
        frame = self._frames[idx]
        return GameObservation(step=idx, t=frame.t, frame=frame)

    def reset(self) -> GameObservation:
        self._idx = 0
        return self._observation(0)

    def step(self, action: ControllerAction) -> GameClientStep:
        self.actions.append(action)
        last = len(self._frames) - 1
        if self._idx >= last:
            if self._loop:
                self._idx = 0
                return GameClientStep(self._observation(self._idx), truncated=False)
            return GameClientStep(self._observation(last), truncated=True)
        self._idx += 1
        return GameClientStep(
            self._observation(self._idx),
            truncated=self._idx >= last and not self._loop,
        )

    def close(self) -> None:
        self.closed = True


class ActionProgressGameClient:
    """Tiny action-dependent side-scroller proxy.

    The client is deterministic, frame-only, and cheap enough for CI. It is not a
    gameplay emulator; it exists to exercise policy/evaluator wiring with reward
    that depends on controller actions before spending time in BizHawk.
    """

    def __init__(
        self,
        *,
        width: int = 32,
        height: int = 28,
        channels: int = 1,
        max_steps: int = 300,
        fps: float = 60.0,
        acceleration: float = 0.004,
        friction: float = 0.88,
        run_multiplier: float = 1.6,
        jump_bonus: float = 0.001,
    ) -> None:
        self._w = int(width)
        self._h = int(height)
        self._c = int(channels)
        if self._w <= 0 or self._h <= 0:
            msg = "ActionProgressGameClient width/height must be > 0"
            raise ValueError(msg)
        if self._c not in {1, 3, 4}:
            msg = "ActionProgressGameClient channels must be one of {1, 3, 4}"
            raise ValueError(msg)
        if max_steps < 1:
            msg = "ActionProgressGameClient max_steps must be >= 1"
            raise ValueError(msg)
        if not 0.0 <= friction < 1.0:
            msg = "ActionProgressGameClient friction must be in [0, 1)"
            raise ValueError(msg)
        self._max_steps = int(max_steps)
        self._fps = float(fps)
        self._acceleration = float(acceleration)
        self._friction = float(friction)
        self._run_multiplier = float(run_multiplier)
        self._jump_bonus = float(jump_bonus)
        self._step_count = 0
        self._x = 0.0
        self._velocity = 0.0
        self._pose_y = 0
        self.actions: list[ControllerAction] = []
        self.closed = False

    def reset(self) -> GameObservation:
        self._step_count = 0
        self._x = 0.0
        self._velocity = 0.0
        self._pose_y = 0
        self.actions.clear()
        return self._observation()

    def step(self, action: ControllerAction) -> GameClientStep:
        self.actions.append(action)
        self._step_count += 1
        direction = int(action.right) - int(action.left)
        vertical = int(action.up) - int(action.down)
        run_scale = self._run_multiplier if action.b else 1.0
        self._velocity = self._velocity * self._friction + (
            direction * self._acceleration * run_scale
        )
        if (action.a or action.up) and action.right:
            self._velocity += self._jump_bonus
        if action.down:
            self._velocity *= 0.72
        self._pose_y = max(-2, min(2, self._pose_y - vertical))
        self._velocity = max(-0.025, min(0.035, self._velocity))
        self._x = max(0.0, min(1.0, self._x + self._velocity))
        terminated = self._x >= 1.0
        truncated = self._step_count >= self._max_steps
        return GameClientStep(
            self._observation(),
            terminated=terminated,
            truncated=truncated and not terminated,
        )

    def close(self) -> None:
        self.closed = True

    def _observation(self) -> GameObservation:
        frame = ScreenFrame(
            width=self._w,
            height=self._h,
            channels=self._c,
            data=self._frame_bytes(),
            frame_id=self._step_count,
            t=self._step_count / self._fps,
        )
        metrics = VisionGameMetrics(
            score=int(self._x * 1000.0),
            lives=3,
            x_progress=self._x,
            confidence={"score": 1.0, "lives": 1.0, "x_progress": 1.0},
        )
        return GameObservation(
            step=self._step_count,
            t=frame.t,
            frame=frame,
            metrics=metrics,
        )

    def _frame_bytes(self) -> bytes:
        data = bytearray(self._w * self._h * self._c)
        mario_x = min(self._w - 1, max(0, int(round(self._x * (self._w - 1)))))
        ground_y = self._h - 2
        mario_y = max(0, min(ground_y, ground_y - 1 + self._pose_y))
        for x in range(self._w):
            self._set_pixel(data, x, ground_y, 80)
        self._set_pixel(data, mario_x, mario_y, 255)
        self._set_pixel(data, mario_x, ground_y, 220)
        return bytes(data)

    def _set_pixel(self, data: bytearray, x: int, y: int, value: int) -> None:
        base = (y * self._w + x) * self._c
        if self._c == 1:
            data[base] = value
            return
        data[base] = value
        data[base + 1] = min(255, value + 20)
        data[base + 2] = max(0, value - 20)
        if self._c == 4:
            data[base + 3] = 255
