"""Vision-only game environment contracts.

These DTOs and protocols define the seam between NeuroForge brains and game
emulators such as BizHawk. The boundary intentionally carries screen pixels,
controller actions, and frame-derived metrics only; emulator RAM/state is not
part of the public contract.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

__all__ = [
    "ControllerAction",
    "EpisodeDecision",
    "GameClientStep",
    "GameObservation",
    "GameTransition",
    "IEpisodeManager",
    "IFrameMetricExtractor",
    "IGameClient",
    "IRewardModel",
    "IVisionGamePolicy",
    "NINTENDO_BUTTONS",
    "ScreenFrame",
    "VisionGameMetrics",
]

NINTENDO_BUTTONS: tuple[str, ...] = (
    "Up",
    "Down",
    "Left",
    "Right",
    "A",
    "B",
    "Start",
    "Select",
)


@dataclass(frozen=True, slots=True)
class ControllerAction:
    """One Nintendo-style controller action.

    The button names match BizHawk Lua ``joypad.set`` keys.
    """

    up: bool = False
    down: bool = False
    left: bool = False
    right: bool = False
    a: bool = False
    b: bool = False
    start: bool = False
    select: bool = False

    def __post_init__(self) -> None:
        if self.left and self.right:
            msg = "ControllerAction cannot press Left and Right together"
            raise ValueError(msg)
        if self.up and self.down:
            msg = "ControllerAction cannot press Up and Down together"
            raise ValueError(msg)

    def pressed(self) -> tuple[str, ...]:
        """Return the pressed buttons using BizHawk-compatible names."""
        out: list[str] = []
        for attr, name in (
            ("up", "Up"),
            ("down", "Down"),
            ("left", "Left"),
            ("right", "Right"),
            ("a", "A"),
            ("b", "B"),
            ("start", "Start"),
            ("select", "Select"),
        ):
            if getattr(self, attr):
                out.append(name)
        return tuple(out)

    def to_bizhawk(self) -> dict[str, bool]:
        """Return a sparse mapping suitable for BizHawk ``joypad.set``."""
        return {name: True for name in self.pressed()}

    def as_dense_tuple(self) -> tuple[bool, ...]:
        """Return button states in :data:`NINTENDO_BUTTONS` order."""
        pressed = set(self.pressed())
        return tuple(name in pressed for name in NINTENDO_BUTTONS)


@dataclass(frozen=True, slots=True)
class ScreenFrame:
    """A raw screen capture frame.

    ``data`` is row-major bytes with ``width * height * channels`` entries.
    Supported channel counts are 1 (grayscale), 3 (RGB), and 4 (RGBA).
    """

    width: int
    height: int
    channels: int
    data: bytes
    frame_id: int = 0
    t: float = 0.0

    def __post_init__(self) -> None:
        if self.width <= 0:
            msg = "ScreenFrame.width must be > 0"
            raise ValueError(msg)
        if self.height <= 0:
            msg = "ScreenFrame.height must be > 0"
            raise ValueError(msg)
        if self.channels not in {1, 3, 4}:
            msg = "ScreenFrame.channels must be one of {1, 3, 4}"
            raise ValueError(msg)

        data = bytes(self.data)
        expected = self.width * self.height * self.channels
        if len(data) != expected:
            msg = f"ScreenFrame.data must contain {expected} bytes; got {len(data)}"
            raise ValueError(msg)
        object.__setattr__(self, "data", data)

    @property
    def n_bytes(self) -> int:
        """Return the frame payload size in bytes."""
        return len(self.data)

    def as_memoryview(self) -> memoryview:
        """Return a zero-copy view over the immutable frame bytes."""
        return memoryview(self.data)


def _empty_confidence() -> dict[str, float]:
    """Typed default factory for :attr:`VisionGameMetrics.confidence`."""
    return {}


@dataclass(frozen=True, slots=True)
class VisionGameMetrics:
    """Metrics inferred from screen pixels.

    Fields default to ``None`` when an extractor cannot infer a value from the
    current frame. ``confidence`` maps metric names to ``[0, 1]`` confidences.
    """

    score: int | None = None
    lives: int | None = None
    world: str | None = None
    level: str | None = None
    time_left: int | None = None
    x_progress: float | None = None
    player_state: str | None = None
    confidence: dict[str, float] = field(default_factory=_empty_confidence)

    def __post_init__(self) -> None:
        copied = dict(self.confidence)
        for key, value in copied.items():
            if not (0.0 <= float(value) <= 1.0):
                msg = f"VisionGameMetrics.confidence[{key!r}] must be in [0, 1]"
                raise ValueError(msg)
            copied[key] = float(value)
        object.__setattr__(self, "confidence", copied)


@dataclass(frozen=True, slots=True)
class GameObservation:
    """What the brain sees at one emulator step."""

    step: int
    t: float
    frame: ScreenFrame
    metrics: VisionGameMetrics = field(default_factory=VisionGameMetrics)

    def with_metrics(self, metrics: VisionGameMetrics) -> GameObservation:
        """Return this observation with new frame-derived metrics."""
        return GameObservation(step=self.step, t=self.t, frame=self.frame, metrics=metrics)


@dataclass(frozen=True, slots=True)
class GameClientStep:
    """Raw step result from an emulator client."""

    observation: GameObservation
    terminated: bool = False
    truncated: bool = False

    @property
    def done(self) -> bool:
        """Return whether the episode should stop."""
        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class GameTransition:
    """A policy action and the resulting frame-only transition."""

    before: GameObservation
    action: ControllerAction
    after: GameObservation
    reward: float
    terminated: bool = False
    truncated: bool = False
    termination_reason: str | None = None

    @property
    def done(self) -> bool:
        """Return whether the episode should stop."""
        return self.terminated or self.truncated


@dataclass(frozen=True, slots=True)
class EpisodeDecision:
    """An episode manager's verdict on whether an episode should end.

    Termination is inferred from vision-derived metric deltas (e.g. a life lost,
    a level cleared, or a prolonged stall) rather than from emulator RAM, so it
    stays within the vision-only contract.
    """

    terminated: bool = False
    reason: str | None = None


@runtime_checkable
class IGameClient(Protocol):
    """Frame-only emulator client protocol."""

    def reset(self) -> GameObservation:
        """Reset the emulator and return the first screen observation."""
        ...

    def step(self, action: ControllerAction) -> GameClientStep:
        """Apply a controller action and return the next screen observation."""
        ...

    def close(self) -> None:
        """Release emulator/client resources."""
        ...


@runtime_checkable
class IFrameMetricExtractor(Protocol):
    """Extract HUD/game metrics from screen pixels only."""

    def extract(self, frame: ScreenFrame) -> VisionGameMetrics:
        """Infer metrics from a single screen frame."""
        ...


@runtime_checkable
class IRewardModel(Protocol):
    """Compute reward from frame-derived observations."""

    def reward(self, previous: GameObservation, current: GameObservation) -> float:
        """Return the scalar reward for a transition."""
        ...


@runtime_checkable
class IVisionGamePolicy(Protocol):
    """Policy interface for a vision-only game-playing brain."""

    def act(self, observation: GameObservation) -> ControllerAction:
        """Choose a controller action from the current observation."""
        ...


@runtime_checkable
class IEpisodeManager(Protocol):
    """Decide, from frame-derived observations only, when an episode ends."""

    def should_end(
        self, before: GameObservation, after: GameObservation,
    ) -> EpisodeDecision:
        """Return whether the episode should terminate after this transition."""
        ...
