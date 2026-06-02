"""Tests for vision-only game contracts."""

from __future__ import annotations

import pytest

from neuroforge.contracts.game import (
    ControllerAction,
    GameClientStep,
    GameObservation,
    IFrameMetricExtractor,
    IGameClient,
    IRewardModel,
    IVisionGamePolicy,
    ScreenFrame,
    VisionGameMetrics,
)


def _frame(value: int = 0, *, frame_id: int = 0) -> ScreenFrame:
    return ScreenFrame(width=2, height=2, channels=1, data=bytes([value] * 4), frame_id=frame_id)


@pytest.mark.unit
def test_screen_frame_validates_payload_size_and_copies_bytes() -> None:
    raw = bytearray([1, 2, 3, 4])
    frame = ScreenFrame(width=2, height=2, channels=1, data=raw)  # type: ignore[arg-type]

    raw[0] = 99

    assert frame.data == b"\x01\x02\x03\x04"
    assert frame.n_bytes == 4
    assert frame.as_memoryview()[0] == 1


@pytest.mark.unit
def test_screen_frame_rejects_non_screen_shaped_data() -> None:
    with pytest.raises(ValueError, match="must contain 12 bytes"):
        ScreenFrame(width=2, height=2, channels=3, data=b"short")

    with pytest.raises(ValueError, match="channels"):
        ScreenFrame(width=2, height=2, channels=2, data=b"\x00" * 8)


@pytest.mark.unit
def test_controller_action_exports_bizhawk_buttons() -> None:
    action = ControllerAction(right=True, a=True, b=True)

    assert action.pressed() == ("Right", "A", "B")
    assert action.to_bizhawk() == {"Right": True, "A": True, "B": True}
    assert action.as_dense_tuple() == (
        False,
        False,
        False,
        True,
        True,
        True,
        False,
        False,
    )


@pytest.mark.unit
def test_controller_action_rejects_impossible_dpad_pairs() -> None:
    with pytest.raises(ValueError, match="Left and Right"):
        ControllerAction(left=True, right=True)
    with pytest.raises(ValueError, match="Up and Down"):
        ControllerAction(up=True, down=True)


@pytest.mark.unit
def test_observation_replaces_metrics_without_touching_frame() -> None:
    observation = GameObservation(step=7, t=0.14, frame=_frame(frame_id=7))
    metrics = VisionGameMetrics(score=100, confidence={"score": 0.9})

    updated = observation.with_metrics(metrics)

    assert updated.frame is observation.frame
    assert updated.metrics.score == 100
    assert updated.metrics.confidence == {"score": 0.9}


@pytest.mark.unit
def test_metric_confidence_is_bounded() -> None:
    with pytest.raises(ValueError, match="confidence"):
        VisionGameMetrics(confidence={"score": 1.2})


class _Client:
    def reset(self) -> GameObservation:
        return GameObservation(step=0, t=0.0, frame=_frame())

    def step(self, action: ControllerAction) -> GameClientStep:
        return GameClientStep(GameObservation(step=1, t=1.0 / 60.0, frame=_frame(1)))

    def close(self) -> None:
        return None


class _Extractor:
    def extract(self, frame: ScreenFrame) -> VisionGameMetrics:
        return VisionGameMetrics(score=frame.data[0])


class _Reward:
    def reward(self, previous: GameObservation, current: GameObservation) -> float:
        return float((current.metrics.score or 0) - (previous.metrics.score or 0))


class _Policy:
    def act(self, observation: GameObservation) -> ControllerAction:
        return ControllerAction(right=True)


@pytest.mark.unit
def test_game_protocols_are_runtime_checkable() -> None:
    assert isinstance(_Client(), IGameClient)
    assert isinstance(_Extractor(), IFrameMetricExtractor)
    assert isinstance(_Reward(), IRewardModel)
    assert isinstance(_Policy(), IVisionGamePolicy)
