"""Tests for the vision-only game episode loop."""

from __future__ import annotations

import pytest

from neuroforge.contracts.game import (
    ControllerAction,
    EpisodeDecision,
    GameClientStep,
    GameObservation,
    ScreenFrame,
    VisionGameMetrics,
)
from neuroforge.game.clients.scripted import ScriptedGameClient
from neuroforge.game.loop import VisionOnlyGameLoop
from neuroforge.game.rewards import VisionMetricRewardConfig, VisionMetricRewardModel


def _observation(step: int, value: int, *, lives: int = 3) -> GameObservation:
    frame = ScreenFrame(
        width=1,
        height=1,
        channels=1,
        data=bytes([value]),
        frame_id=step,
        t=step / 60.0,
    )
    return GameObservation(
        step=step,
        t=step / 60.0,
        frame=frame,
        metrics=VisionGameMetrics(lives=lives),
    )


class _ScriptedClient:
    def __init__(self) -> None:
        self.actions: list[ControllerAction] = []
        self._step = 0
        self.closed = False

    def reset(self) -> GameObservation:
        self._step = 0
        return _observation(0, 10)

    def step(self, action: ControllerAction) -> GameClientStep:
        self.actions.append(action)
        self._step += 1
        return GameClientStep(
            _observation(self._step, 10 + self._step, lives=3 - int(self._step >= 2)),
            terminated=self._step >= 2,
        )

    def close(self) -> None:
        self.closed = True


class _FrameOnlyExtractor:
    def __init__(self) -> None:
        self.seen: list[ScreenFrame] = []

    def extract(self, frame: ScreenFrame) -> VisionGameMetrics:
        self.seen.append(frame)
        return VisionGameMetrics(
            score=int(frame.data[0]) * 10,
            lives=3 - int(frame.frame_id >= 2),
            x_progress=float(frame.frame_id),
        )


class _HoldRightPolicy:
    def act(self, observation: GameObservation) -> ControllerAction:
        return ControllerAction(right=True, b=True)


@pytest.mark.unit
def test_vision_only_loop_routes_frames_to_metrics_and_actions_to_client() -> None:
    client = _ScriptedClient()
    extractor = _FrameOnlyExtractor()
    reward = VisionMetricRewardModel(
        VisionMetricRewardConfig(score_scale=0.1, progress_scale=1.0)
    )
    loop = VisionOnlyGameLoop(
        client=client,
        policy=_HoldRightPolicy(),
        metric_extractor=extractor,
        reward_model=reward,
    )

    transitions = list(loop.run(max_steps=10))
    loop.close()

    assert len(transitions) == 2
    assert transitions[-1].done is True
    assert [action.to_bizhawk() for action in client.actions] == [
        {"Right": True, "B": True},
        {"Right": True, "B": True},
    ]
    assert [frame.frame_id for frame in extractor.seen] == [0, 1, 2]
    assert transitions[0].reward == pytest.approx(2.0)
    assert transitions[1].reward == pytest.approx(-98.0)
    assert client.closed is True


@pytest.mark.unit
def test_vision_only_loop_validates_max_steps() -> None:
    loop = VisionOnlyGameLoop(client=_ScriptedClient(), policy=_HoldRightPolicy())

    with pytest.raises(ValueError, match="max_steps"):
        list(loop.run(max_steps=0))


@pytest.mark.unit
def test_reward_model_handles_missing_metrics_without_crashing() -> None:
    reward = VisionMetricRewardModel(
        VisionMetricRewardConfig(missing_metric_reward=-0.5)
    )

    value = reward.reward(
        GameObservation(step=0, t=0.0, frame=_observation(0, 0).frame),
        GameObservation(step=1, t=0.01, frame=_observation(1, 0).frame),
    )

    assert value == pytest.approx(-2.0)


class _TerminateAtX:
    """Episode manager that ends the episode once x_progress crosses a threshold."""

    def __init__(self, threshold: float) -> None:
        self._threshold = threshold

    def should_end(
        self, before: GameObservation, after: GameObservation,
    ) -> EpisodeDecision:
        del before
        x = after.metrics.x_progress
        if x is not None and x >= self._threshold:
            return EpisodeDecision(terminated=True, reason="clear")
        return EpisodeDecision(terminated=False)


@pytest.mark.unit
def test_episode_manager_can_terminate_the_loop() -> None:
    # The client never terminates on its own; the episode manager must.
    loop = VisionOnlyGameLoop(
        client=ScriptedGameClient(width=1, height=1, channels=1),
        policy=_HoldRightPolicy(),
        metric_extractor=_FrameOnlyExtractor(),  # x_progress == frame_id
        episode_manager=_TerminateAtX(2.0),
    )

    transitions = list(loop.run(max_steps=10))

    assert len(transitions) == 2  # frame_id reaches 2 on the second step
    assert transitions[-1].terminated is True


class _CountingExtractor:
    def __init__(self) -> None:
        self.resets = 0

    def reset(self) -> None:
        self.resets += 1

    def extract(self, frame: ScreenFrame) -> VisionGameMetrics:
        return VisionGameMetrics(x_progress=float(frame.frame_id))


class _CountingReward:
    def __init__(self) -> None:
        self.begins = 0

    def begin_episode(self) -> None:
        self.begins += 1

    def reward(self, previous: GameObservation, current: GameObservation) -> float:
        del previous, current
        return 0.0


@pytest.mark.unit
def test_loop_resets_env_collaborators_each_episode() -> None:
    # Guards the x_progress-carryover fix: scroll/reward state must be cleared
    # at the start of every episode, not just the first.
    extractor = _CountingExtractor()
    reward = _CountingReward()
    loop = VisionOnlyGameLoop(
        client=ScriptedGameClient(width=1, height=1, channels=1, max_steps=1),
        policy=_HoldRightPolicy(),
        metric_extractor=extractor,
        reward_model=reward,
    )

    loop.reset()
    loop.reset()

    assert extractor.resets == 2
    assert reward.begins == 2
