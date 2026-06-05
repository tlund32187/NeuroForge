"""Vision-only game episode loop."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from neuroforge.contracts.applications.games import (
    GameObservation,
    GameTransition,
    IEpisodeManager,
    IFrameMetricExtractor,
    IGameClient,
    IRewardModel,
    IVisionGamePolicy,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = ["VisionOnlyGameLoop"]


def _maybe_begin_episode(obj: object) -> None:
    """Call ``obj.begin_episode()`` / ``obj.reset()`` if either exists.

    Lets the loop centralise per-episode lifecycle for the env-side collaborators
    (metric extractor scroll history, stateful reward model, episode manager)
    without forcing every implementation to define the hook.
    """
    for name in ("begin_episode", "reset"):
        hook = getattr(obj, name, None)
        if callable(hook):
            hook()
            return


class VisionOnlyGameLoop:
    """Orchestrate a game client, policy, metric extractor, and reward model.

    The loop intentionally routes only :class:`ScreenFrame` data into the
    metric extractor and only frame-derived observations into the reward model.
    This keeps the integration compatible with vision-only runs where game RAM
    and emulator memory APIs are off-limits.
    """

    def __init__(
        self,
        *,
        client: IGameClient,
        policy: IVisionGamePolicy,
        metric_extractor: IFrameMetricExtractor | None = None,
        reward_model: IRewardModel | None = None,
        episode_manager: IEpisodeManager | None = None,
        close_client: bool = True,
        timings: dict[str, float] | None = None,
    ) -> None:
        self._client = client
        self._policy = policy
        self._metric_extractor = metric_extractor
        self._reward_model = reward_model
        self._episode_manager = episode_manager
        self._close_client = close_client
        self._timings = timings

    def reset(self) -> GameObservation:
        """Reset per-episode state, reset the client, and return the first frame.

        The env-side collaborators are reset *before* the first frame is read so
        scroll-based ``x_progress`` does not carry over from the previous episode
        (and so a savestate-load jump never registers as bogus forward progress).
        """
        for collaborator in (
            self._metric_extractor,
            self._reward_model,
            self._episode_manager,
        ):
            if collaborator is not None:
                _maybe_begin_episode(collaborator)
        started = time.perf_counter()
        try:
            observation = self._client.reset()
        finally:
            self._record("client_reset_seconds", time.perf_counter() - started)
        return self._with_frame_metrics(observation)

    def step(self, current: GameObservation) -> GameTransition:
        """Run one policy/client/reward step."""
        started = time.perf_counter()
        action = self._policy.act(current)
        self._record("policy_seconds", time.perf_counter() - started)
        started = time.perf_counter()
        result = self._client.step(action)
        self._record("client_step_seconds", time.perf_counter() - started)
        after = self._with_frame_metrics(result.observation)
        started = time.perf_counter()
        reward = 0.0
        if self._reward_model is not None:
            reward = self._reward_model.reward(current, after)
        self._record("reward_seconds", time.perf_counter() - started)
        terminated = result.terminated
        termination_reason: str | None = "client" if terminated else None
        if self._episode_manager is not None:
            started = time.perf_counter()
            decision = self._episode_manager.should_end(current, after)
            self._record("episode_seconds", time.perf_counter() - started)
            terminated = terminated or decision.terminated
            if decision.terminated:
                termination_reason = decision.reason
        if result.truncated and termination_reason is None:
            termination_reason = "truncated"
        return GameTransition(
            before=current,
            action=action,
            after=after,
            reward=float(reward),
            terminated=terminated,
            truncated=result.truncated,
            termination_reason=termination_reason,
        )

    def run(self, *, max_steps: int) -> Iterator[GameTransition]:
        """Yield up to ``max_steps`` transitions, stopping when the client ends."""
        if max_steps <= 0:
            msg = "max_steps must be > 0"
            raise ValueError(msg)

        current = self.reset()
        for _ in range(max_steps):
            transition = self.step(current)
            yield transition
            current = transition.after
            if transition.done:
                break

    def close(self) -> None:
        """Release client resources, unless the client is borrowed (shared)."""
        if self._close_client:
            self._client.close()

    def _with_frame_metrics(self, observation: GameObservation) -> GameObservation:
        if self._metric_extractor is None:
            return observation
        started = time.perf_counter()
        try:
            metrics = self._metric_extractor.extract(observation.frame)
        finally:
            self._record("metrics_seconds", time.perf_counter() - started)
        return observation.with_metrics(metrics)

    def _record(self, key: str, seconds: float) -> None:
        if self._timings is None:
            return
        self._timings[key] = self._timings.get(key, 0.0) + max(0.0, float(seconds))
