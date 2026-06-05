"""Vision-only game episode loop."""

from __future__ import annotations

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
    ) -> None:
        self._client = client
        self._policy = policy
        self._metric_extractor = metric_extractor
        self._reward_model = reward_model
        self._episode_manager = episode_manager
        self._close_client = close_client

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
        return self._with_frame_metrics(self._client.reset())

    def step(self, current: GameObservation) -> GameTransition:
        """Run one policy/client/reward step."""
        action = self._policy.act(current)
        result = self._client.step(action)
        after = self._with_frame_metrics(result.observation)
        reward = (
            self._reward_model.reward(current, after)
            if self._reward_model is not None
            else 0.0
        )
        terminated = result.terminated
        if self._episode_manager is not None:
            decision = self._episode_manager.should_end(current, after)
            terminated = terminated or decision.terminated
        return GameTransition(
            before=current,
            action=action,
            after=after,
            reward=float(reward),
            terminated=terminated,
            truncated=result.truncated,
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
        return observation.with_metrics(self._metric_extractor.extract(observation.frame))
