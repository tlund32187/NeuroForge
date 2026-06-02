# pyright: basic
"""Save/load learned policy state for the game-training loop.

Persists what learning produced — the per-projection synaptic weights and
eligibility traces — plus light metadata (episode/frame, reward baseline), so a
run can be paused and resumed. Weights are the durable artifact; eligibility is
short-lived but restoring it makes resumption seamless.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from neuroforge.learning.online_rstdp import OnlineRSTDPTrainer

__all__ = ["PolicyCheckpoint"]

_SCHEMA_VERSION = 1


class PolicyCheckpoint:
    """Round-trip the trainer's learned state to a ``.pt`` file."""

    @staticmethod
    def save(
        path: str | Path,
        *,
        trainer: OnlineRSTDPTrainer,
        episode: int = 0,
        frame: int = 0,
        extra: dict[str, Any] | None = None,
        encoder: Any | None = None,
    ) -> None:
        """Write the trainer's weights + eligibility + metadata to *path*.

        If *encoder* has a ``state_dict`` (e.g. a learning ``PerceptionStack``),
        its learned state is saved too, so a resumed run's policy and perception
        stay aligned (the policy was trained against that perception).
        """
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()
        payload: dict[str, Any] = {
            "schema": _SCHEMA_VERSION,
            "episode": int(episode),
            "frame": int(frame),
            "reward_baseline": float(trainer.reward_baseline),
            "weights": trainer.weights_snapshot(),
            "eligibility": trainer.eligibility_snapshot(),
            "extra": dict(extra or {}),
        }
        if encoder is not None and hasattr(encoder, "state_dict"):
            payload["encoder"] = encoder.state_dict()
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, str(out))

    @staticmethod
    def load(
        path: str | Path,
        *,
        trainer: OnlineRSTDPTrainer,
        encoder: Any | None = None,
    ) -> dict[str, Any]:
        """Load weights + eligibility (+ encoder state) from *path*; return metadata."""
        from neuroforge.core.torch_utils import require_torch

        torch = require_torch()
        payload: dict[str, Any] = torch.load(
            str(path), map_location="cpu", weights_only=False,
        )
        weights = payload.get("weights")
        if isinstance(weights, dict):
            trainer.load_weights(weights)
        eligibility = payload.get("eligibility")
        if isinstance(eligibility, dict):
            trainer.load_eligibility(eligibility)
        encoder_state = payload.get("encoder")
        if (
            encoder is not None
            and isinstance(encoder_state, dict)
            and hasattr(encoder, "load_state_dict")
        ):
            encoder.load_state_dict(encoder_state)
        return payload
