"""Save/load learned policy state for the game-training loop.

Persists what learning produced — the per-projection synaptic weights and
eligibility traces — plus light metadata (episode/frame, reward baseline), so a
run can be paused and resumed. Weights are the durable artifact; eligibility is
short-lived but restoring it makes resumption seamless.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

if TYPE_CHECKING:
    from neuroforge.learning.training_loop import OnlineRSTDPTrainer

__all__ = ["PolicyCheckpoint", "SMB3RunState", "resume_status_lines"]

_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SMB3RunState:
    """Small serializable snapshot of SMB3 run progress."""

    episode: int = 0
    frame: int = 0
    reward_baseline: float = 0.0


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
        from neuroforge.kernel.torch_utils import require_torch

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
        allow_partial: bool = False,
    ) -> dict[str, Any]:
        """Load weights + eligibility (+ encoder state) from *path*; return metadata."""
        from neuroforge.kernel.torch_utils import require_torch

        torch = require_torch()
        payload: dict[str, Any] = torch.load(
            str(path), map_location="cpu", weights_only=False,
        )
        load_summary: dict[str, Any] = {
            "allow_partial": bool(allow_partial),
            "weights": {},
            "eligibility": {},
            "encoder_present": False,
            "encoder_loaded": False,
        }
        weights = payload.get("weights")
        if isinstance(weights, dict):
            load_summary["weights"] = trainer.load_weights(
                cast("dict[str, Any]", weights),
                allow_partial=allow_partial,
            )
        eligibility = payload.get("eligibility")
        if isinstance(eligibility, dict):
            load_summary["eligibility"] = trainer.load_eligibility(
                cast("dict[str, Any]", eligibility),
                allow_partial=allow_partial,
            )
        encoder_state = payload.get("encoder")
        load_summary["encoder_present"] = isinstance(encoder_state, dict)
        if (
            encoder is not None
            and isinstance(encoder_state, dict)
            and hasattr(encoder, "load_state_dict")
        ):
            encoder.load_state_dict(encoder_state)
            load_summary["encoder_loaded"] = True
        payload["load_summary"] = load_summary
        return payload


def resume_status_lines(resume: object) -> list[str]:
    """Return concise console lines describing checkpoint resume status."""
    if not isinstance(resume, Mapping):
        return []
    resume_map = cast("Mapping[str, Any]", resume)
    if not bool(resume_map.get("requested", False)):
        return []

    path = str(resume_map.get("path", ""))
    path_text = f" ({path})" if path else ""
    if not bool(resume_map.get("loaded", False)):
        reason = str(resume_map.get("reason", "") or "not_loaded")
        return [f"Resume requested but not loaded: {reason}{path_text}."]

    weights_copied = _int_field(resume_map, "weights_copied")
    weights_target = _int_field(resume_map, "weights_target")
    eligibility_copied = _int_field(resume_map, "eligibility_copied")
    eligibility_target = _int_field(resume_map, "eligibility_target")
    encoder = "yes" if bool(resume_map.get("encoder_loaded", False)) else "no"
    partial = "yes" if _has_partial_load(resume_map.get("load_summary", {})) else "no"
    return [
        "Resume loaded"
        f"{path_text}: weights={_coverage(weights_copied, weights_target)}, "
        f"eligibility={_coverage(eligibility_copied, eligibility_target)}, "
        f"encoder={encoder}, partial={partial}."
    ]


def _coverage(copied: int, target: int) -> str:
    if target <= 0:
        return f"{copied}/{target}"
    return f"{copied}/{target} ({100.0 * copied / target:.1f}%)"


def _int_field(data: Mapping[str, Any], key: str) -> int:
    value = data.get(key, 0)
    if isinstance(value, bool):
        return 0
    if isinstance(value, int):
        return value
    return 0


def _has_partial_load(load_summary: object) -> bool:
    if not isinstance(load_summary, Mapping):
        return False
    summary = cast("Mapping[str, Any]", load_summary)
    for section in ("weights", "eligibility"):
        raw_section = summary.get(section, {})
        if not isinstance(raw_section, Mapping):
            continue
        for raw_item in cast("Mapping[str, Any]", raw_section).values():
            if isinstance(raw_item, Mapping) and bool(
                cast("Mapping[str, Any]", raw_item).get("partial", False),
            ):
                return True
    return False
