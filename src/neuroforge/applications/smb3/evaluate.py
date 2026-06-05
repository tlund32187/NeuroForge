"""Evaluate an SMB3 policy checkpoint without changing it.

This is the live "is it actually getting better?" harness. It loads the policy
and perception state from artifacts/smb3_policy.pt, freezes plasticity, disables
motor noise by default, and records the same episode metrics as training. The
checkpoint path is read-only here: no final or periodic save points back to it.

Useful environment overrides for quick smoke runs:

  NEUROFORGE_EVAL_EPISODES=1
  NEUROFORGE_EVAL_FRAMES=120
  NEUROFORGE_EVAL_RANDOM_BASELINE=1
  NEUROFORGE_EVAL_DETERMINISTIC=0
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path
from typing import Any

from neuroforge.applications.smb3.env import env_bool, env_int
from neuroforge.applications.smb3.evolved_config import (
    apply_evolved_genome_config,
    evolved_config_status_lines,
)
from neuroforge.applications.tasks.game_training import GameTrainingTask
from neuroforge.contracts.messaging import EventTopic
from neuroforge.environments.games.clients import BizHawkConnectionError
from neuroforge.environments.games.clients.bizhawk.launcher import EmuHawkLauncher
from neuroforge.environments.games.smb3 import BizHawkClient, BizHawkClientConfig
from neuroforge.environments.games.smb3.adapters.bizhawk_smb3_adapter import (
    build_smb3_curriculum,
    build_smb3_episode_manager,
    build_smb3_game_training_config,
    build_smb3_hud_extractor,
    build_smb3_perception_stack,
    build_smb3_reward_model,
    existing_savestates,
)
from neuroforge.environments.games.smb3.state import resume_status_lines
from neuroforge.messaging.bus import EventBus
from neuroforge.observability.events.recorder import EventRecorderMonitor

EMUHAWK_PATH = r"C:\BizHawk\EmuHawk.exe"
ROM_PATH = r"C:\BizHawk\ROM\Super Mario Bros. 3 (USA) (Rev 1)\Super Mario Bros. 3 (USA) (Rev 1).nes"  # noqa: E501
SAVESTATE_PATHS: tuple[str, ...] = (
    r"C:\BizHawk\States\smb3_level1.State",
)

PORT = env_int("NEUROFORGE_SMB3_PORT", 8650, min_value=1)
FRAMESKIP = env_int("NEUROFORGE_SMB3_FRAMESKIP", 4, min_value=1)
BIZHAWK_SPEED_PERCENT = env_int("NEUROFORGE_SMB3_SPEED_PERCENT", 400, min_value=1)
MAX_EPISODES = env_int("NEUROFORGE_EVAL_EPISODES", 3, min_value=1)
FRAMES_PER_EPISODE = env_int("NEUROFORGE_EVAL_FRAMES", 600, min_value=1)
TELEMETRY_EVERY = env_int("NEUROFORGE_EVAL_TELEMETRY_EVERY", 60, min_value=0)
DETERMINISTIC = env_bool("NEUROFORGE_EVAL_DETERMINISTIC", True)
RUN_RANDOM_BASELINE = env_bool("NEUROFORGE_EVAL_RANDOM_BASELINE", False)

_REPO = Path(__file__).resolve().parents[4]
LUA_SCRIPT = _REPO / "scripts" / "bizhawk" / "neuroforge_bridge.lua"
CHECKPOINT = Path(os.environ.get("NEUROFORGE_SMB3_CHECKPOINT", str(_REPO / "artifacts" / "smb3_policy.pt")))  # noqa: E501
RUNS_DIR = _REPO / "artifacts" / "runs"
USE_EVOLVED_GENOME = env_bool("NEUROFORGE_SMB3_USE_EVOLVED", False)
EVOLUTION_CHECKPOINT = os.environ.get("NEUROFORGE_SMB3_EVOLUTION_CHECKPOINT")
EVOLVED_MODE = os.environ.get("NEUROFORGE_SMB3_EVOLVED_MODE", "compatible").strip().lower()


class _EvaluationMonitor:
    """Collect and print episode-level eval metrics."""

    enabled = True

    def __init__(self, label: str) -> None:
        self._label = label
        self.trials: list[dict[str, Any]] = []

    def on_event(self, event: Any) -> None:
        topic = event.topic.value
        data = event.data
        if topic == "run_start":
            print(f"  [{self._label}] resumed={bool(data.get('resumed'))}")
            for line in resume_status_lines(data.get("resume")):
                print(f"  [{self._label}] {line}")
        elif topic == "training_trial":
            trial = dict(data)
            self.trials.append(trial)
            print(
                f"  [{self._label} ep {trial.get('episode')}] "
                f"frames={trial.get('frames')} "
                f"reward/frame={_number(trial, 'reward_mean'):+.3f} "
                f"max_x={_number(trial, 'max_x_progress'):.3f} "
                f"buttons={_number(trial, 'action_button_mean'):.2f} "
                f"changes={_number(trial, 'action_change_mean'):.2f}",
            )
        elif topic in {"training_end", "run_end"}:
            print(f"  [{self._label} {topic}] {dict(data)}")

    def reset(self) -> None:
        self.trials.clear()

    def snapshot(self) -> dict[str, Any]:
        return {"trials": list(self.trials)}

    def summary(self) -> dict[str, float]:
        reward_means = [_number(trial, "reward_mean") for trial in self.trials]
        return {
            "episodes": float(len(self.trials)),
            "frames": sum(_number(trial, "frames") for trial in self.trials),
            "reward_mean": _mean(reward_means),
            "max_x_progress": max(
                (_number(trial, "max_x_progress") for trial in self.trials),
                default=0.0,
            ),
            "action_button_mean": _mean(
                [_number(trial, "action_button_mean") for trial in self.trials],
            ),
            "action_change_mean": _mean(
                [_number(trial, "action_change_mean") for trial in self.trials],
            ),
        }


def _number(data: dict[str, Any], key: str) -> float:
    value = data.get(key, 0.0)
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    return 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _validate_static_paths() -> bool:
    ok = True
    for label, path in (("EmuHawk", EMUHAWK_PATH), ("ROM", ROM_PATH), ("Lua", str(LUA_SCRIPT))):
        if not Path(path).exists():
            print(f"ERROR: {label} not found: {path}")
            ok = False
    return ok


def _resolve_savestates() -> tuple[str, ...]:
    present = existing_savestates(SAVESTATE_PATHS)
    missing = tuple(path for path in SAVESTATE_PATHS if path not in present)
    for path in missing:
        print(f"  WARNING: savestate not found: {path}")
    if not present:
        print("  WARNING: no savestates found; eval will boot from the ROM.")
    return present


def _build_eval_config(*, checkpoint_path: Path | None) -> Any:
    cfg = build_smb3_game_training_config(
        max_episodes=MAX_EPISODES,
        frames_per_episode=FRAMES_PER_EPISODE,
        telemetry_every=TELEMETRY_EVERY,
        checkpoint_path=None,
        checkpoint_every=0,
        resume=checkpoint_path is not None,
        resume_checkpoint_path=str(checkpoint_path) if checkpoint_path else None,
        plastic=False,
        deterministic=DETERMINISTIC,
    )
    selection = apply_evolved_genome_config(
        cfg,
        use_evolved=USE_EVOLVED_GENOME,
        evolved_mode=EVOLVED_MODE,
        evolution_checkpoint=EVOLUTION_CHECKPOINT,
        runs_dir=RUNS_DIR,
    )
    for line in evolved_config_status_lines(selection, action="eval"):
        print(f"  {line}")
    return selection.config


def _run_evaluation(
    *,
    label: str,
    savestates: tuple[str, ...],
    checkpoint_path: Path | None,
) -> dict[str, float]:
    run_dir = RUNS_DIR / dt.datetime.now().astimezone().strftime(f"smb3_eval_%Y%m%d_%H%M%S_{label}")  # noqa: E501
    bus = EventBus()
    monitor = _EvaluationMonitor(label)
    recorder = EventRecorderMonitor(run_dir)
    for topic in EventTopic:
        bus.subscribe(topic, monitor)
        bus.subscribe(topic, recorder)

    print(f"  [{label}] metrics: {run_dir / 'events' / 'events.ndjson'}")
    client: BizHawkClient | None = None
    try:
        client = BizHawkClient(
            BizHawkClientConfig(
                port=PORT,
                width=256,
                height=240,
                channels=3,
                frameskip=FRAMESKIP,
                connect_timeout_s=60.0,
                step_timeout_s=45.0,
                launch=True,
                transport="socket",
            ),
            launcher=EmuHawkLauncher(
                emuhawk_path=EMUHAWK_PATH,
                lua_script=str(LUA_SCRIPT),
                rom_path=ROM_PATH,
                frameskip=FRAMESKIP,
                speed_percent=BIZHAWK_SPEED_PERCENT,
            ),
        )
        task = GameTrainingTask(
            _build_eval_config(checkpoint_path=checkpoint_path),
            event_bus=bus,
            client=client,
            metric_extractor=build_smb3_hud_extractor(),
            reward_model=build_smb3_reward_model(),
            episode_manager=build_smb3_episode_manager(),
            curriculum=build_smb3_curriculum(savestates),
            encoder=build_smb3_perception_stack(learn=False),
        )
        task.run()
        summary = monitor.summary()
        return summary
    finally:
        recorder.close()
        if client is not None:
            client.close()


def _print_summary(label: str, summary: dict[str, float]) -> None:
    print(
        f"  [{label} summary] episodes={summary['episodes']:.0f} "
        f"frames={summary['frames']:.0f} reward/frame={summary['reward_mean']:+.3f} "
        f"max_x={summary['max_x_progress']:.3f} "
        f"buttons={summary['action_button_mean']:.2f} "
        f"changes={summary['action_change_mean']:.2f}",
    )


def main() -> int:
    print("=" * 70)
    print("NeuroForge - SMB3 checkpoint evaluation (plasticity off)")
    print("=" * 70)
    if not _validate_static_paths():
        return 1

    savestates = _resolve_savestates()
    print(
        f"  episodes={MAX_EPISODES} frames/episode={FRAMES_PER_EPISODE} "
        f"frameskip={FRAMESKIP} deterministic={DETERMINISTIC}",
    )

    jobs: list[tuple[str, Path | None]] = []
    if CHECKPOINT.exists():
        print(f"  checkpoint: {CHECKPOINT}")
        jobs.append(("checkpoint", CHECKPOINT))
    else:
        print(f"  WARNING: checkpoint not found: {CHECKPOINT}")
    if RUN_RANDOM_BASELINE:
        jobs.append(("random_baseline", None))
    if not jobs:
        print("ERROR: nothing to evaluate. Create a checkpoint or enable random baseline.")
        return 1

    summaries: dict[str, dict[str, float]] = {}
    try:
        for label, checkpoint_path in jobs:
            summaries[label] = _run_evaluation(
                label=label,
                savestates=savestates,
                checkpoint_path=checkpoint_path,
            )
            _print_summary(label, summaries[label])
    except KeyboardInterrupt:
        print("\n  Stopped by user.")
        return 130
    except BizHawkConnectionError as exc:
        print(f"\n  Emulator error: {exc}")
        print("  (Keep the EmuHawk window focused while evaluating.)")
        return 1

    if "checkpoint" in summaries and "random_baseline" in summaries:
        delta = (
            summaries["checkpoint"]["max_x_progress"]
            - summaries["random_baseline"]["max_x_progress"]
        )
        print(f"  checkpoint max_x - random max_x = {delta:+.3f}")
    print("  Checkpoint was loaded read-only; this script did not save policy weights.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
