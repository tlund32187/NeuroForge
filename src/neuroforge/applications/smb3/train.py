"""Live online R-STDP training: a spiking brain learns SMB3 by vision.

Wires the whole stack together against the real emulator: BizHawk bridge ->
SMB3 HUD/scroll reward -> stateful spiking policy -> online R-STDP weight
updates. Press Run/play on "NeuroForge: Train SMB3" (or run from a terminal).

PHASE 4 - START INSIDE A LEVEL.  Booting from the ROM drops the brain at the
title/map screens, where there is no reward to learn from (it can only flail).
So training instead begins from a SAVESTATE placed at the start of a level,
where the vision reward (forward scroll progress, score, staying alive) is
dense and learnable. Loading a savestate is an environment reset, not memory
reading - the policy still only ever sees pixels.

ONE-TIME SETUP (create the savestate):
  1. Open EmuHawk, load the SMB3 ROM, and play until Mario is standing at the
     very start of a level (e.g. World 1-1, just after it loads).
  2. File -> Save State -> Save State As...  and save to the path in
     SAVESTATE_PATH below (default: C:\\BizHawk\\States\\smb3_level1.State).
  3. Run this script. Each episode reloads that savestate, so the brain gets
     many fresh attempts at the same level and learns to make progress.

If the savestate file is missing this still runs (booting the ROM) but prints a
warning - it will not learn much until you create the savestate. Learned
weights are checkpointed to artifacts/smb3_policy.pt.
"""

from __future__ import annotations

import datetime as dt
import os
import sys
from pathlib import Path
from typing import Any

from neuroforge.applications.smb3.env import env_bool, env_float, env_int
from neuroforge.applications.smb3.evolved_config import (
    apply_evolved_genome_config,
    evolved_config_status_lines,
)
from neuroforge.applications.tasks.game_training import GameTrainingTask
from neuroforge.environments.games.clients import BizHawkConnectionError
from neuroforge.environments.games.clients.bizhawk.launcher import EmuHawkLauncher
from neuroforge.environments.games.smb3 import (
    BizHawkClient,
    BizHawkClientConfig,
)
from neuroforge.environments.games.smb3.adapters.bizhawk_smb3_adapter import (
    build_smb3_curriculum,
    build_smb3_episode_manager,
    build_smb3_game_training_config,
    build_smb3_hud_extractor,
    build_smb3_perception_stack,
    build_smb3_reward_model,
)
from neuroforge.environments.games.smb3.state import resume_status_lines
from neuroforge.messaging.bus import EventBus
from neuroforge.neuroevolution import attach_learned_checkpoint_to_evolution_checkpoint
from neuroforge.observability.events.recorder import EventRecorderMonitor

# Config: edit if your paths differ.
EMUHAWK_PATH = r"C:\BizHawk\EmuHawk.exe"
ROM_PATH = (
    r"C:\BizHawk\ROM\Super Mario Bros. 3 (USA) (Rev 1)"
    r"\Super Mario Bros. 3 (USA) (Rev 1).nes"
)
# Savestate placed at the START OF A LEVEL (see ONE-TIME SETUP in the docstring).
# Add more paths to train through a curriculum of increasingly hard levels.
SAVESTATE_PATHS: tuple[str, ...] = (
    r"C:\BizHawk\States\smb3_level1.State",
)
PORT = env_int("NEUROFORGE_SMB3_PORT", 8650, min_value=1)
MAX_EPISODES = env_int("NEUROFORGE_SMB3_TRAIN_EPISODES", 50, min_value=1)
FRAMES_PER_EPISODE = env_int("NEUROFORGE_SMB3_TRAIN_FRAMES", 2000, min_value=1)
TELEMETRY_EVERY = env_int("NEUROFORGE_SMB3_TRAIN_TELEMETRY_EVERY", 30, min_value=0)
# Emulator frames advanced per decision (action-repeat). Higher = much faster
# wall-clock (fewer IPC round-trips + brain steps per second of gameplay) AND it
# helps build run momentum. With BizHawk at 400%, 4 targets about 60 decisions/sec.
FRAMESKIP = env_int("NEUROFORGE_SMB3_FRAMESKIP", 4, min_value=1)
# BizHawk throttle speed percent. 400 is the menu's "Speed 400%" setting (4x),
# so FRAMESKIP=4 should land near 60 decisions/sec when transport keeps up.
BIZHAWK_SPEED_PERCENT = env_int("NEUROFORGE_SMB3_SPEED_PERCENT", 400, min_value=1)
# Continue learning from the last checkpoint instead of starting from random
# weights. Set False for a fresh brain.
RESUME = env_bool("NEUROFORGE_SMB3_RESUME", True)
CHECKPOINT_EVERY = env_int("NEUROFORGE_SMB3_CHECKPOINT_EVERY", 1000, min_value=0)


_REPO = Path(__file__).resolve().parents[4]
LUA_SCRIPT = _REPO / "scripts" / "bizhawk" / "neuroforge_bridge.lua"
RUNS_DIR = _REPO / "artifacts" / "runs"
CHECKPOINT = Path(
    os.environ.get("NEUROFORGE_SMB3_CHECKPOINT", str(_REPO / "artifacts" / "smb3_policy.pt"))
)
USE_EVOLVED_GENOME = env_bool("NEUROFORGE_SMB3_USE_EVOLVED", False)
EVOLUTION_CHECKPOINT = os.environ.get("NEUROFORGE_SMB3_EVOLUTION_CHECKPOINT")
EVOLVED_MODE = os.environ.get("NEUROFORGE_SMB3_EVOLVED_MODE", "compatible").strip().lower()
LAMARCKIAN_WRITEBACK = env_bool(
    "NEUROFORGE_SMB3_LAMARCKIAN_WRITEBACK",
    EVOLVED_MODE == "full",
)
RESUME_CHECKPOINT = os.environ.get("NEUROFORGE_SMB3_RESUME_CHECKPOINT")
CONSOLIDATION_STRENGTH = env_float(
    "NEUROFORGE_SMB3_CONSOLIDATION_STRENGTH",
    0.0,
    min_value=0.0,
)


class _ConsoleMonitor:
    """Print training telemetry so you can watch the loop learn."""

    enabled = True

    def __init__(self, every: int) -> None:
        self._every = max(1, every)

    def on_event(self, event: Any) -> None:
        topic = event.topic.value
        data = event.data
        if topic == "scalar" and int(data.get("frame", 0)) % self._every == 0:
            print(
                f"  frame {data.get('frame'):>5} | reward {data.get('reward_raw', 0.0):+7.2f}"
                f" | shaped {data.get('reward_shaped', 0.0):+6.3f}"
                f" | dw {data.get('dw_norm', 0.0):7.4f}"
                f" | x {data.get('x_progress', 0.0):.3f}"
                f" | score {data.get('score', '?')}",
            )
        elif topic == "training_trial":
            print(
                f"  [episode {data.get('episode')}] frames={data.get('frames')}"
                f" reward/frame={data.get('reward_mean', 0.0):+.3f}"
                f" reward_sum={data.get('reward_sum', 0.0):+.1f}"
                f" max_x={data.get('max_x_progress', 0.0):.3f}"
                f" stage={data.get('curriculum_stage', 0)}",
            )
        elif topic == "run_start" and not data.get("resumed"):
            for line in resume_status_lines(data.get("resume")):
                print(f"  {line}")
        elif topic == "run_start":
            for line in resume_status_lines(data.get("resume")):
                print(f"  {line}")
            print("  Resumed from checkpoint - continuing to learn from prior runs.")
            strength = float(data.get("consolidation_strength", 0.0))
            if data.get("resumed") and strength > 0.0:
                print(f"  Consolidation anchor enabled (strength={strength:.4g}).")
        elif topic in {"training_end", "run_end"}:
            print(f"  [{topic}] {dict(data)}")

    def reset(self) -> None: ...

    def snapshot(self) -> dict[str, Any]:
        return {}


def _resolve_savestates() -> tuple[str, ...]:
    """Keep only savestate files that exist; warn loudly about any that don't."""
    present = tuple(p for p in SAVESTATE_PATHS if Path(p).exists())
    missing = tuple(p for p in SAVESTATE_PATHS if not Path(p).exists())
    for path in missing:
        print(f"  WARNING: savestate not found: {path}")
    if not present:
        print(
            "  No savestate available - booting from the ROM instead. The brain\n"
            "  will start at the title/map screens (no reward there) and mostly\n"
            "  flail. See ONE-TIME SETUP at the top of this file to create one.",
        )
    else:
        print(f"  Curriculum savestates: {len(present)} (starting in-level).")
    return present


def _apply_evolved_genome_config(cfg: Any) -> Any:
    """Optionally select the best evolved genome; returns the full selection.

    For a structural (graph) champion the selection also carries a
    `
etwork_builder`` that compiles its invented topology into the policy.
    """
    selection = apply_evolved_genome_config(
        cfg,
        use_evolved=USE_EVOLVED_GENOME,
        evolved_mode=EVOLVED_MODE,
        evolution_checkpoint=EVOLUTION_CHECKPOINT,
        runs_dir=RUNS_DIR,
    )
    for line in evolved_config_status_lines(selection, action="training"):
        print(f"  {line}")
    return selection


def _maybe_write_lamarckian_state(selection: Any) -> None:
    """Attach the just-trained policy checkpoint to the selected evolution genome."""
    if not LAMARCKIAN_WRITEBACK:
        return
    if not getattr(selection, "applied", False):
        print("  Lamarckian writeback skipped: no evolved genome was selected.")
        return
    checkpoint_path = getattr(selection, "checkpoint_path", None)
    if checkpoint_path is None:
        print("  Lamarckian writeback skipped: no evolution checkpoint path.")
        return
    if not CHECKPOINT.exists():
        print(f"  Lamarckian writeback skipped: learned checkpoint missing at {CHECKPOINT}.")
        return
    try:
        summary = attach_learned_checkpoint_to_evolution_checkpoint(
            checkpoint_path,
            CHECKPOINT,
            genome_id=str(getattr(selection, "genome_id", "")),
            source="smb3_train",
        )
    except ValueError as exc:
        print(f"  Lamarckian writeback skipped: {exc}")
        return
    print(
        "  Lamarckian writeback: "
        f"{summary['learned_checkpoint']} -> {summary['checkpoint_path']} "
        f"(population matches={summary['population_updates']})",
    )


def main() -> int:
    print("=" * 70)
    print("NeuroForge - SMB3 online R-STDP training (Phase 4: in-level curriculum)")
    print("=" * 70)
    for label, path in (("EmuHawk", EMUHAWK_PATH), ("ROM", ROM_PATH), ("Lua", str(LUA_SCRIPT))):
        if not Path(path).exists():
            print(f"ERROR: {label} not found: {path}\nEdit the CONFIG block in this file.")
            return 1

    savestates = _resolve_savestates()
    curriculum = build_smb3_curriculum(
        savestates,
        advance_threshold=0.9,
        min_episodes_per_stage=4,
    )

    extractor = build_smb3_hud_extractor()
    print(f"  HUD digit OCR calibrated: {extractor.is_calibrated}")
    print(f"  BizHawk speed target: {BIZHAWK_SPEED_PERCENT}% | frameskip: {FRAMESKIP}")
    print(f"  Lamarckian writeback: {LAMARCKIAN_WRITEBACK}")
    print("  Launching EmuHawk; the brain will start driving (and learning) shortly.\n")

    bus = EventBus()
    from neuroforge.contracts.messaging import EventTopic

    monitor = _ConsoleMonitor(TELEMETRY_EVERY)
    for topic in EventTopic:
        bus.subscribe(topic, monitor)

    # Persist every metric to NDJSON so a run can be evaluated afterward and runs
    # can be compared over time (one file per run under artifacts/runs/).
    run_dir = RUNS_DIR / dt.datetime.now().astimezone().strftime("run_%Y%m%d_%H%M%S")
    recorder = EventRecorderMonitor(run_dir)
    for topic in EventTopic:
        bus.subscribe(topic, recorder)
    print(f"  Metrics log: {run_dir / 'events' / 'events.ndjson'}")

    client = BizHawkClient(
        BizHawkClientConfig(
            port=PORT, width=256, height=240, channels=3, frameskip=FRAMESKIP,
            connect_timeout_s=60.0, step_timeout_s=45.0, launch=True, transport="socket",
        ),
        launcher=EmuHawkLauncher(
            emuhawk_path=EMUHAWK_PATH, lua_script=str(LUA_SCRIPT), rom_path=ROM_PATH,
            frameskip=FRAMESKIP, speed_percent=BIZHAWK_SPEED_PERCENT,
        ),
    )
    cfg = build_smb3_game_training_config(
        # Exploration is essential early: a random brain has a fixed direction bias,
        # so without stochastic decoding + a stochastic d-pad tie-break it gets stuck
        # going one way forever and never earns the progress reward for the other.
        # Hold the decoded heading for a window so the brain stops dithering
        # ("runs in place") and builds run momentum - the action-side trace rule.
        # Larger learning signal so the (rare) forward-progress reward isn't lost.
        max_episodes=MAX_EPISODES,
        frames_per_episode=FRAMES_PER_EPISODE,
        telemetry_every=TELEMETRY_EVERY,
        checkpoint_every=CHECKPOINT_EVERY,
        checkpoint_path=str(CHECKPOINT),
        resume=RESUME,
        resume_checkpoint_path=RESUME_CHECKPOINT,
        consolidation_strength=CONSOLIDATION_STRENGTH,
    )
    selection = _apply_evolved_genome_config(cfg)
    cfg = selection.config
    # An evolved GRAPH champion (full mode) compiles its invented topology here;
    # a hyperparameter champion / no champion leaves this None (build from config).
    network_builder = selection.network_builder
    # Bio-faithful perception front-end (Track A), full stack: A0 retinal contrast
    # -> A1 STDP feature maps -> A2 trace-rule object cells, plus A3 motion. A1/A2
    # learn online while playing and are checkpointed with the policy (so resume
    # stays coherent). The brain now sees learned, invariant features + motion.
    encoder = build_smb3_perception_stack(learn=True)
    # Reward shaping (starting points - tune against the metrics log). Forward
    # progress dominates; idle/stall are gentle nudges (not a swamping per-frame
    # drag), so reward/frame is interpretable and progress stands out.
    reward_model = build_smb3_reward_model()
    task = GameTrainingTask(
        cfg,
        event_bus=bus,
        client=client,
        metric_extractor=extractor,
        reward_model=reward_model,
        episode_manager=build_smb3_episode_manager(),
        curriculum=curriculum,
        encoder=encoder,
        network_builder=network_builder,
    )

    try:
        result = task.run()
        print("\n" + "=" * 70)
        print(f"  DONE: {result}")
        print(f"  Learned weights checkpointed to {CHECKPOINT}")
        _maybe_write_lamarckian_state(selection)
        print(f"  Full metrics: {run_dir / 'events' / 'events.ndjson'}")
        print("=" * 70)
    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    except BizHawkConnectionError as exc:
        print(f"\n  Emulator disconnected: {exc}")
        print("  (Keep the EmuHawk window focused while training.)")
    finally:
        recorder.close()
        client.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
