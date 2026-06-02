"""Short SMB3 neuroevolution run using the existing game-training evaluator.

This is the live Track E entrypoint. It evolves SMB3 policy genomes while each
genome is scored by a short `GameTrainingTask` episode budget. The evaluator
uses the same BizHawk client, HUD extractor, reward, curriculum, and perception
stack as online SMB3 training.

Start with tiny populations/generation counts. Live evaluation is expensive:
population * generations * episodes * frames, plus emulator startup when
`LAUNCH_EMUHAWK=True`.
"""

from __future__ import annotations

import datetime as dt
import sys
from pathlib import Path
from typing import Any

from neuroforge.contracts.monitors import EventTopic
from neuroforge.evolution import (
    EvolutionConfig,
    SMB3LiveFitnessConfig,
    build_live_smb3_fitness_evaluator,
    existing_savestates,
)
from neuroforge.game.clients import BizHawkConnectionError
from neuroforge.monitors.bus import EventBus
from neuroforge.monitors.event_recorder import EventRecorderMonitor
from neuroforge.tasks.evolution import EvolutionTask

# Machine-local paths. Edit these when your BizHawk install moves.
EMUHAWK_PATH = r"C:\BizHawk\EmuHawk.exe"
ROM_PATH = r"C:\BizHawk\ROM\Super Mario Bros. 3 (USA) (Rev 1)\Super Mario Bros. 3 (USA) (Rev 1).nes"  # noqa: E501
SAVESTATE_PATHS: tuple[str, ...] = (
    r"C:\BizHawk\States\smb3_level1.State",
)

PORT = 8650
FRAMESKIP = 4
BIZHAWK_SPEED_PERCENT = 400
LAUNCH_EMUHAWK = True

POPULATION_SIZE = 4
GENERATIONS = 2
ELITE_COUNT = 1
EVAL_EPISODES = 1
EVAL_FRAMES_PER_EPISODE = 600
EVOLUTION_WORKERS = 1  # keep live BizHawk at 1 unless each worker owns a separate emulator

_REPO = Path(__file__).resolve().parents[1]
LUA_SCRIPT = _REPO / "bizhawk" / "neuroforge_bridge.lua"
RUNS_DIR = _REPO / "artifacts" / "runs"


class _ConsoleMonitor:
    enabled = True

    def on_event(self, event: Any) -> None:
        topic = event.topic.value
        data = event.data
        if topic == "scalar" and "best_fitness" in data:
            print(
                f"  generation {data.get('generation')} | "
                f"best={float(data.get('best_fitness', 0.0)):.3f} | "
                f"mean={float(data.get('mean_fitness', 0.0)):.3f} | "
                f"species={data.get('species_count')}",
            )
        elif topic == "training_trial" and event.source == "evolution":
            print(
                f"    genome {data.get('genome_id')} | "
                f"fitness={float(data.get('fitness', 0.0)):.3f} | "
                f"x={float(data.get('max_x_progress', 0.0)):.3f} | "
                f"reward={float(data.get('reward_mean', 0.0)):+.3f}",
            )
        elif topic == "run_end" and event.source == "evolution":
            print(f"  [evolution run_end] {dict(data)}")

    def reset(self) -> None:
        return

    def snapshot(self) -> dict[str, Any]:
        return {}


def _validate_paths() -> bool:
    ok = True
    for label, path in (("EmuHawk", EMUHAWK_PATH), ("ROM", ROM_PATH), ("Lua", str(LUA_SCRIPT))):
        if not Path(path).exists():
            print(f"ERROR: {label} not found: {path}")
            ok = False
    present = existing_savestates(SAVESTATE_PATHS)
    for path in SAVESTATE_PATHS:
        if path not in present:
            print(f"WARNING: savestate not found: {path}")
    if not present:
        print("WARNING: no savestates found; evaluations will boot from the ROM.")
    return ok


def main() -> int:
    print("=" * 70)
    print("NeuroForge - SMB3 neuroevolution")
    print("=" * 70)
    if not _validate_paths():
        return 1

    run_dir = RUNS_DIR / dt.datetime.now().astimezone().strftime("evolve_%Y%m%d_%H%M%S")
    bus = EventBus()
    console = _ConsoleMonitor()
    recorder = EventRecorderMonitor(run_dir)
    for topic in EventTopic:
        bus.subscribe(topic, console)
        bus.subscribe(topic, recorder)

    live_cfg = SMB3LiveFitnessConfig(
        emuhawk_path=EMUHAWK_PATH,
        rom_path=ROM_PATH,
        lua_script=str(LUA_SCRIPT),
        savestate_paths=SAVESTATE_PATHS,
        port=PORT,
        frameskip=FRAMESKIP,
        speed_percent=BIZHAWK_SPEED_PERCENT,
        max_episodes=EVAL_EPISODES,
        frames_per_episode=EVAL_FRAMES_PER_EPISODE,
        launch=LAUNCH_EMUHAWK,
    )
    evaluator = build_live_smb3_fitness_evaluator(live_cfg)
    evo_cfg = EvolutionConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        elite_count=ELITE_COUNT,
        mutation_rate=0.35,
        mutation_power=1.0,
        seed=1234,
        checkpoint_path=str(run_dir / "evolution" / "checkpoint.json"),
        resume=True,
        max_workers=EVOLUTION_WORKERS,
    )
    print(
        f"  population={POPULATION_SIZE} generations={GENERATIONS} "
        f"eval={EVAL_EPISODES}x{EVAL_FRAMES_PER_EPISODE} frames "
        f"frameskip={FRAMESKIP} speed={BIZHAWK_SPEED_PERCENT}%",
    )
    print(f"  metrics: {run_dir / 'events' / 'events.ndjson'}")

    try:
        result = EvolutionTask(evo_cfg, event_bus=bus, evaluator=evaluator).run()
        print("\n" + "=" * 70)
        print(f"  DONE: {result}")
        print(f"  Evolution checkpoint: {evo_cfg.checkpoint_path}")
        print(f"  Full metrics: {run_dir / 'events' / 'events.ndjson'}")
        print("=" * 70)
    except KeyboardInterrupt:
        print("\n  Stopped by user.")
    except BizHawkConnectionError as exc:
        print(f"\n  Emulator error: {exc}")
    finally:
        recorder.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
