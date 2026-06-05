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
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, cast

from neuroforge.applications.smb3.env import env_bool, env_float, env_int
from neuroforge.applications.smb3.fitness import (
    SMB3LiveFitnessConfig,
    build_live_smb3_fitness_evaluator,
    existing_savestates,
)
from neuroforge.applications.tasks.evolution import EvolutionTask
from neuroforge.contracts.messaging import EventTopic
from neuroforge.environments.games.clients import BizHawkConnectionError
from neuroforge.messaging.bus import EventBus
from neuroforge.neuroevolution import (
    EvolutionConfig,
    GraphReproduction,
    HyperNEATReproduction,
    InnovationRegistry,
    make_graph_seed_population,
    make_hyperneat_seed_population,
    max_connection_innovation,
)
from neuroforge.observability.events.recorder import EventRecorderMonitor
from neuroforge.perception.vision.encoding.retina import RetinaEncoder, RetinaEncoderConfig

# Machine-local paths. Edit these when your BizHawk install moves.
EMUHAWK_PATH = r"C:\BizHawk\EmuHawk.exe"
ROM_PATH = r"C:\BizHawk\ROM\Super Mario Bros. 3 (USA) (Rev 1)\Super Mario Bros. 3 (USA) (Rev 1).nes"  # noqa: E501
SAVESTATE_PATHS: tuple[str, ...] = (
    r"C:\BizHawk\States\smb3_level1.State",
)

PORT = env_int("NEUROFORGE_SMB3_PORT", 8650, min_value=1)
FRAMESKIP = env_int("NEUROFORGE_SMB3_FRAMESKIP", 4, min_value=1)
BIZHAWK_SPEED_PERCENT = env_int("NEUROFORGE_SMB3_SPEED_PERCENT", 400, min_value=1)
LAUNCH_EMUHAWK = env_bool("NEUROFORGE_SMB3_LAUNCH_EMUHAWK", True)

POPULATION_SIZE = env_int("NEUROFORGE_SMB3_EVOLVE_POPULATION", 32, min_value=2)
GENERATIONS = env_int("NEUROFORGE_SMB3_EVOLVE_GENERATIONS", 40, min_value=1)
ELITE_COUNT = env_int("NEUROFORGE_SMB3_EVOLVE_ELITES", 1, min_value=1)
EVAL_EPISODES = env_int("NEUROFORGE_SMB3_EVOLVE_EVAL_EPISODES", 1, min_value=1)
EVAL_FRAMES_PER_EPISODE = env_int(
    "NEUROFORGE_SMB3_EVOLVE_EVAL_FRAMES",
    3600,
    min_value=1,
)
# Keep live BizHawk at 1 unless each worker owns a separate emulator.
EVOLUTION_WORKERS = env_int("NEUROFORGE_SMB3_EVOLVE_WORKERS", 1, min_value=1)
EVOLUTION_SEED = env_int("NEUROFORGE_SMB3_EVOLVE_SEED", 1234)
MUTATION_RATE = env_float("NEUROFORGE_SMB3_EVOLVE_MUTATION_RATE", 0.25, min_value=0.0)
MUTATION_POWER = env_float("NEUROFORGE_SMB3_EVOLVE_MUTATION_POWER", 0.75, min_value=1e-9)
SPECIES_THRESHOLD = env_float(
    "NEUROFORGE_SMB3_EVOLVE_SPECIES_THRESHOLD",
    0.5,
    min_value=1e-9,
)
EVAL_REPEATS = env_int("NEUROFORGE_SMB3_EVOLVE_EVAL_REPEATS", 2, min_value=1)
FITNESS_PROGRESS_SCALE = env_float(
    "NEUROFORGE_SMB3_EVOLVE_PROGRESS_SCALE",
    100.0,
    min_value=0.0,
)
FITNESS_SURVIVAL_SCALE = env_float(
    "NEUROFORGE_SMB3_EVOLVE_SURVIVAL_SCALE",
    1.0,
    min_value=0.0,
)
FITNESS_DURABLE_PROGRESS_WEIGHT = env_float(
    "NEUROFORGE_SMB3_EVOLVE_DURABLE_PROGRESS_WEIGHT",
    1.0,
    min_value=0.0,
)
# "graph" evolves network STRUCTURE (NEAT-style topology invention); "hyperneat"
# evolves a CPPN that paints SPATIAL receptive fields over a retinotopic substrate;
# "policy" evolves the fixed hyperparameter vector. Structure invention is the goal.
def _resolve_genome_kind() -> str:
    kind = os.environ.get("NEUROFORGE_SMB3_EVOLVE_GENOME", "graph").strip().lower()
    return kind if kind in {"graph", "policy", "hyperneat"} else "graph"


GENOME_KIND = _resolve_genome_kind()

_REPO = Path(__file__).resolve().parents[4]
LUA_SCRIPT = _REPO / "scripts" / "bizhawk" / "neuroforge_bridge.lua"
RUNS_DIR = _REPO / "artifacts" / "runs"
EVOLUTION_RUN_DIR_ENV = "NEUROFORGE_SMB3_EVOLUTION_RUN_DIR"


class _ConsoleMonitor:
    enabled = True

    def __init__(self) -> None:
        self._started_at = time.monotonic()
        self._saw_progress = False

    def on_event(self, event: Any) -> None:
        topic = event.topic.value
        data = event.data
        if topic == "run_start" and event.source == "evolution":
            self._started_at = time.monotonic()
            self._print_resume(data)
        elif topic == "evaluation_progress" and event.source == "evolution":
            self._saw_progress = True
            self._print_progress(data)
        elif topic == "scalar" and "best_fitness" in data:
            print(
                f"  generation {int(data.get('generation', 0)) + 1} complete | "
                f"best={float(data.get('best_fitness', 0.0)):.3f} | "
                f"mean={float(data.get('mean_fitness', 0.0)):.3f} | "
                f"species={data.get('species_count')}",
                flush=True,
            )
        elif (
            topic == "training_trial"
            and event.source == "evolution"
            and not self._saw_progress
        ):
            parents = str(data.get("parent_ids", ""))
            parent_text = f" parents={parents}" if parents else ""
            print(
                f"    genome {data.get('genome_id')} | "
                f"fitness={float(data.get('fitness', 0.0)):.3f} | "
                f"x={float(data.get('max_x_progress', 0.0)):.3f} | "
                f"reward={float(data.get('reward_mean', 0.0)):+.3f}"
                f"{parent_text}",
                flush=True,
            )
        elif topic == "run_end" and event.source == "evolution":
            print(f"  [evolution run_end] {dict(data)}", flush=True)

    def _print_progress(self, data: dict[str, Any]) -> None:
        phase = str(data.get("phase", ""))
        generation = int(data.get("generation", 0)) + 1
        generations = int(data.get("generations", 0))
        individual = int(data.get("individual", 0)) + 1
        population_size = int(data.get("population_size", 0))
        genome_id = data.get("genome_id", "")
        prefix = (
            f"  gen {generation}/{generations} | "
            f"individual {individual}/{population_size} | "
            f"genome {genome_id}"
        )
        progress_text = self._progress_text(data)
        if phase == "start":
            print(f"{prefix} | evaluating | {progress_text}", flush=True)
        elif phase == "complete":
            print(
                f"{prefix} | "
                f"fitness={float(data.get('fitness', 0.0)):.3f} | "
                f"x={float(data.get('max_x_progress', 0.0)):.3f} | "
                f"reward={float(data.get('reward_mean', 0.0)):+.3f} | "
                f"{progress_text}",
                flush=True,
            )
        elif phase == "error":
            print(
                f"{prefix} | error={data.get('error', '')} | {progress_text}",
                flush=True,
            )

    def _progress_text(self, data: dict[str, Any]) -> str:
        done = int(data.get("run_evaluations", 0))
        total = int(data.get("run_total_evaluations", 0))
        if total <= 0:
            return "progress=?"
        remaining = max(0, int(data.get("remaining_evaluations", total - done)))
        percent = 100.0 * min(done, total) / total
        text = f"done={done}/{total} ({percent:4.1f}%) | remaining={remaining}"
        eta = self._eta(done, remaining)
        if eta:
            text = f"{text} | eta={eta}"
        return text

    def _eta(self, done: int, remaining: int) -> str:
        if done <= 0 or remaining <= 0:
            return ""
        elapsed = max(0.0, time.monotonic() - self._started_at)
        seconds = elapsed * remaining / done
        return _format_duration(seconds)

    def _print_resume(self, data: dict[str, Any]) -> None:
        raw_resume = data.get("resume")
        if not isinstance(raw_resume, dict):
            return
        resume = cast("dict[str, object]", raw_resume)
        if not bool(resume.get("loaded", False)):
            return
        generation = _as_int(resume.get("generation"), 0) + 1
        generations = _as_int(data.get("generations"), 0)
        population_size = _as_int(resume.get("population_size"), 0)
        evaluations = _as_int(resume.get("evaluations"), 0)
        schema_version = _as_int(resume.get("schema_version"), 0)
        print(
            f"  resumed checkpoint: gen {generation}/{generations} | "
            f"pop={population_size} | evals={evaluations} | "
            f"schema=v{schema_version} | {_as_text(resume.get('path'))}",
            flush=True,
        )
        differences = _resume_difference_text(resume.get("config_differences"))
        if differences:
            print(f"  active config changes: {differences}", flush=True)

    def reset(self) -> None:
        return

    def snapshot(self) -> dict[str, Any]:
        return {}


def _format_duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _as_text(value: object) -> str:
    return "" if value is None else str(value)


def _resume_difference_text(raw: object) -> str:
    if not isinstance(raw, dict):
        return ""
    differences = cast("dict[str, object]", raw)
    interesting = (
        "population_size",
        "generations",
        "elite_count",
        "mutation_rate",
        "mutation_power",
        "crossover_rate",
        "species_threshold",
        "seed",
        "max_workers",
        "preserve_global_best",
    )
    parts: list[str] = []
    for key in interesting:
        raw_item = differences.get(key)
        if not isinstance(raw_item, dict):
            continue
        item = cast("dict[str, object]", raw_item)
        checkpoint = _as_text(item.get("checkpoint"))
        active = _as_text(item.get("active"))
        parts.append(f"{key} {checkpoint}->{active}")
    return ", ".join(parts)


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


def _resolve_run_dir() -> Path:
    raw = os.environ.get(EVOLUTION_RUN_DIR_ENV)
    if raw:
        return Path(raw).expanduser()
    return RUNS_DIR / dt.datetime.now().astimezone().strftime("evolve_%Y%m%d_%H%M%S")


def _innovation_registry_for(checkpoint_path: str) -> InnovationRegistry:
    """Continue innovation numbers above any already in a resumed checkpoint.

    A fresh run starts at 0; resuming a structural run starts past the highest
    connection innovation so new add-node/add-connection mutations never collide
    with genes already in the population.
    """
    path = Path(checkpoint_path)
    if not path.exists():
        return InnovationRegistry()
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return InnovationRegistry()
    if not isinstance(raw, dict):
        return InnovationRegistry()
    payload = cast("dict[str, Any]", raw)
    population = payload.get("population")
    items: list[object] = cast("list[object]", population) if isinstance(population, list) else []
    payloads: list[dict[str, Any]] = [
        cast("dict[str, Any]", item) for item in items if isinstance(item, dict)
    ]
    best = payload.get("best")
    if isinstance(best, dict):
        best_genome = cast("dict[str, Any]", best).get("genome")
        if isinstance(best_genome, dict):
            payloads.append(cast("dict[str, Any]", best_genome))
    return InnovationRegistry(start=max_connection_innovation(payloads) + 1)


def main() -> int:
    print("=" * 70)
    print("NeuroForge - SMB3 neuroevolution")
    print("=" * 70)
    if not _validate_paths():
        return 1

    run_dir = _resolve_run_dir()
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
        eval_repeats=EVAL_REPEATS,
        fitness_progress_scale=FITNESS_PROGRESS_SCALE,
        fitness_survival_scale=FITNESS_SURVIVAL_SCALE,
        fitness_durable_progress_weight=FITNESS_DURABLE_PROGRESS_WEIGHT,
    )
    # HyperNEAT needs a clean retinotopic input grid; pair it with the A0 retina so
    # the substrate's input coordinates are a true 2-D field. Other kinds keep the
    # full A0+A1+A2+A3 perception stack.
    if GENOME_KIND == "hyperneat":
        def _retina_encoder() -> Any:
            return RetinaEncoder(
                RetinaEncoderConfig(
                    out_h=28, out_w=32, device=live_cfg.device, dtype=live_cfg.dtype,
                ),
            )

        evaluator = build_live_smb3_fitness_evaluator(
            live_cfg, encoder_factory=_retina_encoder,
        )
    else:
        evaluator = build_live_smb3_fitness_evaluator(live_cfg)
    evo_cfg = EvolutionConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        elite_count=ELITE_COUNT,
        mutation_rate=MUTATION_RATE,
        mutation_power=MUTATION_POWER,
        species_threshold=SPECIES_THRESHOLD,
        seed=EVOLUTION_SEED,
        checkpoint_path=str(run_dir / "evolution" / "checkpoint.json"),
        resume=True,
        max_workers=EVOLUTION_WORKERS,
    )
    # Graph/HyperNEAT modes evolve structure (both share the innovation registry so
    # resume continues past the checkpoint's markings); policy mode evolves the vector.
    reproduction: Any = None
    seed_population: Any = None
    if GENOME_KIND == "graph":
        innovations = _innovation_registry_for(str(evo_cfg.checkpoint_path))
        reproduction = GraphReproduction(evo_cfg, innovations)
        seed_population = make_graph_seed_population(innovations)
    elif GENOME_KIND == "hyperneat":
        innovations = _innovation_registry_for(str(evo_cfg.checkpoint_path))
        reproduction = HyperNEATReproduction(evo_cfg, innovations)
        seed_population = make_hyperneat_seed_population(innovations)

    print(
        f"  genome={GENOME_KIND} population={POPULATION_SIZE} generations={GENERATIONS} "
        f"eval={EVAL_EPISODES}x{EVAL_FRAMES_PER_EPISODE} frames repeats={EVAL_REPEATS} "
        f"frameskip={FRAMESKIP} speed={BIZHAWK_SPEED_PERCENT}%",
    )
    print(
        f"  evolution: species_threshold={SPECIES_THRESHOLD:.3f} "
        f"mutation_rate={MUTATION_RATE:.3f} mutation_power={MUTATION_POWER:.3f}",
    )
    print(
        f"  fitness: progress_scale={FITNESS_PROGRESS_SCALE:.1f} "
        f"survival_scale={FITNESS_SURVIVAL_SCALE:.1f} "
        f"durable_progress_weight={FITNESS_DURABLE_PROGRESS_WEIGHT:.1f}",
    )
    print(f"  metrics: {run_dir / 'events' / 'events.ndjson'}")

    try:
        result = EvolutionTask(
            evo_cfg,
            event_bus=bus,
            evaluator=evaluator,
            reproduction=reproduction,
            seed_population=seed_population,
        ).run()
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
        evaluator.close()  # tear down the shared emulator reused across genomes
        recorder.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
