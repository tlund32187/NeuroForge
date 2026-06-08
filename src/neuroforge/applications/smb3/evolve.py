"""Live SMB3 neuroevolution entrypoint.

This module wires the reusable neuroevolution engine into the live SMB3 stack:
BizHawk, HUD extraction, SMB3 rewards/termination, perception, policy decoding,
online plasticity, run artifacts, and checkpointing.

Typical launch:

    python scripts/evolve_smb3.py

Useful environment examples:

    NEUROFORGE_SMB3_EVOLVE_GENOME=hyperneat
    NEUROFORGE_SMB3_EVOLVE_PROFILE=explore
    NEUROFORGE_SMB3_EVOLVE_SEED_FROM=latest
    NEUROFORGE_SMB3_EVOLVE_WORKERS=2
    NEUROFORGE_SMB3_EVOLVE_POPULATION=32
    NEUROFORGE_SMB3_EVOLVE_GENERATIONS=40

Profiles:

``explore``
    Fast, noisy search. Shorter evaluations, one repeat, stricter early stop.

``validate``
    Slower finalist confirmation. Longer evaluations and more repeats.

``custom``
    Starts from validate-like defaults while expecting explicit env overrides.

Genome modes:

``graph``
    Evolves explicit topology and policy/training hyperparameters.

``hyperneat``
    Evolves a CPPN that paints a structured SMB3 perception substrate.

``policy``
    Evolves only the fixed policy hyperparameter vector.

Live evaluation is expensive: population * generations * episodes * frames,
plus emulator startup if ``NEUROFORGE_SMB3_LAUNCH_EMUHAWK=1``.
"""

from __future__ import annotations

import dataclasses
import datetime as dt
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Final, Literal, cast

from neuroforge.applications.smb3.env import env_bool, env_float, env_int
from neuroforge.applications.smb3.fitness import (
    SMB3LiveFitnessConfig,
    build_live_smb3_fitness_evaluator,
    build_smb3_hyperneat_encoder_factory,
    build_smb3_hyperneat_substrate,
    existing_savestates,
)
from neuroforge.applications.tasks.evolution import EvolutionTask
from neuroforge.contracts.messaging import EventTopic
from neuroforge.environments.games.clients import BizHawkConnectionError
from neuroforge.messaging.bus import EventBus
from neuroforge.neuroevolution import (
    AdaptiveSpeciation,
    EvolutionConfig,
    InnovationRegistry,
    SpeciesAwareReproduction,
    ThreadLocalFitnessEvaluatorPool,
    load_best_genome_checkpoint,
    make_graph_seed_population,
    make_hyperneat_seed_population,
    max_connection_innovation,
)
from neuroforge.observability.events.recorder import EventRecorderMonitor

EvolveProfile = Literal["explore", "validate", "custom"]
GenomeKind = Literal["graph", "policy", "hyperneat"]
SelectionMode = Literal["roulette", "rank", "tournament"]

# Machine-local paths. Edit these when your BizHawk install moves.
EMUHAWK_PATH = r"C:\BizHawk\EmuHawk.exe"
ROM_PATH = (
    r"C:\BizHawk\ROM\Super Mario Bros. 3 (USA) (Rev 1)"
    r"\Super Mario Bros. 3 (USA) (Rev 1).nes"
)
SAVESTATE_PATHS: tuple[str, ...] = (
    r"C:\BizHawk\States\smb3_level1.State",
)

PORT = env_int("NEUROFORGE_SMB3_PORT", 8650, min_value=1)
FRAMESKIP = env_int("NEUROFORGE_SMB3_FRAMESKIP", 4, min_value=1)
BIZHAWK_SPEED_PERCENT = env_int("NEUROFORGE_SMB3_SPEED_PERCENT", 400, min_value=1)
LAUNCH_EMUHAWK = env_bool("NEUROFORGE_SMB3_LAUNCH_EMUHAWK", True)


def _resolve_evolve_profile() -> EvolveProfile:
    """Return the active SMB3 evolution profile from the environment.

    Unknown values intentionally fall back to ``explore`` so a typo does not
    accidentally select the slower validation profile.
    """
    profile = os.environ.get("NEUROFORGE_SMB3_EVOLVE_PROFILE", "explore").strip().lower()
    if profile in {"explore", "validate", "custom"}:
        return cast("EvolveProfile", profile)
    return "explore"


def _resolve_selection_mode(default: SelectionMode = "tournament") -> SelectionMode:
    """Return the parent-selection strategy used during reproduction."""
    mode = os.environ.get("NEUROFORGE_SMB3_EVOLVE_SELECTION_MODE", default).strip().lower()
    if mode in {"roulette", "rank", "tournament"}:
        return cast("SelectionMode", mode)
    return default


def _env_csv(name: str, default: tuple[str, ...]) -> tuple[str, ...]:
    """Read a comma-separated environment variable."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    values = tuple(part.strip() for part in raw.split(",") if part.strip())
    return values or default


def _env_float_clamped(
    name: str,
    default: float,
    *,
    min_value: float | None = None,
    max_value: float | None = None,
) -> float:
    value = env_float(name, default, min_value=min_value)
    if max_value is not None and value > max_value:
        return default
    return value


EVOLVE_PROFILE: Final[EvolveProfile] = _resolve_evolve_profile()

# Profile field reference:
#
# eval_frames
#     Frames per episode before truncation. Explore uses short attempts so many
#     genomes can be sampled; validate gives finalists more runway.
#
# eval_repeats
#     Rollouts per genome. Higher repeats reduce noisy lucky/unlucky episodes
#     but multiply runtime.
#
# stall_patience
#     Frames with no x_progress before ending an episode as stalled.
#
# min_progress_frames / min_progress
#     Early poor-progress gate. If min_progress_frames > 0 and the genome has
#     not reached min_progress by then, the rollout ends as "min_progress".
#     x_progress is normalized level progress in [0, 1].
#
# max_decide_ticks
#     Caps SNN simulation ticks per decision during evolution. 0 disables the
#     cap. Lower values are faster but may reduce deliberation.
#
# death_penalty
#     Flat fitness cost per death fraction. This is intentionally small because
#     SMB3 has multiple lives and we want exploration, not timid survival.
#
# stall_penalty / min_progress_penalty
#     Fitness costs for non-exploratory rollouts. These should usually be harsher
#     than death because standing still teaches us less than dying farther ahead.
#
# level_clear_bonus
#     Fitness bonus for clearing the level.
#
# score_gain_scale
#     Fitness per raw HUD score point gained. It looks tiny because SMB3 score
#     values are large compared with x_progress. At 0.01:
#       +100 score -> +1 fitness
#       +1000 score -> +10 fitness
#     Raise this if you want kills, coins, bricks, and power-ups to dominate more;
#     lower it if score farming starts beating level progress.
#
# button_overuse_penalty / button_overuse_threshold
#     Penalizes action_button_mean above the threshold. This discourages holding
#     too many buttons at once while still allowing run+jump combinations.
#
# horizontal_conflict_penalty
#     Penalizes left/right conflict. A little conflict is tolerated; lots of it
#     usually means noisy dithering.
#
# novelty_weight
#     Adds behavior-diversity pressure during selection. Explore keeps this on
#     so unusual progress/action/termination patterns survive; validate keeps it
#     off so finalists are judged by objective fitness alone.
#
# Every value below can be overridden by an explicit NEUROFORGE_SMB3_EVOLVE_*
# environment variable, e.g. NEUROFORGE_SMB3_EVOLVE_SCORE_GAIN_SCALE=0.02.
_PROFILE_DEFAULTS: dict[EvolveProfile, dict[str, int | float]] = {
    "explore": {
        "eval_frames": 900,
        "eval_repeats": 1,
        "stall_patience": 240,
        "min_progress_frames": 240,
        "min_progress": 0.012,
        "max_decide_ticks": 16,
        "death_penalty": 0.5,
        "stall_penalty": 0.5,
        "min_progress_penalty": 3.0,
        "level_clear_bonus": 100.0,
        "score_gain_scale": 0.015,
        "button_overuse_penalty": 1.8,
        "button_overuse_threshold": 2.0,
        "horizontal_conflict_penalty": 3.0,
        "novelty_weight": 3.0,
    },
    "validate": {
        "eval_frames": 3600,
        "eval_repeats": 2,
        "stall_patience": 600,
        "min_progress_frames": 0,
        "min_progress": 0.0,
        "max_decide_ticks": 0,
        "death_penalty": 2.0,
        "stall_penalty": 4.0,
        "min_progress_penalty": 6.0,
        "level_clear_bonus": 100.0,
        "score_gain_scale": 0.01,
        "button_overuse_penalty": 1.5,
        "button_overuse_threshold": 2.0,
        "horizontal_conflict_penalty": 4.0,
        "novelty_weight": 0.0,
    },
    "custom": {
        "eval_frames": 3600,
        "eval_repeats": 2,
        "stall_patience": 600,
        "min_progress_frames": 0,
        "min_progress": 0.0,
        "max_decide_ticks": 0,
        "death_penalty": 2.0,
        "stall_penalty": 4.0,
        "min_progress_penalty": 6.0,
        "level_clear_bonus": 100.0,
        "score_gain_scale": 0.01,
        "button_overuse_penalty": 1.5,
        "button_overuse_threshold": 2.0,
        "horizontal_conflict_penalty": 4.0,
        "novelty_weight": 0.0,
    },
}
_PROFILE = _PROFILE_DEFAULTS[EVOLVE_PROFILE]
_MAX_LIVE_WORKERS = 4

# Evolution budget. These dominate wall-clock cost:
# population * generations * episodes * frames_per_episode * repeats.
POPULATION_SIZE = env_int("NEUROFORGE_SMB3_EVOLVE_POPULATION", 32, min_value=2)
GENERATIONS = env_int("NEUROFORGE_SMB3_EVOLVE_GENERATIONS", 40, min_value=1)
ELITE_COUNT = env_int("NEUROFORGE_SMB3_EVOLVE_ELITES", 2, min_value=1)
EVAL_EPISODES = env_int("NEUROFORGE_SMB3_EVOLVE_EVAL_EPISODES", 1, min_value=1)
EVAL_FRAMES_PER_EPISODE = env_int(
    "NEUROFORGE_SMB3_EVOLVE_EVAL_FRAMES",
    int(_PROFILE["eval_frames"]),
    min_value=1,
)

# Live BizHawk parallelism. Each worker needs its own emulator/client port.
# Keep this modest; the hard cap prevents accidental laptop-melting launches.
_DEFAULT_EVOLUTION_WORKERS = 2 if LAUNCH_EMUHAWK else 1
REQUESTED_EVOLUTION_WORKERS = env_int(
    "NEUROFORGE_SMB3_EVOLVE_WORKERS",
    _DEFAULT_EVOLUTION_WORKERS,
    min_value=1,
)
EVOLUTION_WORKERS = min(REQUESTED_EVOLUTION_WORKERS, _MAX_LIVE_WORKERS)
EVOLUTION_WORKER_PORTS = tuple(PORT + idx for idx in range(EVOLUTION_WORKERS))
EVOLUTION_SEED = env_int("NEUROFORGE_SMB3_EVOLVE_SEED", 1234)

# Structural-search pressure. Higher mutation rate/power explores faster but can
# destroy useful CPPN/graph motifs; species_threshold controls diversity.
MUTATION_RATE = env_float("NEUROFORGE_SMB3_EVOLVE_MUTATION_RATE", 0.18, min_value=0.0)
MUTATION_POWER = env_float("NEUROFORGE_SMB3_EVOLVE_MUTATION_POWER", 0.5, min_value=1e-9)
FITNESS_CROSSOVER_RATE = _env_float_clamped(
    "NEUROFORGE_SMB3_EVOLVE_CROSSOVER_RATE",
    0.85,
    min_value=0.0,
    max_value=1.0,
)
SPECIES_THRESHOLD = env_float(
    "NEUROFORGE_SMB3_EVOLVE_SPECIES_THRESHOLD",
    0.5,
    min_value=1e-9,
)
SPECIES_TARGET_MIN = env_int(
    "NEUROFORGE_SMB3_EVOLVE_SPECIES_TARGET_MIN",
    6,
    min_value=1,
)
SPECIES_TARGET_MAX = max(
    SPECIES_TARGET_MIN,
    env_int("NEUROFORGE_SMB3_EVOLVE_SPECIES_TARGET_MAX", 12, min_value=1),
)
SPECIES_THRESHOLD_ADJUSTMENT = env_float(
    "NEUROFORGE_SMB3_EVOLVE_SPECIES_THRESHOLD_ADJUSTMENT",
    0.08,
    min_value=0.0,
)

# Parent selection and behavior diversity. Tournament/rank are more stable than
# roulette for noisy live SMB3 fitness because they care about ordering rather
# than exact score magnitude. Novelty changes selection fitness in explore mode;
# validate disables it so checkpoint comparisons use objective fitness only.
SELECTION_MODE: Final[SelectionMode] = _resolve_selection_mode("tournament")
TOURNAMENT_SIZE = env_int(
    "NEUROFORGE_SMB3_EVOLVE_TOURNAMENT_SIZE",
    3,
    min_value=1,
)
RANK_SELECTION_PRESSURE = env_float(
    "NEUROFORGE_SMB3_EVOLVE_RANK_PRESSURE",
    1.7,
    min_value=1.0,
)
NOVELTY_WEIGHT = env_float(
    "NEUROFORGE_SMB3_EVOLVE_NOVELTY_WEIGHT",
    float(_PROFILE["novelty_weight"]),
    min_value=0.0,
)
NOVELTY_K = env_int(
    "NEUROFORGE_SMB3_EVOLVE_NOVELTY_K",
    5,
    min_value=1,
)
NOVELTY_ARCHIVE_SIZE = env_int(
    "NEUROFORGE_SMB3_EVOLVE_NOVELTY_ARCHIVE_SIZE",
    256,
    min_value=0,
)
NOVELTY_METRIC_KEYS = _env_csv(
    "NEUROFORGE_SMB3_EVOLVE_NOVELTY_KEYS",
    (
        "max_x_progress",
        "fitness_score_gain_score",
        "survival_frac",
        "action_button_mean",
        "action_left_frac",
        "action_right_frac",
        "action_a_frac",
        "action_b_frac",
        "termination.death",
        "termination.stall",
        "termination.min_progress",
    ),
)

# Evaluation guardrails. Explore mode uses shorter patience so bad genomes do
# not spend minutes dithering; validate mode keeps these looser.
EVAL_REPEATS = env_int(
    "NEUROFORGE_SMB3_EVOLVE_EVAL_REPEATS",
    int(_PROFILE["eval_repeats"]),
    min_value=1,
)
EVOLVE_STALL_PATIENCE = env_int(
    "NEUROFORGE_SMB3_EVOLVE_STALL_PATIENCE",
    int(_PROFILE["stall_patience"]),
    min_value=1,
)
EVOLVE_MIN_PROGRESS_FRAMES = env_int(
    "NEUROFORGE_SMB3_EVOLVE_MIN_PROGRESS_FRAMES",
    int(_PROFILE["min_progress_frames"]),
    min_value=0,
)
EVOLVE_MIN_PROGRESS = env_float(
    "NEUROFORGE_SMB3_EVOLVE_MIN_PROGRESS",
    float(_PROFILE["min_progress"]),
    min_value=0.0,
)
EVOLVE_MAX_DECIDE_TICKS = env_int(
    "NEUROFORGE_SMB3_EVOLVE_MAX_DECIDE_TICKS",
    int(_PROFILE["max_decide_ticks"]),
    min_value=0,
)
PERF_TELEMETRY = env_bool("NEUROFORGE_SMB3_PERF_TELEMETRY", False)

# Fitness shaping. Progress is still the main signal; score_gain rewards coins,
# kills, bricks, and power-ups when the calibrated HUD can read score changes.
FITNESS_PROGRESS_SCALE = env_float(
    "NEUROFORGE_SMB3_EVOLVE_PROGRESS_SCALE",
    125.0,
    min_value=0.0,
)
FITNESS_SCORE_GAIN_SCALE = env_float(
    "NEUROFORGE_SMB3_EVOLVE_SCORE_GAIN_SCALE",
    float(_PROFILE["score_gain_scale"]),
    min_value=0.0,
)
FITNESS_SURVIVAL_SCALE = env_float(
    "NEUROFORGE_SMB3_EVOLVE_SURVIVAL_SCALE",
    1.0,
    min_value=0.0,
)
FITNESS_DURABLE_PROGRESS_WEIGHT = env_float(
    "NEUROFORGE_SMB3_EVOLVE_DURABLE_PROGRESS_WEIGHT",
    1.5,
    min_value=0.0,
)
FITNESS_DEATH_PENALTY = env_float(
    "NEUROFORGE_SMB3_EVOLVE_DEATH_PENALTY",
    float(_PROFILE["death_penalty"]),
    min_value=0.0,
)
FITNESS_STALL_PENALTY = env_float(
    "NEUROFORGE_SMB3_EVOLVE_STALL_PENALTY",
    float(_PROFILE["stall_penalty"]),
    min_value=0.0,
)
FITNESS_MIN_PROGRESS_PENALTY = env_float(
    "NEUROFORGE_SMB3_EVOLVE_MIN_PROGRESS_PENALTY",
    float(_PROFILE["min_progress_penalty"]),
    min_value=0.0,
)
FITNESS_LEVEL_CLEAR_BONUS = env_float(
    "NEUROFORGE_SMB3_EVOLVE_LEVEL_CLEAR_BONUS",
    float(_PROFILE["level_clear_bonus"]),
    min_value=0.0,
)
FITNESS_BUTTON_OVERUSE_PENALTY = env_float(
    "NEUROFORGE_SMB3_EVOLVE_BUTTON_OVERUSE_PENALTY",
    float(_PROFILE["button_overuse_penalty"]),
    min_value=0.0,
)
FITNESS_BUTTON_OVERUSE_THRESHOLD = env_float(
    "NEUROFORGE_SMB3_EVOLVE_BUTTON_OVERUSE_THRESHOLD",
    float(_PROFILE["button_overuse_threshold"]),
    min_value=0.0,
)
FITNESS_HORIZONTAL_CONFLICT_PENALTY = env_float(
    "NEUROFORGE_SMB3_EVOLVE_HORIZONTAL_CONFLICT_PENALTY",
    float(_PROFILE["horizontal_conflict_penalty"]),
    min_value=0.0,
)
# "graph" evolves network STRUCTURE (NEAT-style topology invention); "hyperneat"
# evolves a CPPN that paints SPATIAL receptive fields over a retinotopic substrate;
# "policy" evolves the fixed hyperparameter vector. Structure invention is the goal.
def _resolve_genome_kind() -> GenomeKind:
    """Return the genome family selected by ``NEUROFORGE_SMB3_EVOLVE_GENOME``."""
    kind = os.environ.get("NEUROFORGE_SMB3_EVOLVE_GENOME", "graph").strip().lower()
    if kind in {"graph", "policy", "hyperneat"}:
        return cast("GenomeKind", kind)
    return "graph"


GENOME_KIND: Final[GenomeKind] = _resolve_genome_kind()
SMB3_HYPERNEAT_SUBSTRATE = build_smb3_hyperneat_substrate()

# Optional: seed the initial population from a previous run's best genome instead
# of random init. Point this at a prior run's evolution/checkpoint.json. Genome 0
# is an exact clone of that best brain; the rest are mutated variants for
# diversity. Only the structure-evolving modes (graph/hyperneat) support seeding.
SEED_FROM = os.environ.get("NEUROFORGE_SMB3_EVOLVE_SEED_FROM", "").strip()
# Sentinels for SEED_FROM that mean "find the most recent run of this genome
# kind under artifacts/runs" instead of an explicit checkpoint path.
_SEED_FROM_LATEST = {"latest", "auto"}
_GENOME_TYPE_BY_KIND = {
    "graph": "GraphGenome",
    "hyperneat": "HyperNEATGenome",
    "policy": "PolicyGenome",
}

_REPO = Path(__file__).resolve().parents[4]
LUA_SCRIPT = _REPO / "scripts" / "bizhawk" / "neuroforge_bridge.lua"
RUNS_DIR = _REPO / "artifacts" / "runs"
EVOLUTION_RUN_DIR_ENV = "NEUROFORGE_SMB3_EVOLUTION_RUN_DIR"


class _ConsoleMonitor:
    """Small event monitor that prints live generation/individual progress."""

    enabled = True

    def __init__(self) -> None:
        self._started_at = time.monotonic()
        self._saw_progress = False

    def on_event(self, event: Any) -> None:
        """Handle evolution events emitted by :class:`EvolutionTask`."""
        topic = event.topic.value
        data = event.data
        if topic == "run_start" and event.source == "evolution":
            self._started_at = time.monotonic()
            self._print_resume(data)
        elif topic == "evaluation_progress" and event.source == "evolution":
            self._saw_progress = True
            self._print_progress(data)
        elif topic == "scalar" and "best_fitness" in data:
            generation_selection = float(data.get("best_fitness", 0.0))
            generation_objective = float(
                data.get("best_objective_fitness", generation_selection)
            )
            run_best = float(data.get("run_best_fitness", generation_selection))
            best_species_uid = _as_int(data.get("best_species_uid"), -1)
            best_species_text = (
                f" | best_species=s{best_species_uid}"
                if best_species_uid >= 0
                else ""
            )
            print(
                f"  generation {int(data.get('generation', 0)) + 1} complete | "
                f"gen_obj={generation_objective:.3f} | "
                f"gen_sel={generation_selection:.3f} | "
                f"run_best={run_best:.3f} | "
                f"mean={float(data.get('mean_fitness', 0.0)):.3f} | "
                f"species={data.get('species_count')} | "
                f"genome={self._format_genome(data)}"
                f"{self._fitness_breakdown_text(data)}"
                f"{best_species_text}",
                flush=True,
            )
        elif (
            topic == "training_trial"
            and event.source == "evolution"
            and not self._saw_progress
        ):
            parents = str(data.get("parent_ids", ""))
            parent_text = f" | parents={parents}" if parents else ""
            reproduction_text = self._reproduction_text(parents)
            species_text = self._species_text(data)
            print(
                f"    genome {data.get('genome_id')} | "
                f"fitness={float(data.get('fitness', 0.0)):.3f} | "
                f"x={float(data.get('max_x_progress', 0.0)):.3f} | "
                f"reward={float(data.get('reward_mean', 0.0)):+.3f}"
                f"{self._fitness_breakdown_text(data)}"
                f"{reproduction_text}{parent_text}{species_text}",
                flush=True,
            )
        elif topic == "run_end" and event.source == "evolution":
            print(f"  [evolution run_end] {dict(data)}", flush=True)

    def _print_progress(self, data: dict[str, Any]) -> None:
        """Print one in-flight evaluation start/completion/error line."""
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
            return
        elif phase == "complete":
            timing = self._timing_text(data)
            parents = str(data.get("parent_ids", ""))
            parent_text = f" | parents={parents}" if parents else ""
            reproduction_text = self._reproduction_text(parents)
            species_text = self._species_text(data)
            print(
                f"{prefix} | "
                f"fitness={float(data.get('fitness', 0.0)):.3f} | "
                f"x={float(data.get('max_x_progress', 0.0)):.3f} | "
                f"reward={float(data.get('reward_mean', 0.0)):+.3f} | "
                f"{timing}{progress_text}{self._fitness_breakdown_text(data)}"
                f"{reproduction_text}{parent_text}{species_text}",
                flush=True,
            )
        elif phase == "error":
            print(
                f"{prefix} | error={data.get('error', '')} | {progress_text}",
                flush=True,
            )

    def _progress_text(self, data: dict[str, Any]) -> str:
        """Return total run progress and ETA text for one progress event."""
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

    def _timing_text(self, data: dict[str, Any]) -> str:
        """Return cache/timing/fps text for a completed evaluation."""
        if float(data.get("evaluation_cache_hit", 0.0)) >= 1.0:
            return "cache=hit | "
        seconds = float(data.get("evaluation_wall_seconds", 0.0))
        fps = float(data.get("evaluation_fps", 0.0))
        parts: list[str] = []
        if seconds > 0.0:
            parts.append(f"time={_format_duration(seconds)}")
        if fps > 0.0:
            parts.append(f"fps={fps:.1f}")
        return f"{' | '.join(parts)} | " if parts else ""

    @staticmethod
    def _reproduction_text(parents: str) -> str:
        """Return compact reproduction mode text from serialized parent IDs."""
        parent_ids = [item.strip() for item in parents.split(",") if item.strip()]
        if len(set(parent_ids)) >= 2:
            return " | op=crossover"
        if parent_ids:
            return " | op=mutation"
        return ""

    @staticmethod
    def _fitness_breakdown_text(data: dict[str, Any]) -> str:
        """Return a compact breakdown of the fitness contributions."""
        parts = [
            f"reward={float(data.get('reward_mean', 0.0)):+.3f}",
            f"prog={float(data.get('fitness_progress_score', 0.0)):+.3f}",
            f"score={float(data.get('fitness_score_gain_score', 0.0)):+.3f}",
            f"surv={float(data.get('fitness_survival_score', 0.0)):+.3f}",
            f"dur={float(data.get('fitness_durable_progress_bonus', 0.0)):+.3f}",
            f"term={float(data.get('fitness_terminal_score', 0.0)):+.3f}",
            f"btn={float(data.get('fitness_button_overuse_score', 0.0)):+.3f}",
            f"conf={float(data.get('fitness_horizontal_conflict_score', 0.0)):+.3f}",
        ]
        novelty_score = float(data.get("novelty_score", 0.0))
        novelty_weight = float(data.get("novelty_weight", 0.0))
        novelty_bonus = float(data.get("fitness_novelty_bonus", 0.0))
        parts.append(
            f"nov={novelty_score:+.3f}@w{novelty_weight:.2f}=>{novelty_bonus:+.3f}"
        )
        return " | fit[" + ", ".join(parts) + "]"

    @staticmethod
    def _species_text(data: dict[str, Any]) -> str:
        """Return stable species text with persistent uid and local id."""
        species_uid = _as_int(data.get("species_uid"), -1)
        species_id = _as_int(data.get("species_id"), -1)
        if species_uid >= 0 and species_id >= 0:
            return f" | species=s{species_uid} (id={species_id})"
        if species_uid >= 0:
            return f" | species=s{species_uid}"
        if species_id >= 0:
            return f" | species={species_id}"
        return ""

    @staticmethod
    def _format_genome(data: dict[str, Any]) -> str:
        genome = data.get("best_genome")
        if genome is None:
            return str(data.get("best_genome_id", ""))
        genome_id = str(data.get("best_genome_id", ""))
        genome_type = ""
        if isinstance(genome, dict):
            genome_dict = cast("dict[str, Any]", genome)
            genome_type = str(genome_dict.get("type", ""))
        if genome_type:
            return f"{genome_id} ({genome_type})".strip()
        return genome_id

    def _eta(self, done: int, remaining: int) -> str:
        """Estimate remaining wall time from completed evaluations."""
        if done <= 0 or remaining <= 0:
            return ""
        elapsed = max(0.0, time.monotonic() - self._started_at)
        seconds = elapsed * remaining / done
        return _format_duration(seconds)

    def _print_resume(self, data: dict[str, Any]) -> None:
        """Print checkpoint-resume metadata when a run resumes."""
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
    """Format seconds as a compact console duration."""
    seconds = max(0, int(round(seconds)))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes:02d}m"
    if minutes:
        return f"{minutes}m{secs:02d}s"
    return f"{secs}s"


def _as_int(value: object, default: int = 0) -> int:
    """Best-effort integer conversion for event/checkpoint metadata."""
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
    """Return a display-safe string for optional metadata values."""
    return "" if value is None else str(value)


def _resume_difference_text(raw: object) -> str:
    """Summarize important active-vs-checkpoint config differences."""
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
        "selection_mode",
        "tournament_size",
        "rank_selection_pressure",
        "novelty_weight",
        "novelty_metric_keys",
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
    """Check machine-local emulator, ROM, Lua bridge, and savestate paths."""
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
    """Return the run artifact directory, honoring the override env var."""
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


def _checkpoint_best_type(path: Path) -> str:
    """Return the genome ``type`` tag of a checkpoint's best.

    Returns an empty string when the file is missing, unreadable, or does not
    contain a best genome payload.
    """
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    if not isinstance(raw, dict):
        return ""
    best = cast("dict[str, Any]", raw).get("best")
    if not isinstance(best, dict):
        return ""
    genome = cast("dict[str, Any]", best).get("genome")
    if not isinstance(genome, dict):
        return ""
    return str(cast("dict[str, Any]", genome).get("type", "policy"))


def _latest_seed_checkpoint(kind: str) -> Path | None:
    """Newest checkpoint under RUNS_DIR whose best genome matches *kind*.

    Lets the launch button seed from "latest" and always pick up the most recent
    run of the same genome kind, so repeated clicks chain forward from the prior
    result. Filtering by kind avoids seeding a hyperneat run from a graph run.
    """
    candidates = sorted(
        (path for path in RUNS_DIR.glob("**/evolution/checkpoint.json") if path.is_file()),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidates:
        if _checkpoint_best_type(path) == kind:
            return path
    return None


def _seed_population_from_best(
    best_genome: Any,
    innovations: InnovationRegistry,
    *,
    rate: float,
    power: float,
    target_substrate: Any | None = None,
) -> Any:
    """Return a seed factory that starts the search from a previous best genome.

    Genome 0 is an exact clone of *best_genome* so the proven brain is always in
    the starting pool; the remaining genomes are mutated variants of it, giving
    the search local diversity to explore from instead of restarting from random
    init. Structural mutations draw fresh numbers from *innovations*, which the
    caller seeds past the source checkpoint so they never collide with existing
    genes.
    """

    source = _adapt_seed_genome(best_genome, innovations, target_substrate)

    def seed(size: int, rng: Any) -> list[Any]:
        population = [source.as_offspring(child_id="g0_0", generation=0)]
        population.extend(
            source.mutate(
                child_id=f"g0_{idx}",
                generation=0,
                rng=rng,
                rate=rate,
                power=power,
                innovations=innovations,
            )
            for idx in range(1, size)
        )
        return population

    return seed


def _adapt_seed_genome(
    genome: Any,
    innovations: InnovationRegistry,
    target_substrate: Any | None,
) -> Any:
    """Move a seed genome onto the target substrate when it supports migration.

    HyperNEAT genomes use this when older best checkpoints need extra CPPN inputs
    for the structured SMB3 perception substrate. Graph genomes do not need a
    substrate and are returned unchanged.
    """
    if target_substrate is None:
        return genome
    adapt = getattr(genome, "with_substrate", None)
    if not callable(adapt):
        return genome
    return adapt(target_substrate, innovations=innovations)


def _innovation_mutation_hook(innovations: InnovationRegistry) -> Any:
    """Return a SpeciesAwareReproduction mutation hook for structural genomes."""

    def mutate(
        child: Any,
        child_id: str,
        generation: int,
        rng: Any,
        rate: float,
        power: float,
    ) -> Any:
        return child.mutate(
            child_id=child_id,
            generation=generation,
            rng=rng,
            rate=rate,
            power=power,
            innovations=innovations,
        )

    return mutate


def _build_evaluator_for_worker(
    live_cfg: SMB3LiveFitnessConfig,
    *,
    run_dir: Path,
    worker_index: int,
) -> Any:
    """Build one live evaluator bound to a unique BizHawk worker port.

    Each parallel worker owns its evaluator, emulator client, bridge error log,
    and port. HyperNEAT workers also receive the structured SMB3 perception
    encoder so the genome substrate and runtime input size match.
    """
    worker_dir = run_dir / "bizhawk"
    worker_dir.mkdir(parents=True, exist_ok=True)
    worker_cfg = dataclasses.replace(
        live_cfg,
        port=live_cfg.port + worker_index,
        bridge_error_path=str(worker_dir / f"bridge_error_worker_{worker_index}.log"),
    )
    if GENOME_KIND == "hyperneat":
        return build_live_smb3_fitness_evaluator(
            worker_cfg,
            encoder_factory=build_smb3_hyperneat_encoder_factory(
                device=worker_cfg.device,
                dtype=worker_cfg.dtype,
            ),
        )
    return build_live_smb3_fitness_evaluator(worker_cfg)


def main() -> int:
    """Run live SMB3 evolution and write artifacts/checkpoints.

    Returns a process-style status code: ``0`` for normal completion or a handled
    user stop, ``1`` when required local paths or seed checkpoints are invalid.
    """
    print("=" * 70)
    print("NeuroForge - SMB3 neuroevolution")
    print("=" * 70)
    if not _validate_paths():
        return 1

    # Resolve + validate any seed-from-best checkpoint before launching the
    # emulator, so a bad SEED_FROM fails fast instead of after EmuHawk startup.
    # SEED_FROM is either an explicit checkpoint path or "latest"/"auto", which
    # picks the most recent run of this genome kind under artifacts/runs.
    seed_best: Any = None
    seed_path = ""
    if SEED_FROM:
        if GENOME_KIND not in {"graph", "hyperneat"}:
            print(
                "WARNING: NEUROFORGE_SMB3_EVOLVE_SEED_FROM is set but "
                f"genome={GENOME_KIND}; seeding is only supported for "
                "graph/hyperneat. Starting from defaults.",
            )
        else:
            if SEED_FROM.lower() in _SEED_FROM_LATEST:
                latest = _latest_seed_checkpoint(GENOME_KIND)
                if latest is None:
                    print(
                        f"ERROR: no prior {GENOME_KIND} run found under "
                        f"{RUNS_DIR} to seed from.",
                    )
                    return 1
                seed_path = str(latest)
                print(f"  seed source: latest {GENOME_KIND} run -> {seed_path}")
            elif not Path(SEED_FROM).exists():
                print(f"ERROR: seed checkpoint not found: {SEED_FROM}")
                return 1
            else:
                seed_path = SEED_FROM
            seed_best = load_best_genome_checkpoint(seed_path)
            actual_type = type(seed_best.genome).__name__
            if actual_type != _GENOME_TYPE_BY_KIND[GENOME_KIND]:
                print(
                    f"ERROR: seed checkpoint holds a {actual_type} but "
                    f"genome={GENOME_KIND}; point SEED_FROM at a matching "
                    f"{GENOME_KIND} run.",
                )
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
        perf_telemetry=PERF_TELEMETRY,
        stall_patience=EVOLVE_STALL_PATIENCE,
        min_progress_frames=EVOLVE_MIN_PROGRESS_FRAMES,
        min_progress=EVOLVE_MIN_PROGRESS,
        max_decide_ticks=EVOLVE_MAX_DECIDE_TICKS,
        fitness_progress_scale=FITNESS_PROGRESS_SCALE,
        fitness_score_gain_scale=FITNESS_SCORE_GAIN_SCALE,
        fitness_survival_scale=FITNESS_SURVIVAL_SCALE,
        fitness_durable_progress_weight=FITNESS_DURABLE_PROGRESS_WEIGHT,
        fitness_death_penalty=FITNESS_DEATH_PENALTY,
        fitness_stall_penalty=FITNESS_STALL_PENALTY,
        fitness_min_progress_penalty=FITNESS_MIN_PROGRESS_PENALTY,
        fitness_level_clear_bonus=FITNESS_LEVEL_CLEAR_BONUS,
        fitness_button_overuse_penalty=FITNESS_BUTTON_OVERUSE_PENALTY,
        fitness_button_overuse_threshold=FITNESS_BUTTON_OVERUSE_THRESHOLD,
        fitness_horizontal_conflict_penalty=FITNESS_HORIZONTAL_CONFLICT_PENALTY,
    )
    if EVOLUTION_WORKERS > 1:
        evaluator = ThreadLocalFitnessEvaluatorPool(
            lambda worker_index: _build_evaluator_for_worker(
                live_cfg,
                run_dir=run_dir,
                worker_index=worker_index,
            ),
            max_workers=EVOLUTION_WORKERS,
        )
    else:
        evaluator = _build_evaluator_for_worker(live_cfg, run_dir=run_dir, worker_index=0)
    evo_cfg = EvolutionConfig(
        population_size=POPULATION_SIZE,
        generations=GENERATIONS,
        elite_count=ELITE_COUNT,
        mutation_rate=MUTATION_RATE,
        mutation_power=MUTATION_POWER,
        crossover_rate=FITNESS_CROSSOVER_RATE,
        species_threshold=SPECIES_THRESHOLD,
        seed=EVOLUTION_SEED,
        checkpoint_path=str(run_dir / "evolution" / "checkpoint.json"),
        resume=True,
        max_workers=EVOLUTION_WORKERS,
        selection_mode=SELECTION_MODE,
        tournament_size=TOURNAMENT_SIZE,
        rank_selection_pressure=RANK_SELECTION_PRESSURE,
        novelty_weight=NOVELTY_WEIGHT,
        novelty_k=NOVELTY_K,
        novelty_archive_size=NOVELTY_ARCHIVE_SIZE,
        novelty_metric_keys=NOVELTY_METRIC_KEYS,
    )
    species_target_min = min(SPECIES_TARGET_MIN, POPULATION_SIZE)
    species_target_max = min(max(species_target_min, SPECIES_TARGET_MAX), POPULATION_SIZE)
    speciation = AdaptiveSpeciation(
        threshold=SPECIES_THRESHOLD,
        target_min=species_target_min,
        target_max=species_target_max,
        adjustment=SPECIES_THRESHOLD_ADJUSTMENT,
    )
    # Graph/HyperNEAT modes evolve structure (both share the innovation registry so
    # resume continues past the checkpoint's markings); policy mode evolves the vector.
    # With NEUROFORGE_SMB3_EVOLVE_SEED_FROM the structural modes start their initial
    # population from a previous run's best genome instead of random init.
    reproduction: Any = SpeciesAwareReproduction(evo_cfg)
    seed_population: Any = None
    if GENOME_KIND in {"graph", "hyperneat"}:
        # When seeding from a prior best, number innovations past THAT checkpoint
        # (not the empty new-run path) so structural mutations never collide.
        innovation_source = seed_path if seed_best is not None else str(evo_cfg.checkpoint_path)
        innovations = _innovation_registry_for(innovation_source)
        reproduction = SpeciesAwareReproduction(
            evo_cfg,
            mutate_child=_innovation_mutation_hook(innovations),
        )
        if seed_best is not None:
            target_substrate = (
                SMB3_HYPERNEAT_SUBSTRATE if GENOME_KIND == "hyperneat" else None
            )
            seed_population = _seed_population_from_best(
                seed_best.genome,
                innovations,
                rate=MUTATION_RATE,
                power=MUTATION_POWER,
                target_substrate=target_substrate,
            )
            print(
                f"  seeding from best genome: fitness={seed_best.fitness:.3f} "
                f"gen={seed_best.generation} type={type(seed_best.genome).__name__} "
                f"| {seed_path}",
            )
        elif GENOME_KIND == "graph":
            seed_population = make_graph_seed_population(innovations)
        else:
            seed_population = make_hyperneat_seed_population(
                innovations,
                substrate=SMB3_HYPERNEAT_SUBSTRATE,
            )

    print(
        f"  genome={GENOME_KIND} population={POPULATION_SIZE} generations={GENERATIONS} "
        f"eval={EVAL_EPISODES}x{EVAL_FRAMES_PER_EPISODE} frames repeats={EVAL_REPEATS} "
        f"frameskip={FRAMESKIP} speed={BIZHAWK_SPEED_PERCENT}%",
    )
    print(
        f"  profile={EVOLVE_PROFILE} workers={EVOLUTION_WORKERS}"
        f"/{REQUESTED_EVOLUTION_WORKERS} ports={','.join(str(p) for p in EVOLUTION_WORKER_PORTS)} "
        f"perf_telemetry={PERF_TELEMETRY}",
    )
    print(
        f"  evolution: species_threshold={SPECIES_THRESHOLD:.3f} "
        f"target_species={species_target_min}-{species_target_max} "
        f"threshold_adjust={SPECIES_THRESHOLD_ADJUSTMENT:.3f} "
        f"mutation_rate={MUTATION_RATE:.3f} mutation_power={MUTATION_POWER:.3f} "
        f"crossover_rate={FITNESS_CROSSOVER_RATE:.3f}",
    )
    print(
        f"  selection: mode={SELECTION_MODE} tournament_size={TOURNAMENT_SIZE} "
        f"rank_pressure={RANK_SELECTION_PRESSURE:.2f} "
        f"novelty_weight={NOVELTY_WEIGHT:.2f} novelty_k={NOVELTY_K} "
        f"archive={NOVELTY_ARCHIVE_SIZE}",
    )
    if NOVELTY_METRIC_KEYS:
        print(f"  novelty keys: {', '.join(NOVELTY_METRIC_KEYS)}")
    print(
        f"  guardrails: stall_patience={EVOLVE_STALL_PATIENCE} "
        f"min_progress={EVOLVE_MIN_PROGRESS:.4f}@{EVOLVE_MIN_PROGRESS_FRAMES}f "
        f"max_decide_ticks={EVOLVE_MAX_DECIDE_TICKS or 'off'}",
    )
    print(
        f"  fitness: progress_scale={FITNESS_PROGRESS_SCALE:.1f} "
        f"score_gain_scale={FITNESS_SCORE_GAIN_SCALE:.3f} "
        f"survival_scale={FITNESS_SURVIVAL_SCALE:.1f} "
        f"durable_progress_weight={FITNESS_DURABLE_PROGRESS_WEIGHT:.1f}",
    )
    print(
        f"  fitness weights: progress={FITNESS_PROGRESS_SCALE:.1f}x max_x "
        f"score_gain={FITNESS_SCORE_GAIN_SCALE:.3f}x score_gain_total "
        f"survival={FITNESS_SURVIVAL_SCALE:.1f}x survival_frac "
        f"durable={FITNESS_DURABLE_PROGRESS_WEIGHT:.1f}x progress_score*survival_frac",
    )
    print(
        f"  fitness penalties: death={FITNESS_DEATH_PENALTY:.1f} "
        f"stall={FITNESS_STALL_PENALTY:.1f} "
        f"min_progress={FITNESS_MIN_PROGRESS_PENALTY:.1f} "
        f"button_overuse={FITNESS_BUTTON_OVERUSE_PENALTY:.1f}"
        f">@{FITNESS_BUTTON_OVERUSE_THRESHOLD:.1f} "
        f"lr_conflict={FITNESS_HORIZONTAL_CONFLICT_PENALTY:.1f} "
        f"clear_bonus={FITNESS_LEVEL_CLEAR_BONUS:.1f}",
    )
    if GENOME_KIND == "hyperneat":
        print(
            "  perception: A1/A2/A3 structured HyperNEAT substrate "
            f"inputs={SMB3_HYPERNEAT_SUBSTRATE.input_count()} "
            f"query_dim={SMB3_HYPERNEAT_SUBSTRATE.query_dim()}",
        )
    print(f"  metrics: {run_dir / 'events' / 'events.ndjson'}")

    try:
        result = EvolutionTask(
            evo_cfg,
            event_bus=bus,
            evaluator=evaluator,
            speciation=speciation,
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
