"""Phase 6 multi-seed stability harness."""

from __future__ import annotations

import math
from dataclasses import fields, is_dataclass
from typing import Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent
from neuroforge.core.determinism.mode import DeterminismConfig, apply_determinism
from neuroforge.monitors.bus import EventBus
from neuroforge.monitors.stability_monitor import StabilityConfig, StabilityMonitor
from neuroforge.monitors.trial_stats_monitor import TrialStatsMonitor

__all__ = ["run_multi_seed"]

_FLAG_KEYS: tuple[str, ...] = (
    "stab_nan_inf",
    "stab_weight_explode",
    "stab_rate_saturation",
    "stab_oscillation",
    "stab_stagnation",
)
_CRITICAL_FLAG_KEYS: tuple[str, ...] = ("stab_nan_inf", "stab_weight_explode")


def _parse_flag(value: object) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(float(value) != 0.0)
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "t", "yes", "y"}:
            return 1
        if norm in {"0", "false", "f", "no", "n", ""}:
            return 0
        try:
            return int(float(norm) != 0.0)
        except ValueError:
            return 0
    return 0


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _percentile(values: list[int], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(ordered[lo])
    frac = pos - lo
    return float(ordered[lo] * (1.0 - frac) + ordered[hi] * frac)


def _filter_config_kwargs(cfg_type: type[Any], raw: dict[str, Any]) -> dict[str, Any]:
    if not is_dataclass(cfg_type):
        return dict(raw)
    allowed = {f.name for f in fields(cfg_type)}
    return {k: v for k, v in raw.items() if k in allowed}


def _base_bool(config: dict[str, Any], key: str, default: bool) -> bool:
    if key not in config:
        return default
    value = config[key]
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        norm = value.strip().lower()
        if norm in {"1", "true", "t", "yes", "y"}:
            return True
        if norm in {"0", "false", "f", "no", "n", ""}:
            return False
    return bool(value)


def _base_int(config: dict[str, Any], key: str, default: int) -> int:
    if key not in config:
        return default
    value = config[key]
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(round(value))
    if isinstance(value, str):
        try:
            return int(value.strip())
        except ValueError:
            return default
    return default


class _HarnessCollector:
    """Collect final scalar accuracy + stability flags for one seed run."""

    def __init__(self) -> None:
        self.enabled = True
        self.last_accuracy: float | None = None
        self.last_flags: dict[str, int] = {key: 0 for key in _FLAG_KEYS}

    def on_event(self, event: MonitorEvent) -> None:
        if not self.enabled:
            return
        if event.topic not in (
            EventTopic.TRAINING_TRIAL,
            EventTopic.SCALAR,
            EventTopic.TRAINING_END,
        ):
            return
        for key in _FLAG_KEYS:
            if key in event.data:
                self.last_flags[key] = _parse_flag(event.data.get(key))
        accuracy = _coerce_float(event.data.get("accuracy"))
        if accuracy is not None:
            self.last_accuracy = accuracy

    def reset(self) -> None:
        self.last_accuracy = None
        self.last_flags = {key: 0 for key in _FLAG_KEYS}

    def snapshot(self) -> dict[str, Any]:
        return {
            "last_accuracy": self.last_accuracy,
            "last_flags": dict(self.last_flags),
        }


def run_multi_seed(
    task_name: str,
    seeds: list[int],
    base_config: dict[str, Any],
    *,
    deterministic: bool = True,
) -> dict[str, Any]:
    """Run a task across multiple seeds and summarise stability/convergence."""
    if not seeds:
        msg = "seeds must not be empty"
        raise ValueError(msg)
    if task_name not in {"multi_gate", "logic_gate"}:
        msg = f"Unsupported task_name={task_name!r}; expected 'multi_gate' or 'logic_gate'"
        raise ValueError(msg)

    trial_stats_enabled = _base_bool(base_config, "trial_stats", True)
    stability_every = max(1, _base_int(base_config, "stability_every", 5))
    fail_fast = _base_bool(base_config, "fail_fast", False)

    converge_unit = "epochs" if task_name == "multi_gate" else "trials"
    per_seed: list[dict[str, Any]] = []
    converge_values: list[int] = []
    total_flag_counts = {key: 0 for key in _FLAG_KEYS}
    successful_flag_counts = {key: 0 for key in _FLAG_KEYS}
    passed = 0

    for seed in seeds:
        apply_determinism(
            DeterminismConfig(
                seed=int(seed),
                deterministic=deterministic,
                benchmark=False,
                warn_only=True,
            )
        )

        bus = EventBus()
        if trial_stats_enabled:
            bus.subscribe_all(TrialStatsMonitor(enabled=True))
        bus.subscribe_all(
            StabilityMonitor(
                StabilityConfig(
                    enabled=True,
                    check_every_n_trials=stability_every,
                    fail_fast=fail_fast,
                )
            )
        )
        collector = _HarnessCollector()
        bus.subscribe_all(collector)

        converged = False
        trials: int | None = None
        epochs: int | None = None
        final_accuracy: float | None = None
        converge_value: int | None = None
        err_msg: str | None = None

        try:
            if task_name == "multi_gate":
                from neuroforge.tasks.multi_gate import MultiGateConfig, MultiGateTask

                cfg_kwargs = _filter_config_kwargs(MultiGateConfig, base_config)
                cfg_kwargs["seed"] = int(seed)
                cfg_multi = MultiGateConfig(**cfg_kwargs)
                result_multi = MultiGateTask(cfg_multi, event_bus=bus).run()
                converged = bool(result_multi.converged)
                trials = int(result_multi.trials)
                epochs = int(result_multi.epochs)
                final_accuracy = (
                    float(result_multi.accuracy_history[-1])
                    if result_multi.accuracy_history
                    else collector.last_accuracy
                )
                if converged:
                    converge_value = int(result_multi.epochs)
            else:
                from neuroforge.tasks.logic_gates import LogicGateConfig, LogicGateTask

                cfg_kwargs = _filter_config_kwargs(LogicGateConfig, base_config)
                cfg_kwargs["seed"] = int(seed)
                cfg_logic = LogicGateConfig(**cfg_kwargs)
                result_logic = LogicGateTask(cfg_logic, event_bus=bus).run()
                converged = bool(result_logic.converged)
                trials = int(result_logic.trials)
                final_accuracy = (
                    float(result_logic.accuracy_history[-1])
                    if result_logic.accuracy_history
                    else collector.last_accuracy
                )
                if converged:
                    converge_value = int(result_logic.trials)
        except RuntimeError as exc:
            err_msg = str(exc)

        flags = dict(collector.last_flags)
        for key in _FLAG_KEYS:
            total_flag_counts[key] += flags.get(key, 0)
        if converged:
            passed += 1
            for key in _FLAG_KEYS:
                successful_flag_counts[key] += flags.get(key, 0)
        if converge_value is not None:
            converge_values.append(converge_value)

        per_seed.append(
            {
                "seed": int(seed),
                "converged": converged,
                "trials": trials,
                "epochs": epochs,
                "epochs_to_converge": int(epochs) if converged and epochs is not None else None,
                "trials_to_converge": int(trials) if converged and trials is not None else None,
                "final_accuracy": final_accuracy,
                "stability_flags": flags,
                "error": err_msg,
            }
        )

    pass_rate = float(passed / len(seeds))
    critical_successful = {
        key: successful_flag_counts[key] for key in _CRITICAL_FLAG_KEYS
    }

    return {
        "task": task_name,
        "deterministic": deterministic,
        "seeds": [int(seed) for seed in seeds],
        "n_runs": len(seeds),
        "n_passed": passed,
        "pass_rate": pass_rate,
        "converge_unit": converge_unit,
        "median_converge": _percentile(converge_values, 0.5),
        "p90_converge": _percentile(converge_values, 0.9),
        "total_flag_counts": total_flag_counts,
        "successful_flag_counts": successful_flag_counts,
        "critical_flag_counts_successful": critical_successful,
        "runs": per_seed,
    }
