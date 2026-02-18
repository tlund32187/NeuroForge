"""TrainingMonitor — records per-trial training metrics.

Captures accuracy, error, convergence status, and truth-table
confidence emitted on ``EventTopic.TRAINING_TRIAL`` and related
events.
"""

from __future__ import annotations

from collections import deque
from typing import Any

from neuroforge.contracts.monitors import EventTopic, MonitorEvent

__all__ = ["TrainingMonitor"]


class TrainingMonitor:
    """Records training events (trials, start/end).

    Snapshot layout::

        {
            "gate": "XOR",
            "converged": False,
            "trials": [...],
            "accuracy_history": [...],
            "truth_table": {
                "(0, 0)": {"expected": 0, "predicted": 0, "confidence": 0.85},
                ...
            },
            "topology": { ... },
        }
    """

    def __init__(self, *, enabled: bool = True, max_trials: int = 50_000) -> None:
        self.enabled = enabled
        self._max_trials = max_trials

        # Training state
        self._gate: str = ""
        self._converged: bool = False
        self._trials: deque[dict[str, Any]] = deque(maxlen=max_trials)
        self._accuracy_history: deque[float] = deque(maxlen=max_trials)
        self._truth_table: dict[str, dict[str, Any]] = {}
        self._topology: dict[str, Any] = {}
        self._window_size: int = 5

    # ── IMonitor implementation ─────────────────────────────────────

    def on_event(self, event: MonitorEvent) -> None:
        """Handle training-related events."""
        if not self.enabled:
            return

        if event.topic == EventTopic.TRAINING_START:
            self._gate = event.data.get("gate", "")
            self._converged = False
            self._trials.clear()
            self._accuracy_history.clear()
            self._truth_table.clear()
            # Use per_pattern_streak as the confidence window size.
            self._window_size = int(
                event.data.get("per_pattern_streak", 5)
            ) or 5

        elif event.topic == EventTopic.TOPOLOGY:
            self._topology = dict(event.data)

        elif event.topic == EventTopic.TRAINING_TRIAL:
            trial_data: dict[str, Any] = {
                "trial": event.step,
                "input": event.data.get("input", ()),
                "expected": event.data.get("expected", 0),
                "predicted": event.data.get("predicted", 0),
                "correct": event.data.get("correct", False),
                "error": event.data.get("error", 0.0),
                "out_spike_count": event.data.get("out_spike_count", 0),
            }
            self._trials.append(trial_data)

            acc = event.data.get("accuracy", 0.0)
            self._accuracy_history.append(float(acc))

            # Update per-pattern confidence (rolling window).
            inp_key = str(event.data.get("input", ()))
            if inp_key not in self._truth_table:
                self._truth_table[inp_key] = {
                    "expected": event.data.get("expected", 0),
                    "total_count": 0,
                    "last_predicted": 0,
                    "window": deque(maxlen=self._window_size),
                }
            entry = self._truth_table[inp_key]
            entry["total_count"] += 1
            entry["last_predicted"] = event.data.get("predicted", 0)
            entry["window"].append(bool(event.data.get("correct", False)))

        elif event.topic == EventTopic.TRAINING_END:
            self._converged = event.data.get("converged", False)

    def reset(self) -> None:
        """Clear all training data."""
        self._gate = ""
        self._converged = False
        self._trials.clear()
        self._accuracy_history.clear()
        self._truth_table.clear()
        self._topology.clear()

    def snapshot(self) -> dict[str, Any]:
        """Return JSON-serialisable training snapshot."""
        # Compute per-pattern confidence (from rolling window).
        tt_snap: dict[str, Any] = {}
        for k, v in self._truth_table.items():
            window: deque[bool] = v["window"]
            confidence = sum(window) / len(window) if window else 0.0
            tt_snap[k] = {
                "expected": v["expected"],
                "last_predicted": v["last_predicted"],
                "confidence": round(confidence, 4),
                "total_count": v["total_count"],
            }

        return {
            "gate": self._gate,
            "converged": self._converged,
            "total_trials": len(self._trials),
            "accuracy_history": list(self._accuracy_history),
            "truth_table": tt_snap,
            "topology": self._topology,
            "last_trials": list(self._trials)[-50:],  # last 50 for UI
        }
