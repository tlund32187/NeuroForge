# NeuroForge Architecture

A map of how NeuroForge fits together and how to extend it. Start here if you
are new to the codebase; pair it with the phase notes in the top-level
[README](../README.md).

## 1. The big picture

NeuroForge simulates **spiking neural networks** (leaky integrate-and-fire
neurons) over discrete time steps, trains them on **tasks**, and streams what
happens to a **dashboard** — live or replayed from disk.

Everything is wired through three decoupled layers:

```
   ┌─────────────┐   publishes    ┌────────────┐   dispatches   ┌──────────────┐
   │    Task     │ ─ MonitorEvent ▶│  EventBus  │ ─ on_event() ─▶│   Monitors   │
   │ (training)  │                 │ (pub/sub)  │                │ (observers)  │
   └─────────────┘                 └────────────┘                └──────────────┘
        │  builds via                                              │ write / forward
        ▼                                                          ▼
   FactoryHub (neurons, synapses,                       artifacts/<run_id>/…  +
   readouts, losses, encoders,                          WebSocket → browser UI
   vision backbones)
```

A task never talks to a monitor or the dashboard directly. It only **publishes
events**. Monitors **subscribe** to the topics they care about. This is the
single most important design rule in the codebase — respect it and features
stay decoupled.

## 2. Event flow (publish / subscribe)

- **Contracts** — [`contracts/monitors.py`](../src/neuroforge/contracts/monitors.py)
  defines `EventTopic` (SPIKE, WEIGHT, SCALAR, TRAINING_TRIAL, TOPOLOGY,
  RUN_START/END, …), the `MonitorEvent` payload, and the `IMonitor` /
  `IEventBus` protocols.
- **Bus** — [`monitors/bus.py`](../src/neuroforge/monitors/bus.py) is the concrete
  `EventBus`: topic-keyed subscriber lists guarded by a re-entrant lock, with a
  snapshot-then-dispatch `publish()` so subscriptions can change mid-dispatch
  and a background dashboard thread can subscribe safely. Dispatch is
  **synchronous** — keep `on_event` handlers fast.
- **Tasks** publish via the shared `_emit(...)` helper on
  [`tasks/base.py:BaseTask`](../src/neuroforge/tasks/base.py).
- **Monitors** (~14 of them in [`monitors/`](../src/neuroforge/monitors/))
  implement `IMonitor` and turn events into something useful: `ArtifactWriter`
  (writes `scalars.csv`, `topology.json`, run metadata), `EventRecorderMonitor`
  (writes `events.ndjson` for replay), `StabilityMonitor`, `TrialStatsMonitor`,
  the vision monitors, and the dashboard's WebSocket broadcaster.

## 3. Module map

| Package | Responsibility |
|---|---|
| `contracts/` | Protocol/interface definitions (neurons, synapses, learning, monitors, encoding). The seams everything else plugs into. |
| `core/` | Low-level utilities: determinism/seeding, device resolution (`torch_utils`: `smart_device`, `resolve_device_dtype`, `cuda_maybe_sync`), the type-safe `Registry`, Dale's law, surrogate gradients. |
| `engine/` | `CoreEngine` — the population/projection simulation loop. |
| `neurons/`, `synapses/` | LIF model (+ surrogate), static / Dale's / delayed synapses. |
| `network/` | Network specs, `NetworkFactory`, topology builders (dense/sparse), `gate_builder`. |
| `factories/` | `FactoryHub` (`DEFAULT_HUB`) — registry-driven construction of neurons, synapses, readouts, losses, encoders, vision backbones. |
| `encoding/`, `readout/`, `losses/` | Rate encoders, readout decoders, loss functions. |
| `tasks/` | Trainable tasks: `LogicGateTask`, `MultiGateTask`, `VisionClassificationTask`. See the registry below. |
| `vision/` | Spiking conv/pool/res blocks, `lif_convnet_v1` backbone, vision engine + factory. |
| `monitors/` | The observer side of the event bus (see §2). |
| `runners/` | `cli.py` (the `neuroforge` command), training/stability/benchmark harnesses, the vision runner. |
| `dashboard/` | aiohttp + WebSocket server and the static front-end (Core / Vision / Topology / Params tabs). |

## 4. How to add a new task

Tasks are catalogued in
[`tasks/registry.py`](../src/neuroforge/tasks/registry.py) so callers dispatch by
key instead of hard-coding `if/elif`.

1. **Write the config + task.** Create `tasks/my_task.py` with a
   `@dataclass` config and a `MyTask(BaseTask)` class. Subclass `BaseTask` and
   call `super().__init__(event_bus, stop_check)` so you inherit `_emit()` and
   `_should_stop()`. Build components from the injected hub
   (`self._hub`, default `DEFAULT_HUB`) — not from the global — so tests can
   substitute a fake hub.
2. **Publish events**, don't call monitors. At minimum emit `run_start`,
   `training_trial`/`scalar` per step, and `run_end`. Existing tasks are the
   reference.
3. **Register it.** Add a one-line entry to `TASK_REGISTRY` with a lazy loader
   returning `(ConfigCls, TaskCls)`. The CLI (`runners/cli.py:_cmd_run`) then
   runs it generically via `_run_task`.
4. **Expose it where needed.** The dashboard dispatches by a `gate` key via the
   `_train_handlers` table in
   [`dashboard/server.py`](../src/neuroforge/dashboard/server.py); add a handler
   there only if the task should be launchable from the UI.
5. **Test it.** Add a smoke test under `tests/` that runs a few steps on CPU and
   asserts events/artifacts are produced (see
   `tests/unit/test_logic_gates_vision_pipeline.py`).

## 5. How to add a new monitor

Monitors need no registry — that is the payoff of the pub/sub design.

1. Implement the `IMonitor` protocol: `on_event(event)`, `enabled`, `reset()`,
   `snapshot()`.
2. Subscribe it: `bus.subscribe(EventTopic.SCALAR, my_monitor)` for one topic,
   or `bus.subscribe_all(my_monitor)` for every topic.
3. Keep `on_event` cheap — dispatch is synchronous and shared across all
   subscribers. Copy tensors to CPU lazily and only when you actually export.

## 6. Running it

```bash
pip install -e ".[torch,dashboard,dev]"
neuroforge run --task multi_gate          # train; writes artifacts/<run_id>/
neuroforge ui                             # dashboard at http://127.0.0.1:8050
pytest                                    # ~470 tests
```

On Windows, `./start_ui.ps1` creates the venv, installs extras, and launches the
dashboard in one step.
