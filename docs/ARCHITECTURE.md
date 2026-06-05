# NeuroForge Architecture

NeuroForge is organized around narrow packages with explicit boundaries. The
current package map is:

| Package | Responsibility |
|---|---|
| `contracts/` | Protocols, DTOs, and interface-level abstractions. |
| `kernel/` | Device, dtype, torch, determinism, time, CUDA, and parallel utilities. |
| `messaging/` | Publish/subscribe infrastructure and the concrete event bus. |
| `biology/` | Biological neural primitives: compartments, neurons, synapses, receptors, ion channels, plasticity, neuromodulators, and astrocytes. |
| `simulation/` | Runtime execution, topology specs/builders, and run context. |
| `construction/` | Registries, composition root, and cross-domain factories. |
| `learning/` | Training loops, losses, encoders, readouts, objectives, and supervised-learning utilities. |
| `perception/` | Reusable perception and generic vision infrastructure. |
| `environments/` | Worlds and external clients, including games and emulator bridges. |
| `agents/` | Agent, brain, policy, sensor, actuator, memory, and rollout abstractions. |
| `neuroevolution/` | Reusable population search, genomes, NEAT, CPPN, HyperNEAT, and evolution operators. |
| `applications/` | Complete workflows such as SMB3 training, evolution, and evaluation. |
| `observability/` | Event recording, monitors, artifacts, and metrics. |
| `interfaces/` | CLI and dashboard entrypoints that delegate to applications. |

The game-playing architecture is defined in
[`docs/architecture/agent_game_architecture.md`](architecture/agent_game_architecture.md).
That document is the canonical guide for agent vs brain vs policy boundaries,
BizHawk vs SMB3 environment separation, generic vision vs SMB3 HUD extraction,
and reusable neuroevolution vs SMB3-specific orchestration.

## Event Flow

Tasks and applications publish `MonitorEvent` instances to the event bus.
Monitors subscribe to topics and write artifacts, metrics, event streams, or
dashboard updates. Producers do not call monitors directly.

## Extension Rules

- Add reusable biology under `biology/`.
- Add execution/runtime mechanics under `simulation/` or `kernel/`.
- Add reusable vision under `perception/vision/`.
- Add game-specific observations, actions, rewards, HUD parsing, and adapters
  under `environments/games/<game>/`.
- Add reusable agents, brains, sensors, actuators, and policies under `agents/`.
- Add reusable NEAT/HyperNEAT/CPPN infrastructure under `neuroevolution/`.
- Add complete SMB3 workflows under `applications/smb3/`.
- Add CLI/dashboard entrypoints under `interfaces/`.
- Keep configs, examples, helper scripts, Lua scripts, docs, tests, and
  generated artifacts outside `src/`.

No forwarding import wrappers are part of the architecture. Migrations update
callers to canonical paths and remove old module paths in the same phase.
