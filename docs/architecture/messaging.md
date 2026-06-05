# Messaging Boundary

Phase 1 separates publish/subscribe infrastructure from monitor
implementations.

## Packages

- `neuroforge.contracts.messaging` defines event topics, event DTOs, and the
  publisher/subscriber/bus protocols.
- `neuroforge.messaging` owns the concrete `EventBus` and small re-export
  modules for topics, events, publishers, and subscribers.
- `neuroforge.contracts.monitoring` defines `IMonitor`, the lifecycle and
  snapshot protocol for observability subscribers.
- `neuroforge.monitors` contains concrete monitors that subscribe to events and
  turn them into artifacts, dashboard snapshots, or derived metrics.

## Dependency Rules

- Publishers depend on messaging contracts or `neuroforge.messaging.bus`.
- Monitors depend on messaging contracts.
- Messaging must not import monitor implementations.
- Monitor implementations should not be required by simulation, task, or
  application publishers.

## Canonical Imports

Code that publishes or subscribes to events should import:

```python
from neuroforge.contracts.messaging import EventTopic, MonitorEvent
from neuroforge.messaging.bus import EventBus
```

Monitor implementations that need the lifecycle protocol should import:

```python
from neuroforge.contracts.monitoring import IMonitor
```

Legacy wrapper paths are not retained. Do not add forwarding modules for the
old monitor-contract or monitor-bus locations.
