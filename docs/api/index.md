# API Reference

Complete API documentation for TraceIQ.

## Core Classes

### InfluenceTracker

The main class for tracking AI-to-AI influence.

```python
from traceiq import InfluenceTracker, TrackerConfig

tracker = InfluenceTracker(
    config: TrackerConfig | None = None,
    use_mock_embedder: bool = False,
)
```

**Parameters:**
- `config` - Configuration object. Uses defaults if not provided.
- `use_mock_embedder` - Use deterministic mock embeddings instead of sentence-transformers.

**Methods:**

#### `track_event()`

Track a single interaction.

```python
result = tracker.track_event(
    sender_id: str,
    receiver_id: str,
    sender_content: str,
    receiver_content: str,
    metadata: dict | None = None,
) -> dict
```

**Returns:** Dictionary with:
- `event_id` - Unique event identifier
- `sender_id` - Sender agent ID
- `receiver_id` - Receiver agent ID
- `influence_score` - Influence score (-1 to 1)
- `drift_delta` - Drift from baseline (0 to 2)
- `flags` - List of flags (`high_drift`, `high_influence`)
- `cold_start` - True if first event for receiver

#### `bulk_track()`

Track multiple interactions.

```python
results = tracker.bulk_track(
    interactions: list[dict],
) -> list[dict]
```

**Parameters:**
- `interactions` - List of dicts with `sender_id`, `receiver_id`, `sender_content`, `receiver_content`, optional `metadata`

#### `summary()`

Generate summary report.

```python
report = tracker.summary(top_n: int = 10) -> SummaryReport
```

#### `export_csv()` / `export_jsonl()`

Export data to file.

```python
tracker.export_csv(output_path: str | Path)
tracker.export_jsonl(output_path: str | Path)
```

#### `get_events()` / `get_scores()`

Retrieve stored data.

```python
events: list[InteractionEvent] = tracker.get_events()
scores: list[ScoreResult] = tracker.get_scores()
```

#### `close()`

Close the tracker and release resources.

```python
tracker.close()
```

**Context Manager:**

```python
with InfluenceTracker(config=config) as tracker:
    tracker.track_event(...)
# Automatically closed
```

---

### TrackerConfig

Configuration for InfluenceTracker.

```python
from traceiq import TrackerConfig

config = TrackerConfig(
    # Storage
    storage_backend: str = "memory",  # "memory" or "sqlite"
    storage_path: str | None = None,  # Required for sqlite

    # Embedding
    embedding_model: str = "all-MiniLM-L6-v2",
    max_content_length: int = 512,
    embedding_cache_size: int = 10000,

    # Scoring
    baseline_window: int = 10,
    drift_threshold: float = 0.3,
    influence_threshold: float = 0.5,

    # Misc
    random_seed: int | None = None,
)
```

---

### InteractionEvent

Pydantic model for interaction events.

```python
from traceiq import InteractionEvent

event = InteractionEvent(
    event_id: UUID = auto,      # Auto-generated
    sender_id: str,
    receiver_id: str,
    content: str,
    timestamp: datetime = now,  # Auto-generated
    metadata: dict = {},
)
```

---

### ScoreResult

Pydantic model for score results.

```python
from traceiq.models import ScoreResult

score = ScoreResult(
    event_id: UUID,
    influence_score: float,
    drift_delta: float,
    receiver_baseline_drift: float,
    flags: list[str] = [],
    cold_start: bool = False,
)
```

---

### SummaryReport

Pydantic model for summary reports.

```python
from traceiq import SummaryReport

report = SummaryReport(
    total_events: int,
    unique_senders: int,
    unique_receivers: int,
    avg_drift_delta: float,
    avg_influence_score: float,
    high_drift_count: int,
    high_influence_count: int,
    top_influencers: list[tuple[str, float]],
    top_susceptible: list[tuple[str, float]],
    influence_chains: list[list[str]],
)
```

---

## Graph Module

### InfluenceGraph

Graph analytics for influence patterns.

```python
from traceiq.graph import InfluenceGraph

graph = tracker.graph  # Access via tracker
```

**Methods:**

#### `influence_matrix()`

Get influence scores as nested dict.

```python
matrix: dict[str, dict[str, float]] = graph.influence_matrix()
# matrix[sender][receiver] = score
```

#### `top_influencers()`

Get top N influencers by outgoing influence.

```python
top: list[tuple[str, float]] = graph.top_influencers(n=10)
```

#### `top_susceptible()`

Get top N susceptible agents by incoming influence.

```python
susceptible: list[tuple[str, float]] = graph.top_susceptible(n=10)
```

#### `find_influence_chains()`

Find influence propagation paths.

```python
chains: list[list[str]] = graph.find_influence_chains(
    source: str,
    min_weight: float = 0.1,
    max_length: int = 5,
)
```

#### `detect_cycles()`

Detect influence cycles.

```python
cycles: list[list[str]] = graph.detect_cycles(min_weight=0.1)
```

---

## Plotting Module

Requires `pip install traceiq[plot]`.

```python
from traceiq.plotting import (
    plot_drift_over_time,
    plot_influence_heatmap,
    plot_top_influencers,
    plot_top_susceptible,
    plot_influence_network,
)
```

### `plot_drift_over_time()`

```python
plot_drift_over_time(
    events: list[InteractionEvent],
    scores: list[ScoreResult],
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
)
```

### `plot_influence_heatmap()`

```python
plot_influence_heatmap(
    influence_graph: InfluenceGraph,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
)
```

### `plot_top_influencers()`

```python
plot_top_influencers(
    influence_graph: InfluenceGraph,
    n: int = 10,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
)
```

### `plot_influence_network()`

```python
plot_influence_network(
    influence_graph: InfluenceGraph,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 10),
    min_edge_weight: float = 0.1,
)
```

---

## Storage Module

### StorageBackend (Abstract)

```python
from traceiq.storage import StorageBackend

class StorageBackend(ABC):
    def store_event(self, event: InteractionEvent) -> None: ...
    def store_score(self, score: ScoreResult) -> None: ...
    def get_event(self, event_id: UUID) -> InteractionEvent | None: ...
    def get_score(self, event_id: UUID) -> ScoreResult | None: ...
    def get_all_events(self) -> list[InteractionEvent]: ...
    def get_all_scores(self) -> list[ScoreResult]: ...
    def close(self) -> None: ...
```

### MemoryStorage

In-memory storage implementation.

```python
from traceiq.storage import MemoryStorage
storage = MemoryStorage()
```

### SQLiteStorage

Persistent SQLite storage.

```python
from traceiq.storage import SQLiteStorage
storage = SQLiteStorage(db_path="traceiq.db")
```
