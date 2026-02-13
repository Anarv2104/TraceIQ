# Quick Start

Get started with TraceIQ in 5 minutes.

## Basic Usage

### 1. Create a Tracker

```python
from traceiq import InfluenceTracker, TrackerConfig

# Simple setup with defaults
tracker = InfluenceTracker(use_mock_embedder=True)

# Or with custom configuration
config = TrackerConfig(
    storage_backend="memory",  # or "sqlite"
    baseline_window=10,
    drift_threshold=0.3,
    influence_threshold=0.5,
)
tracker = InfluenceTracker(config=config, use_mock_embedder=True)
```

### 2. Track Interactions

```python
# Track a single interaction
result = tracker.track_event(
    sender_id="agent_a",
    receiver_id="agent_b",
    sender_content="We should focus on sustainability.",
    receiver_content="I agree, sustainability is important.",
)

print(f"Event ID: {result['event_id']}")
print(f"Influence Score: {result['influence_score']:.3f}")
print(f"Drift Delta: {result['drift_delta']:.3f}")
print(f"Flags: {result['flags']}")
```

### 3. Track Multiple Interactions

```python
interactions = [
    {
        "sender_id": "agent_a",
        "receiver_id": "agent_b",
        "sender_content": "AI will transform healthcare.",
        "receiver_content": "Yes, medical AI is promising.",
    },
    {
        "sender_id": "agent_b",
        "receiver_id": "agent_c",
        "sender_content": "Healthcare AI needs regulation.",
        "receiver_content": "Regulation is necessary for safety.",
    },
]

results = tracker.bulk_track(interactions)
for r in results:
    print(f"{r['sender_id']} -> {r['receiver_id']}: {r['influence_score']:.3f}")
```

### 4. Get Summary Report

```python
summary = tracker.summary(top_n=5)

print(f"Total events: {summary.total_events}")
print(f"Average drift: {summary.avg_drift_delta:.4f}")
print(f"High influence events: {summary.high_influence_count}")
print(f"Top influencers: {summary.top_influencers}")
print(f"Most susceptible: {summary.top_susceptible}")
```

### 5. Export Data

```python
# Export to CSV
tracker.export_csv("interactions.csv")

# Export to JSONL
tracker.export_jsonl("interactions.jsonl")
```

### 6. Visualize (requires matplotlib)

```python
from traceiq.plotting import plot_influence_heatmap, plot_top_influencers

# Heatmap of influence scores
plot_influence_heatmap(tracker.graph, output_path="heatmap.png")

# Bar chart of top influencers
plot_top_influencers(tracker.graph, n=5, output_path="influencers.png")
```

## Using Real Embeddings

For production use with semantic embeddings:

```python
# Requires: pip install traceiq[embedding]
tracker = InfluenceTracker(use_mock_embedder=False)

# The first call will download the model (~90MB)
result = tracker.track_event(
    sender_id="agent_a",
    receiver_id="agent_b",
    sender_content="Climate change requires immediate action.",
    receiver_content="We need to act on climate now.",
)

# Real embeddings capture semantic similarity
print(f"Semantic influence: {result['influence_score']:.3f}")
```

## Using SQLite Storage

For persistent storage:

```python
config = TrackerConfig(
    storage_backend="sqlite",
    storage_path="traceiq.db",
)

tracker = InfluenceTracker(config=config, use_mock_embedder=True)

# Data persists across sessions
tracker.track_event(...)
tracker.close()

# Later, reopen the same database
tracker2 = InfluenceTracker(config=config, use_mock_embedder=True)
events = tracker2.get_events()  # Previous events are still there
```

## CLI Usage

```bash
# Initialize database
traceiq init --db mydata.db

# Ingest interactions from JSONL
traceiq ingest interactions.jsonl --db mydata.db

# View summary
traceiq summary --db mydata.db

# Export data
traceiq export --db mydata.db -o output.csv

# Generate plots
traceiq plot heatmap --db mydata.db -o heatmap.png
```

## Next Steps

- [Metrics Guide](metrics.md) - Understand drift and influence scores
- [API Reference](api/index.md) - Complete API documentation
- [Examples](examples.md) - Real-world usage patterns
