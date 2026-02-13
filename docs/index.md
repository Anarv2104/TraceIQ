# TraceIQ

**Measure AI-to-AI influence in multi-agent systems.**

TraceIQ provides tools to track, analyze, and visualize how ideas and behaviors spread between AI agents in multi-agent environments.

## Features

- **Influence Tracking**: Monitor how agent outputs influence other agents' responses
- **Drift Detection**: Identify when agents deviate from their baseline behavior
- **Graph Analytics**: Visualize influence networks and detect propagation chains
- **Multiple Backends**: In-memory or SQLite storage
- **CLI & API**: Use programmatically or via command line
- **Visualizations**: Heatmaps, network graphs, and time series plots

## Installation

```bash
# Core installation
pip install traceiq

# With plotting support
pip install traceiq[plot]

# With embedding support
pip install traceiq[embedding]

# Everything
pip install traceiq[all]
```

## Quick Start

```python
from traceiq import InfluenceTracker, TrackerConfig

# Create tracker
tracker = InfluenceTracker(use_mock_embedder=True)

# Track an interaction
result = tracker.track_event(
    sender_id="agent_a",
    receiver_id="agent_b",
    sender_content="We should use renewable energy!",
    receiver_content="That's a great point about energy.",
)

print(f"Influence: {result['influence_score']:.3f}")
print(f"Drift: {result['drift_delta']:.3f}")

# Get summary
summary = tracker.summary()
print(f"Top influencers: {summary.top_influencers}")
```

## Navigation

- [Installation](installation.md) - Detailed installation instructions
- [Quick Start](quickstart.md) - Get started in 5 minutes
- [API Reference](api/index.md) - Complete API documentation
- [CLI Reference](cli.md) - Command-line interface guide
- [Metrics](metrics.md) - Understanding drift and influence scores
- [Examples](examples.md) - Real-world usage examples
