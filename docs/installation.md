# Installation

## Requirements

- Python 3.10 or higher
- pip

## Basic Installation

Install the core package from PyPI:

```bash
pip install traceiq
```

## Installation with Extras

TraceIQ has optional dependencies for different features:

### Plotting Support

For visualization features (heatmaps, network graphs, charts):

```bash
pip install traceiq[plot]
```

### Embedding Support

For real semantic embeddings using sentence-transformers:

```bash
pip install traceiq[embedding]
```

### Full Installation

Install everything:

```bash
pip install traceiq[all]
```

## Development Installation

For contributing to TraceIQ:

```bash
git clone https://github.com/Anarv2104/TraceIQ.git
cd traceiq
pip install -e ".[all,dev]"
```

## Verifying Installation

Test your installation:

```python
from traceiq import InfluenceTracker, __version__

print(f"TraceIQ version: {__version__}")

# Quick test
tracker = InfluenceTracker(use_mock_embedder=True)
result = tracker.track_event(
    sender_id="test_sender",
    receiver_id="test_receiver",
    sender_content="Hello world",
    receiver_content="Hello back",
)
print(f"Installation working! Event tracked: {result['event_id']}")
```

Or via CLI:

```bash
traceiq --version
```

## Troubleshooting

### sentence-transformers not found

If you see an ImportError about sentence-transformers:

```bash
pip install sentence-transformers
```

Or use the mock embedder for testing:

```python
tracker = InfluenceTracker(use_mock_embedder=True)
```

### matplotlib not found

For plotting features:

```bash
pip install matplotlib
```

### NumPy version conflicts

If you encounter NumPy version issues:

```bash
pip install "numpy<2"
```
