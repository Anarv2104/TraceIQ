# Configuration Reference

This document provides a complete reference for TraceIQ's configuration options.

## TrackerConfig

The `TrackerConfig` class controls all aspects of tracker behavior.

```python
from traceiq import TrackerConfig

config = TrackerConfig(
    # Storage
    storage_backend="sqlite",
    storage_path="traceiq.db",

    # Embedding
    embedding_model="all-MiniLM-L6-v2",

    # Scoring
    baseline_window=10,
    epsilon=1e-6,
    anomaly_threshold=2.0,
)
```

## Parameter Reference

### Storage Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `storage_backend` | `"memory"` \| `"sqlite"` | `"memory"` | Storage backend type |
| `storage_path` | `str \| None` | `None` | Path for SQLite database (required if using sqlite) |

### Embedding Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `embedding_model` | `str` | `"all-MiniLM-L6-v2"` | Sentence-transformer model name |
| `max_content_length` | `int` | `5000` | Maximum content length before truncation |
| `embedding_cache_size` | `int` | `10000` | LRU cache size for embeddings |

### Scoring Settings

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `baseline_window` | `int` | `10` | Rolling window size for baseline computation |
| `drift_threshold` | `float` | `0.3` | Threshold for `high_drift` flag |
| `influence_threshold` | `float` | `0.5` | Threshold for `high_influence` flag |
| `epsilon` | `float` | `1e-6` | Numerical stability constant |
| `anomaly_threshold` | `float` | `2.0` | Z-score threshold for anomaly alerts |

### Capability Settings (v0.3.0)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `capability_weights` | `dict[str, float]` | `{}` | Custom weights for capability types |
| `capability_registry_path` | `str \| None` | `None` | Path to JSON capability registry |

### Risk Scoring Settings (v0.4.0)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_risk_scoring` | `bool` | `True` | Enable risk score computation |
| `enable_policy` | `bool` | `False` | Enable policy engine for mitigation |
| `baseline_k` | `int` | `20` | Minimum samples for valid metrics (cold-start threshold) |
| `risk_thresholds` | `tuple[float, float, float]` | `(0.2, 0.5, 0.8)` | (low, medium, high) risk level thresholds |

### Policy Settings (v0.4.0)

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `enable_trust_decay` | `bool` | `True` | Enable trust decay on policy violations |
| `trust_decay_rate` | `float` | `0.1` | Trust decay amount per violation |

### Miscellaneous

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `random_seed` | `int \| None` | `None` | Random seed for reproducibility |

## Example Configurations

### Research Configuration

For research experiments requiring maximum precision:

```python
config = TrackerConfig(
    storage_backend="sqlite",
    storage_path="research.db",
    baseline_window=20,
    epsilon=1e-10,
    anomaly_threshold=2.0,
    baseline_k=30,
)
```

### Production Configuration

For production monitoring with policy enforcement:

```python
config = TrackerConfig(
    storage_backend="sqlite",
    storage_path="/var/lib/traceiq/prod.db",
    enable_risk_scoring=True,
    enable_policy=True,
    anomaly_threshold=2.5,
    risk_thresholds=(0.3, 0.6, 0.85),
)
```

### Quick Testing Configuration

For rapid development and testing:

```python
config = TrackerConfig(
    storage_backend="memory",
    baseline_window=5,
    baseline_k=3,
)
```

## Environment-Specific Notes

- **epsilon**: Use `1e-10` for research (maximum precision), `1e-6` for production (balanced)
- **baseline_k**: Higher values (20-30) provide more stable metrics but slower warm-up
- **anomaly_threshold**: Start with 2.0, adjust based on false positive rate
