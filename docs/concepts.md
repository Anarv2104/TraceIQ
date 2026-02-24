# Core Concepts

This document explains the fundamental concepts behind TraceIQ's influence measurement.

## Overview

TraceIQ measures how AI agents influence each other by tracking changes in their outputs after interactions. It uses semantic embeddings to represent agent states and computes various metrics to quantify influence.

## Key Concepts

### Agent State

An agent's state is represented as a semantic embedding of its output:

```
state = embed(agent_output)
```

TraceIQ supports two embedding backends:
- **MockEmbedder**: Hash-based, deterministic, fast (for testing)
- **SentenceTransformerEmbedder**: Neural embeddings, semantic similarity (for production)

### Baseline

Each receiver agent maintains a rolling baseline of recent embeddings:

```
baseline = rolling_mean(last N embeddings)
```

The baseline represents "typical" behavior. Drift is measured against this baseline.

### Drift

Drift measures how much an agent's output changed:

- **State Drift** (`drift_l2_state`): L2 distance between consecutive outputs
- **Proxy Drift** (`drift_l2_proxy`): L2 distance from rolling baseline

### Influence

Influence measures correlation between sender content and receiver behavior change:

```
influence_score = cosine_similarity(sender_embedding, baseline_shift)
```

Positive values indicate the receiver shifted toward the sender's semantic space. Negative values indicate the receiver shifted away.

### Cold Start

The first few interactions for a receiver have no baseline:
- `cold_start=True` flag is set
- Metrics are marked as invalid (`valid=False`)
- Alerts are suppressed

Always check `result["valid"]` before taking action on metrics.

## Metric Summary

| Metric | What It Measures |
|--------|------------------|
| `drift_l2_state` | Output change magnitude |
| `IQx` | Influence normalized by baseline responsiveness |
| `Z_score` | Anomaly detection (MAD-based) |
| `RWI` | IQx weighted by sender attack surface |
| `PR` | Network-level influence propagation tendency |

## Important Limitations

1. **Correlation, not causation**: High metrics indicate state change correlated with message receipt, not proven causal influence.

2. **Observable outputs only**: TraceIQ cannot access agent internal states or reasoning.

3. **Threshold calibration**: Default thresholds are starting points; calibrate for your specific system.

4. **Embedding quality**: Results depend on the embedding model's ability to capture semantic similarity.

See [docs/THEORY.md](THEORY.md) for formal definitions and proofs.
