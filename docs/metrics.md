# Understanding Metrics

TraceIQ computes several metrics to quantify influence between agents.

## Core Concepts

### Embeddings

Every piece of content (sender message, receiver response) is converted to a high-dimensional vector using sentence embeddings. Similar content produces similar vectors.

### Baseline

Each receiver maintains a "baseline" - a rolling average of their recent response embeddings. This represents their typical behavior pattern.

```
baseline = mean(last N receiver embeddings)
```

The `baseline_window` config parameter controls N (default: 10).

## Metrics

### Drift Delta

**What it measures**: How much the receiver's response deviated from their typical behavior.

**Formula**:
```
drift_delta = 1 - cosine_similarity(current_response, baseline_before)
```

**Range**: 0.0 to 2.0 (typically 0.0 to 1.0)

**Interpretation**:
- `0.0`: Response identical to baseline (no change)
- `0.3`: Moderate deviation from typical behavior
- `0.7+`: Significant behavioral shift
- `1.0`: Response orthogonal to baseline (complete change)

**Flag**: `high_drift` when `drift_delta > drift_threshold`

### Influence Score

**What it measures**: How aligned the sender's content was with the change in the receiver's baseline.

**Formula**:
```
baseline_shift = baseline_after - baseline_before
influence_score = cosine_similarity(sender_embedding, baseline_shift)
```

**Range**: -1.0 to 1.0

**Interpretation**:
- `+1.0`: Sender's content perfectly aligned with receiver's shift
- `+0.5`: Strong positive correlation
- `0.0`: No correlation between sender and receiver's change
- `-0.5`: Sender's content inversely correlated (receiver moved away)

**Flag**: `high_influence` when `influence_score > influence_threshold`

### Receiver Baseline Drift

**What it measures**: The magnitude of how much the receiver's baseline shifted.

**Formula**:
```
receiver_baseline_drift = ||baseline_after - baseline_before||
```

**Interpretation**: Larger values indicate the receiver's overall behavioral pattern is changing more significantly.

## Cold Start

The first interaction for any receiver is marked as `cold_start=True`:
- No baseline exists yet
- `drift_delta = 0.0`
- `influence_score = 0.0`
- The response is used to initialize the baseline

## Thresholds

Configure thresholds in `TrackerConfig`:

```python
config = TrackerConfig(
    drift_threshold=0.3,      # Flag high_drift above this
    influence_threshold=0.5,  # Flag high_influence above this
    baseline_window=10,       # Rolling window size
)
```

## Graph Metrics

### Top Influencers

Agents ranked by sum of outgoing influence scores:

```python
top_influencers = sum(influence_scores for all interactions where agent is sender)
```

### Top Susceptible

Agents ranked by sum of incoming influence:

```python
susceptibility = sum(influence_scores for all interactions where agent is receiver)
```

### Influence Chains

Paths through the influence graph where each edge exceeds a minimum weight, showing how influence propagates.

## Example Analysis

```python
tracker = InfluenceTracker(use_mock_embedder=True)

# Track interactions
results = tracker.bulk_track(interactions)

# Analyze flags
high_drift_events = [r for r in results if "high_drift" in r["flags"]]
high_influence_events = [r for r in results if "high_influence" in r["flags"]]

print(f"Events causing significant change: {len(high_drift_events)}")
print(f"Events with strong influence: {len(high_influence_events)}")

# Get summary
summary = tracker.summary()

# Who influences others the most?
print("Top Influencers:")
for agent, score in summary.top_influencers:
    print(f"  {agent}: {score:.3f}")

# Who is most influenced?
print("Most Susceptible:")
for agent, score in summary.top_susceptible:
    print(f"  {agent}: {score:.3f}")
```

## Best Practices

1. **Baseline Window**: Use larger windows (10-20) for stable baselines, smaller (3-5) for detecting rapid changes

2. **Thresholds**: Start with defaults (0.3 drift, 0.5 influence) and adjust based on your data

3. **Real Embeddings**: For production, use real sentence-transformers instead of mock embedder

4. **Interpretation**: High drift without high influence may indicate external factors; high influence with low drift may indicate subtle but aligned messaging
