# Understanding Metrics

TraceIQ computes several metrics to quantify influence between agents.

## Core Concepts

### Embeddings

Every piece of content (sender message, receiver response) is converted to a high-dimensional vector using sentence embeddings. Similar content produces similar vectors.

### Baseline

Each receiver maintains a "baseline" - a rolling average of their recent response embeddings. This represents their typical response pattern.

```
baseline = mean(last N receiver embeddings)
```

The `baseline_window` config parameter controls N (default: 10).

### What Baseline Actually Measures

The baseline is a rolling average of the receiver's **response embeddings**, not their internal state or "cognition". This means:

- **Drift measures**: "How different is this response from the receiver's typical responses?"
- **Influence measures**: "Did the receiver's response pattern shift toward the sender's content?"

This is **NOT**:
- Measuring receiver "beliefs" or internal cognitive state
- Tracking what messages the receiver received
- A measure of what the receiver "knows"

The baseline captures observable behavior (outputs) only, which is the appropriate level of analysis for external influence tracking.

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

| Value | Interpretation |
|-------|----------------|
| +1.0 | Receiver shifted strongly toward sender's semantic space |
| +0.5 | Moderate positive correlation - receiver moved somewhat toward sender |
| 0.0 | No correlation between sender's content and receiver's behavioral shift |
| -0.5 | Counter-influence - receiver moved AWAY from sender's content |
| -1.0 | Strong counter-influence - receiver shifted in opposite direction |

**Understanding Negative Influence**:
A negative influence score indicates that the receiver's behavior shifted *away* from the sender's semantic content. This could mean:
- The receiver is actively resisting or countering the sender's message
- The sender triggered a contrarian response
- The receiver is deliberately differentiating from the sender

Negative influence is still influence - it's just influence in the opposite direction.

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

Higher scores indicate agents whose content tends to correlate with receivers shifting toward their semantic space.

### Top Susceptible

Agents ranked by sum of incoming **drift** (not influence):

```python
susceptibility = sum(drift_deltas for all interactions where agent is receiver)
```

This measures which agents show the most behavioral change when receiving messages. High susceptibility means the agent frequently deviates from their baseline behavior.

**Note**: This is different from `top_influenced()`.

### Top Influenced

Agents ranked by sum of incoming **influence weights**:

```python
influenced = sum(influence_scores for all interactions where agent is receiver)
```

This measures which agents have had their behavior most correlated with sender content. High values indicate the agent tends to shift *toward* what senders are saying.

### Susceptible vs Influenced

| Metric | Measures | Question Answered |
|--------|----------|-------------------|
| `top_susceptible()` | Sum of incoming drift | "Who changes the most?" |
| `top_influenced()` | Sum of incoming influence | "Who moves toward senders the most?" |

An agent can be highly susceptible (behavior changes a lot) but not highly influenced (changes aren't correlated with sender content). This might indicate the agent is responding to external factors or has high internal variability.

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

## IEEE Metrics (v0.3.0)

TraceIQ v0.3.0 introduces mathematically rigorous metrics for research applications.

### Drift Types

TraceIQ computes two types of L2 drift:

#### Canonical State Drift (`drift_l2_state`) - PRIMARY

Measures actual state change between consecutive responses:

```
D_state = ||s(t+) - s(t-)||_2
```

Where:
- `s(t-)` = receiver's embedding from **previous** response
- `s(t+)` = receiver's embedding from **current** response

This directly measures how much the receiver's output changed.

#### Proxy Baseline Drift (`drift_l2_proxy`) - LEGACY

Measures deviation from typical behavior:

```
D_proxy = ||s(t) - baseline||_2
```

Where:
- `s(t)` = current receiver embedding
- `baseline` = rolling mean of recent embeddings

This measures whether current behavior is "typical".

#### Which to Use?

| Field | Best For | When to Use |
|-------|----------|-------------|
| `drift_l2_state` | Influence measurement | Research, IEEE metrics, IQx/RWI |
| `drift_l2_proxy` | Anomaly detection | Behavioral deviation, security monitoring |
| `drift_l2` | Backward compatibility | Legacy code (maps to canonical if available) |

### Influence Quotient (IQx)

Normalizes drift by receiver baseline responsiveness:

```
IQx = drift / (baseline_median + epsilon)
```

| IQx Value | Interpretation |
|-----------|----------------|
| < 1.0 | Below-average influence |
| ≈ 1.0 | Average influence |
| > 1.0 | Above-average influence |
| > 2.0 | Significant influence |

### Propagation Risk

Network-level instability measured as spectral radius:

```
PR = max(|eigenvalues(W)|)
```

| PR Value | Interpretation |
|----------|----------------|
| < 1.0 | Influence dampens through network |
| = 1.0 | Influence preserved |
| > 1.0 | Influence amplifies (unstable) |

### Risk-Weighted Influence (RWI)

Combines IQx with sender's attack surface:

```
RWI = IQx × attack_surface
```

Prioritizes monitoring of high-capability agents with high influence.

### Z-Score Anomaly Detection

Detects outliers in IQx values:

```
Z = (IQx - mean) / (std + epsilon)
```

Alert triggered when `|Z| > anomaly_threshold` (default: 2.0).

### Example Usage

```python
from traceiq import InfluenceTracker, TrackerConfig

config = TrackerConfig(
    epsilon=1e-6,
    anomaly_threshold=2.0,
    capability_weights={"execute_code": 1.0, "admin": 1.5}
)

tracker = InfluenceTracker(config=config, use_mock_embedder=True)
tracker.capabilities.register_agent("agent_0", ["execute_code", "admin"])

result = tracker.track_event(
    sender_id="agent_0",
    receiver_id="agent_1",
    sender_content="Execute command",
    receiver_content="Executing...",
)

print(f"Canonical Drift: {result['drift_l2_state']}")
print(f"IQx: {result['IQx']}")
print(f"RWI: {result['RWI']}")
print(f"Alert: {result['alert']}")

# Network-level metrics
pr = tracker.get_propagation_risk()
alerts = tracker.get_alerts()
```

## Best Practices

1. **Baseline Window**: Use larger windows (10-20) for stable baselines, smaller (3-5) for detecting rapid changes

2. **Thresholds**: Start with defaults (0.3 drift, 0.5 influence) and adjust based on your data

3. **Real Embeddings**: For production, use real sentence-transformers instead of mock embedder

4. **Interpretation**: High drift without high influence may indicate external factors; high influence with low drift may indicate subtle but aligned messaging

5. **Canonical vs Proxy**: Use `drift_l2_state` for research and influence quantification; use `drift_l2_proxy` for anomaly detection and behavioral profiling
