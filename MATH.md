# TraceIQ Mathematical Framework

This document provides the complete mathematical formulation for TraceIQ's influence quantification metrics, as implemented in v0.3.0.

## Table of Contents

1. [Notation](#notation)
2. [Drift Detection](#drift-detection)
3. [Influence Quotient (IQx)](#influence-quotient-iqx)
4. [Accumulated Influence](#accumulated-influence)
5. [Propagation Risk](#propagation-risk)
6. [Attack Surface](#attack-surface)
7. [Risk-Weighted Influence](#risk-weighted-influence)
8. [Anomaly Detection](#anomaly-detection)
9. [Computational Complexity](#computational-complexity)
10. [Numerical Stability](#numerical-stability)

---

## Notation

| Symbol | Description |
|--------|-------------|
| $s_j(t)$ | State embedding of agent $j$ at time $t$ |
| $D_j(t)$ | Drift of agent $j$ at time $t$ |
| $I_{i \to j}(t)$ | Influence from agent $i$ to agent $j$ at time $t$ |
| $B_j$ | Baseline median drift for agent $j$ |
| $\varepsilon$ | Numerical stability constant (default: $10^{-6}$) |
| $W$ | Weighted adjacency matrix |
| $c$ | Capability identifier |
| $p(c)$ | Weight of capability $c$ |

---

## Drift Detection

TraceIQ v0.3.0 computes two types of L2 drift to support different use cases:

### Canonical State Drift (`drift_l2_state`) - PRIMARY

The **canonical** IEEE drift measures actual state change between consecutive responses:

$$D_j^{\text{state}}(t) = \|s_j(t^+) - s_j(t^-)\|_2$$

where:
- $s_j(t^-)$ = receiver's embedding from **previous response**
- $s_j(t^+)$ = receiver's embedding from **current response**

This directly measures how much the receiver's output changed due to the interaction.

**Used for:**
- IEEE research metrics
- IQx computation (primary)
- Influence quantification

### Proxy Baseline Drift (`drift_l2_proxy`) - LEGACY

The **proxy** drift measures deviation from typical behavior:

$$D_j^{\text{proxy}}(t) = \|s_j(t) - \bar{B}_j\|_2$$

where:
- $s_j(t)$ = current receiver embedding
- $\bar{B}_j$ = rolling mean of receiver's recent embeddings (baseline)

This measures whether current behavior is "typical" for this receiver.

**Used for:**
- Anomaly detection
- Behavioral profiling
- Legacy compatibility

### Which Drift to Use?

| Metric | Best For | When to Use |
|--------|----------|-------------|
| `drift_l2_state` | Influence measurement | Research, IEEE metrics, IQx/RWI |
| `drift_l2_proxy` | Anomaly detection | Behavioral deviation, security monitoring |
| `drift_l2` | Backward compatibility | Legacy code (maps to canonical if available) |

### Cosine Drift (Legacy)

The original cosine-based drift (still available as `drift_delta`):

$$\text{drift}_\text{cosine} = 1 - \frac{s_j(t^+) \cdot s_j(t^-)}{\|s_j(t^+)\| \|s_j(t^-)\|}$$

**Note:** L2 drift is preferred for IEEE metrics as it preserves magnitude information.

---

## Influence Quotient (IQx)

The Influence Quotient normalizes drift by receiver baseline:

$$\text{IQx}_{i \to j}(t) = \frac{D_j(t)}{B_j + \varepsilon}$$

where:
- $B_j$ = rolling median of agent $j$'s historical drift values
- $\varepsilon$ = numerical stability constant

### Interpretation

| IQx Value | Interpretation |
|-----------|----------------|
| IQx < 1.0 | Below-average influence |
| IQx ≈ 1.0 | Average influence |
| IQx > 1.0 | Above-average influence |
| IQx > 2.0 | Significant influence |

### Algorithm

```
function compute_IQx(drift, baseline_median, epsilon):
    return drift / (baseline_median + epsilon)
```

---

## Accumulated Influence

The accumulated influence over a window $W$:

$$\text{AI}_j(W) = \sum_{t \in W} \text{IQx}_{i \to j}(t)$$

This measures total influence received by agent $j$ from agent $i$ over window $W$.

---

## Propagation Risk

Propagation Risk uses the spectral radius of the weighted adjacency matrix:

$$\text{PR}(W) = \rho(W) = \max_{k} |\lambda_k|$$

where $\lambda_k$ are eigenvalues of the adjacency matrix $W$.

### Adjacency Matrix Construction

$$W_{ij} = \text{mean}(\{\text{IQx}_{i \to j}(t) : t \in \text{history}\})$$

### Interpretation

| PR Value | Interpretation |
|----------|----------------|
| PR < 1.0 | Influence dampens through network |
| PR = 1.0 | Influence preserved |
| PR > 1.0 | Influence amplifies (unstable) |

### Algorithm

```
function compute_propagation_risk(adjacency_matrix):
    eigenvalues = eigendecomposition(adjacency_matrix)
    return max(abs(eigenvalues))
```

---

## Attack Surface

Attack Surface quantifies agent risk based on capabilities:

$$\text{AS}_i = \sum_{c \in C_i} p(c)$$

where:
- $C_i$ = set of capabilities for agent $i$
- $p(c)$ = weight of capability $c$

### Default Capability Weights

| Capability | Weight | Rationale |
|------------|--------|-----------|
| `admin` | 1.5 | Full system access |
| `execute_code` | 1.0 | Arbitrary code execution |
| `subprocess` | 0.9 | Process spawning |
| `network_access` | 0.8 | External communication |
| `file_write` | 0.7 | State modification |
| `database_write` | 0.6 | Data modification |
| `memory_access` | 0.5 | Memory operations |
| `api_access` | 0.4 | API interactions |
| `file_read` | 0.3 | Read-only access |
| `database_read` | 0.3 | Read-only queries |

---

## Risk-Weighted Influence

Risk-Weighted Influence combines IQx with attack surface:

$$\text{RWI}_{i \to j}(t) = \text{IQx}_{i \to j}(t) \times \text{AS}_i$$

### Interpretation

RWI prioritizes monitoring of high-capability agents with high influence:

| Scenario | IQx | AS | RWI | Priority |
|----------|-----|----|----|----------|
| Low-priv, low influence | 0.5 | 0.3 | 0.15 | Low |
| Low-priv, high influence | 2.0 | 0.3 | 0.6 | Medium |
| High-priv, low influence | 0.5 | 2.5 | 1.25 | Medium |
| High-priv, high influence | 2.0 | 2.5 | 5.0 | **High** |

---

## Anomaly Detection

### Z-Score

Anomaly detection uses Z-score normalization:

$$Z_{i \to j}(t) = \frac{\text{IQx}_{i \to j}(t) - \mu_j}{\sigma_j + \varepsilon}$$

where:
- $\mu_j$ = rolling mean of IQx values for receiver $j$
- $\sigma_j$ = rolling standard deviation

### Alert Threshold

An alert is triggered when:

$$|Z| > \theta_\text{anomaly}$$

Default: $\theta_\text{anomaly} = 2.0$

### Algorithm

```
function compute_z_score(iqx, mean, std, epsilon):
    return (iqx - mean) / (std + epsilon)

function is_anomaly(z_score, threshold):
    return abs(z_score) > threshold
```

---

## Computational Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| L2 Drift | O(d) | d = embedding dimension |
| IQx | O(1) | Constant time |
| Rolling Median | O(w log w) | w = window size |
| Rolling Mean/Std | O(w) | Incremental updates possible |
| Adjacency Matrix | O(n²) | n = number of agents |
| Spectral Radius | O(n³) | Eigenvalue decomposition |
| Attack Surface | O(c) | c = capabilities count |

### Recommendations

1. **For large graphs (n > 100)**: Consider approximate eigenvalue methods
2. **For streaming**: Use incremental statistics updates
3. **For high-frequency**: Batch IQx computations

---

## Numerical Stability

### Epsilon Usage

The epsilon constant ($\varepsilon = 10^{-6}$) prevents division by zero in:

1. **IQx**: When baseline median is zero (cold start)
2. **Z-score**: When standard deviation is zero (no variance)

### Recommended Epsilon Values

| Use Case | Epsilon | Rationale |
|----------|---------|-----------|
| Research | $10^{-10}$ | Maximum precision |
| Production | $10^{-6}$ | Balanced |
| Fast inference | $10^{-4}$ | Minimal overhead |

### Overflow Prevention

For very large IQx values:

```python
MAX_IQX = 1e6
iqx = min(compute_IQx(drift, baseline, epsilon), MAX_IQX)
```

---

## References

1. Spectral Graph Theory: Chung, F. (1997). *Spectral Graph Theory*
2. Influence Maximization: Kempe, D., Kleinberg, J., Tardos, É. (2003). *Maximizing the spread of influence through a social network*
3. Anomaly Detection: Chandola, V., Banerjee, A., Kumar, V. (2009). *Anomaly detection: A survey*
