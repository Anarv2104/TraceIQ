# TraceIQ Theoretical Framework

This document provides the formal mathematical foundations for TraceIQ's influence quantification metrics. It establishes the theoretical basis for the empirical measurements implemented in the library.

## Table of Contents

1. [Formal Influence Definition](#formal-influence-definition)
2. [Metric Properties and Proofs](#metric-properties-and-proofs)
3. [Statistical Validation](#statistical-validation)
4. [Assumptions and Limitations](#assumptions-and-limitations)
5. [Computational Considerations](#computational-considerations)

---

## Formal Influence Definition

### Definition 1: Agent State

An **agent state** $S_i(t)$ is a vector representation of agent $i$'s behavior at time $t$. In TraceIQ, this is computed as the semantic embedding of the agent's output:

$$S_i(t) = \text{embed}(\text{output}_i(t)) \in \mathbb{R}^d$$

where $\text{embed}: \text{String} \to \mathbb{R}^d$ is a semantic embedding function (e.g., SentenceTransformer) and $d$ is the embedding dimension.

### Definition 2: Counterfactual Influence

The **counterfactual influence** of agent $j$ on agent $i$ at time $t$ is defined as:

$$I(j \to i, t) = D(S_i(t^+), S_i^{\emptyset}(t^+))$$

where:
- $S_i(t^+)$ is agent $i$'s state after receiving message $M_j$ from agent $j$
- $S_i^{\emptyset}(t^+)$ is the **counterfactual** state agent $i$ would have had without receiving $M_j$
- $D(\cdot, \cdot)$ is a distance metric (e.g., L2 norm)

**Interpretation**: Counterfactual influence measures the difference between what happened and what would have happened without the interaction.

### Definition 3: Observable Influence (Approximation)

Computing $S_i^{\emptyset}(t^+)$ requires running the agent without the message, which is expensive and often impractical. TraceIQ uses an **observable approximation**:

$$\hat{I}(j \to i, t) = D(S_i(t^+), S_i(t^-))$$

where $S_i(t^-)$ is agent $i$'s state immediately before receiving $M_j$.

**Justification**: Under the assumption that agent behavior is locally stable (Assumption 2), we have:

$$S_i^{\emptyset}(t^+) \approx S_i(t^-)$$

This approximation measures the actual state change, attributing it to the interaction.

### Definition 4: Influence Quotient (IQx)

The **Influence Quotient** normalizes influence by the receiver's baseline responsiveness:

$$\text{IQx}(j \to i, t) = \frac{\hat{I}(j \to i, t)}{B_i + \varepsilon}$$

where:
- $B_i = \text{median}(\{\hat{I}(\cdot \to i, \tau) : \tau \in W\})$ is the rolling median of agent $i$'s historical influence values over window $W$
- $\varepsilon > 0$ is a small constant for numerical stability

**Interpretation**: IQx $> 1$ indicates above-average influence; IQx $< 1$ indicates below-average influence.

---

## Metric Properties and Proofs

### Theorem 1: IQx Non-Negativity

**Statement**: For all valid inputs, $\text{IQx} \geq 0$.

**Proof**:
- By definition, $\hat{I}(j \to i, t) = \|S_i(t^+) - S_i(t^-)\|_2 \geq 0$ (L2 norm is non-negative)
- The baseline $B_i \geq 0$ (median of non-negative values)
- $\varepsilon > 0$ by construction
- Therefore, $B_i + \varepsilon > 0$
- Thus, $\text{IQx} = \hat{I} / (B_i + \varepsilon) \geq 0 / (B_i + \varepsilon) = 0$ ∎

### Theorem 2: IQx Zero-Influence Property

**Statement**: If there is no state change ($S_i(t^+) = S_i(t^-)$), then $\text{IQx} = 0$.

**Proof**:
- If $S_i(t^+) = S_i(t^-)$, then $\hat{I} = \|S_i(t^+) - S_i(t^-)\|_2 = 0$
- Therefore, $\text{IQx} = 0 / (B_i + \varepsilon) = 0$ ∎

### Theorem 3: IQx Monotonicity in Drift

**Statement**: For fixed baseline $B$, if $\hat{I}_1 < \hat{I}_2$, then $\text{IQx}_1 < \text{IQx}_2$.

**Proof**:
- $\text{IQx}_1 = \hat{I}_1 / (B + \varepsilon)$
- $\text{IQx}_2 = \hat{I}_2 / (B + \varepsilon)$
- Since $(B + \varepsilon)$ is constant and positive, and $\hat{I}_1 < \hat{I}_2$:
- $\text{IQx}_1 < \text{IQx}_2$ ∎

### Theorem 4: IQx Inverse Monotonicity in Baseline

**Statement**: For fixed drift $\hat{I}$, if $B_1 < B_2$, then $\text{IQx}_1 > \text{IQx}_2$.

**Proof**:
- $\text{IQx}_1 = \hat{I} / (B_1 + \varepsilon)$
- $\text{IQx}_2 = \hat{I} / (B_2 + \varepsilon)$
- Since $B_1 < B_2$, we have $(B_1 + \varepsilon) < (B_2 + \varepsilon)$
- Since $\hat{I} > 0$ and we're dividing by a smaller positive number:
- $\text{IQx}_1 > \text{IQx}_2$ ∎

**Interpretation**: Higher baselines mean the agent is normally more responsive, so the same drift indicates less relative influence.

### Theorem 5: Propagation Risk Interpretation

**Statement**: Let $W$ be the weighted adjacency matrix of influence. Then:
- $\rho(W) < 1$ ⟹ Influence decays through the network
- $\rho(W) = 1$ ⟹ Influence is preserved
- $\rho(W) > 1$ ⟹ Influence amplifies

**Proof** (sketch):
Consider the linear dynamical system $x(t+1) = Wx(t)$ where $x(t)$ represents influence levels. The spectral radius $\rho(W)$ determines asymptotic behavior:
- $\|x(t)\| \to 0$ as $t \to \infty$ if $\rho(W) < 1$ (stable, influence decays)
- $\|x(t)\|$ is bounded if $\rho(W) = 1$ (marginal stability)
- $\|x(t)\| \to \infty$ as $t \to \infty$ if $\rho(W) > 1$ (unstable, influence amplifies)

This follows from the Gelfand formula: $\rho(W) = \lim_{n \to \infty} \|W^n\|^{1/n}$ ∎

**Calibration Note**: The PR threshold of 1.0 is theoretically motivated but may require environment-specific calibration. Factors affecting practical thresholds include:
- Network topology (dense vs sparse)
- Agent diversity (homogeneous vs heterogeneous)
- Interaction frequency
- Weight normalization scheme

When deploying PR monitoring, establish baseline PR values for your specific system before alerting on threshold crossings.

### Theorem 6: PR Scaling Property

**Statement**: For scalar $c > 0$, $\rho(cW) = c \cdot \rho(W)$.

**Proof**:
- Let $\lambda$ be an eigenvalue of $W$ with eigenvector $v$: $Wv = \lambda v$
- Then $(cW)v = c(Wv) = c\lambda v$
- So $c\lambda$ is an eigenvalue of $cW$
- Therefore, $\rho(cW) = \max|c\lambda| = c \cdot \max|\lambda| = c \cdot \rho(W)$ ∎

### Theorem 7: Z-Score Standardization

**Statement**: For a sample $\{x_1, ..., x_n\}$ with mean $\mu$ and std $\sigma$, the Z-scores $\{z_i = (x_i - \mu)/\sigma\}$ satisfy $\bar{z} = 0$ and $s_z = 1$.

**Proof**:
- $\bar{z} = \frac{1}{n}\sum_i \frac{x_i - \mu}{\sigma} = \frac{1}{\sigma}\left(\frac{1}{n}\sum_i x_i - \mu\right) = \frac{1}{\sigma}(\mu - \mu) = 0$
- $s_z^2 = \frac{1}{n}\sum_i z_i^2 = \frac{1}{n\sigma^2}\sum_i (x_i - \mu)^2 = \frac{\sigma^2}{\sigma^2} = 1$ ∎

### Theorem 8: Drift Metric Properties

**Statement**: The L2 drift $D(A, B) = \|A - B\|_2$ is a proper metric.

**Proof**: We verify the metric axioms:
1. **Non-negativity**: $D(A, B) = \|A - B\|_2 \geq 0$ (norm is non-negative)
2. **Identity**: $D(A, A) = \|A - A\|_2 = \|0\|_2 = 0$
3. **Symmetry**: $D(A, B) = \|A - B\|_2 = \|-(B - A)\|_2 = \|B - A\|_2 = D(B, A)$
4. **Triangle inequality**: $D(A, C) = \|A - C\|_2 = \|A - B + B - C\|_2 \leq \|A - B\|_2 + \|B - C\|_2 = D(A, B) + D(B, C)$ ∎

---

## Statistical Validation

### Hypothesis Testing Framework

When comparing conditions (e.g., baseline vs. wrong hint), we use:

1. **Two-sample t-test** for normally distributed metrics:
   - $H_0$: $\mu_1 = \mu_2$ (no difference between conditions)
   - $H_1$: $\mu_1 \neq \mu_2$ (conditions differ)

2. **Mann-Whitney U test** for non-normal metrics (e.g., IQx):
   - Non-parametric alternative when normality assumption is violated
   - Tests whether distributions differ

3. **Effect size (Cohen's d)**:
   - $d = (\bar{x}_1 - \bar{x}_2) / s_{\text{pooled}}$
   - Interpretation: |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, > 0.8 large

### Confidence Intervals

We report 95% confidence intervals using:
- **Parametric**: $\bar{x} \pm t_{0.975, n-1} \cdot \text{SE}$
- **Bootstrap**: Percentile method with 10,000 resamples

### Multiple Comparisons

When making multiple comparisons, we note:
- Individual p-values are reported as-is
- Bonferroni correction: $\alpha_{\text{adjusted}} = \alpha / k$ for $k$ comparisons
- We report both raw and adjusted significance

---

## Assumptions and Limitations

### Assumption 1: Embedding Validity

Agent states can be meaningfully represented as semantic embeddings.

**Implications**:
- Requires that output text captures behavioral state
- Embedding quality directly affects metric accuracy
- Different embedding models may yield different results

**Violation indicators**:
- Very short outputs (< 10 words)
- Highly structured/templated outputs
- Non-textual content

### Assumption 2: Local Stability

In the absence of external input, agent behavior is locally stable over short time intervals.

**Formalization**: $\|S_i(t^+)^{\emptyset} - S_i(t^-)\| \ll \|S_i(t^+) - S_i(t^-)\|$ when $M_j$ has high influence.

**Implications**:
- Baseline drift should be small compared to influence-induced drift
- Enables the observable influence approximation

**Violation indicators**:
- High baseline variance for an agent
- IQx values close to 1.0 for all interactions

### Assumption 3: No Hidden Channels

All influence flows through observed message channels.

**Implications**:
- TraceIQ cannot detect influence via:
  - Shared memory
  - Environment modifications
  - Out-of-band communication

**Violation indicators**:
- Correlated state changes without observed messages
- Sudden baseline shifts across multiple agents

### Assumption 4: Stationarity

The baseline distribution is stationary or slowly varying.

**Implications**:
- Rolling window statistics are meaningful
- Historical baselines predict future behavior

**Violation indicators**:
- Consistent upward/downward trends in metrics
- Sudden regime changes

### Limitations

1. **Correlation, not causation**: High IQx indicates state change correlated with message receipt, not proven causal influence.

2. **Embedding sensitivity**: Results depend on embedding model quality and semantic coverage.

3. **Cold start**: First interactions have no baseline, requiring imputation or exclusion.

4. **Delayed effects**: Influence that manifests after several interactions is not captured.

5. **Context truncation**: Long contexts may be truncated by embedding models, losing information.

---

## Computational Considerations

### Time Complexity

| Operation | Complexity | Notes |
|-----------|------------|-------|
| Embedding | O(L) | L = sequence length |
| L2 Drift | O(d) | d = embedding dimension |
| IQx | O(1) | Constant time division |
| Rolling Median | O(w log w) | w = window size |
| Adjacency Matrix | O(n²) | n = number of agents |
| Spectral Radius | O(n³) | Full eigendecomposition |

### Space Complexity

| Storage | Complexity | Notes |
|---------|------------|-------|
| Events | O(E) | E = number of events |
| Baselines | O(n × w) | Per-agent rolling windows |
| Adjacency Matrix | O(n²) | Dense representation |

### Numerical Stability

The epsilon parameter ($\varepsilon$) prevents division by zero:

```
IQx = drift / (baseline + epsilon)
Z = (IQx - mean) / (std + epsilon)
```

Recommended values:
- Research: $\varepsilon = 10^{-10}$ (maximum precision)
- Production: $\varepsilon = 10^{-6}$ (balanced)
- Fast inference: $\varepsilon = 10^{-4}$ (minimal overhead)

---

## References

1. Kempe, D., Kleinberg, J., & Tardos, É. (2003). Maximizing the spread of influence through a social network. *KDD*.

2. Chung, F. (1997). *Spectral Graph Theory*. American Mathematical Society.

3. Chandola, V., Banerjee, A., & Kumar, V. (2009). Anomaly detection: A survey. *ACM Computing Surveys*.

4. Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *EMNLP*.

5. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*. Lawrence Erlbaum Associates.

---

*This document accompanies the TraceIQ v0.3.0 release and the associated IEEE research paper.*
