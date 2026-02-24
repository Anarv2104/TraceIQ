# Research Experiments

TraceIQ includes a reproducible research testbed in the `experiments/` directory.

## Quick Start

```bash
# Install research dependencies
pip install -e ".[research]"

# Run experiments
python experiments/run_exp1_wrong_hint.py
python experiments/run_exp2_propagation.py
python experiments/run_exp3_mitigation.py

# Generate plots
python experiments/plot_all.py
```

## Output Locations

- **Results**: `experiments/results/*.csv`
- **Plots**: `experiments/plots/*.png`

## Experiments Overview

### Experiment 1: Wrong Hint Infection

Tests whether TraceIQ detects influence when an "influencer" agent provides wrong hints to a "solver" agent.

**Conditions:**
- A (Baseline): Solver computes independently
- B (Correct Hint): Influencer provides correct answers
- C (Wrong Hint): Influencer provides wrong answers

**Key outputs:**
- `exp1_accuracy.png`: Accuracy comparison
- `exp1_iqx_box.png`: IQx distribution by condition
- `exp1_alert_rate.png`: Alert rates

### Experiment 2: Multi-hop Propagation

Tracks influence propagation through a chain of agents (A → B → C → D).

**Key outputs:**
- `exp2_propagation_risk.png`: PR over time
- `exp2_agent_influence.png`: Accumulated IQx per agent

### Experiment 3: Mitigation Policy

Tests a mitigation guard that quarantines suspicious interactions.

**Key outputs:**
- `exp3_mitigation_compare.png`: Accuracy with/without mitigation

## Reproducibility

All experiments use fixed random seeds:

```python
RANDOM_SEED = 42
```

Design choices for reproducibility:
- **No LLM APIs**: Agents are deterministic rules-based
- **MockEmbedder**: Hash-based embeddings (no neural model variance)
- **Fixed seeds**: numpy.random.seed set consistently

## Detailed Documentation

See [experiments/README.md](../experiments/README.md) for:
- Detailed experiment descriptions
- CSV column definitions
- MCP server usage
- Extending the testbed
