# TraceIQ

**Measure AI-to-AI Influence in Multi-Agent Systems**

[![PyPI version](https://img.shields.io/pypi/v/traceiq.svg?cacheSeconds=60)](https://pypi.org/project/traceiq/)
[![Python versions](https://img.shields.io/pypi/pyversions/traceiq.svg?cacheSeconds=60)](https://pypi.org/project/traceiq/)
[![License](https://img.shields.io/pypi/l/traceiq.svg?cacheSeconds=60)](https://pypi.org/project/traceiq/)

---

## What It Is

- Tracks and quantifies how AI agents influence each other in multi-agent systems
- Uses semantic embeddings to measure state changes between interactions
- Provides network-level analysis of influence propagation
- Includes anomaly detection for unusual influence patterns
- Supports research with IEEE-standard metrics (IQx, RWI, Z-score)

## What It Does NOT Measure

- **Causal attribution**: TraceIQ detects correlation, not causation
- **Intent**: Cannot determine if influence is intentional or benign
- **Content analysis**: Does not parse meaning, only embedding similarity
- **Internal state**: Measures observable outputs, not agent cognition

## Research Context

TraceIQ is developed as part of research on AI-to-AI influence and multi-agent coordination. The metrics are grounded in a formal mathematical framework documented in [MATH.md](MATH.md) and [docs/THEORY.md](docs/THEORY.md).

---

## Quickstart

```python
from traceiq import InfluenceTracker

# Create tracker (use_mock_embedder=True for testing without sentence-transformers)
tracker = InfluenceTracker(use_mock_embedder=True)

# Track an interaction
result = tracker.track_event(
    sender_id="agent_a",
    receiver_id="agent_b",
    sender_content="We should all switch to renewable energy!",
    receiver_content="You make a good point. Renewables are the future.",
)

# Access metrics
print(f"State Drift: {result['drift_l2_state']}")
print(f"IQx: {result['IQx']}")
print(f"Alert: {result['alert']}")
print(f"Risk Level: {result['risk_level']}")

tracker.close()
```

## Installation

```bash
# Core installation
pip install traceiq

# With plotting support (matplotlib)
pip install "traceiq[plot]"

# With real embeddings (sentence-transformers)
pip install "traceiq[embedding]"

# Everything included
pip install "traceiq[all]"
```

### Development Installation

```bash
git clone https://github.com/Anarv2104/TraceIQ.git
cd TraceIQ
pip install -e ".[all,dev]"
```

---

## Metrics Overview

| Metric | What It Captures |
|--------|------------------|
| **State Drift** | How much a receiver's output changed after an interaction |
| **Influence Quotient (IQx)** | Normalized influence relative to baseline responsiveness |
| **Propagation Risk (PR)** | Network-level instability (spectral radius > 1.0 = amplification) |
| **Risk-Weighted Influence (RWI)** | IQx adjusted for sender's attack surface |
| **Z-score** | Anomaly detection (values > threshold trigger alerts) |

For detailed metric documentation, see [docs/metrics.md](docs/metrics.md).

---

## Basic Usage

### Track Multiple Interactions

```python
interactions = [
    {
        "sender_id": "agent_a",
        "receiver_id": "agent_b",
        "sender_content": "AI will transform healthcare completely.",
        "receiver_content": "Yes, medical AI is very promising.",
    },
    {
        "sender_id": "agent_b",
        "receiver_id": "agent_c",
        "sender_content": "Healthcare AI needs careful regulation.",
        "receiver_content": "Agreed, we need safety standards.",
    },
]

results = tracker.bulk_track(interactions)

for r in results:
    print(f"{r['sender_id']} -> {r['receiver_id']}: IQx={r['IQx']}")
```

### Generate Summary Report

```python
summary = tracker.summary(top_n=5)

print(f"Total Events: {summary.total_events}")
print(f"Top Influencers: {summary.top_influencers}")
print(f"Most Susceptible: {summary.top_susceptible}")
```

### Export Data

```python
tracker.export_csv("influence_data.csv")
tracker.export_jsonl("influence_data.jsonl")
```

### Monitor Anomalies

```python
# Get propagation risk
pr = tracker.get_propagation_risk()
if pr > 1.0:
    print(f"Warning: PR={pr:.2f} - influence may amplify")

# Get anomaly alerts (only valid metrics, not cold-start)
alerts = tracker.get_alerts()
for alert in alerts:
    print(f"Alert: Z={alert.Z_score:.2f}")
```

---

## CLI Usage

```bash
# Initialize a database
traceiq init --db analysis.db

# Ingest interactions from JSONL file
traceiq ingest interactions.jsonl --db analysis.db

# View summary report
traceiq summary --db analysis.db

# Export data
traceiq export --db analysis.db -o results.csv --format csv

# Generate plots
traceiq plot heatmap --db analysis.db -o heatmap.png
traceiq plot network --db analysis.db -o network.png

# IEEE Metrics Commands
traceiq propagation-risk --db analysis.db
traceiq alerts --db analysis.db --threshold 2.0
traceiq risky-agents --db analysis.db --top-n 10
```

For full CLI reference, see [docs/cli.md](docs/cli.md).

---

## Visualizations

```python
from traceiq.plotting import (
    plot_influence_heatmap,
    plot_top_influencers,
    plot_influence_network,
    plot_drift_over_time,
)

# Influence heatmap
plot_influence_heatmap(tracker.graph, output_path="heatmap.png")

# Top influencers bar chart
plot_top_influencers(tracker.graph, n=10, output_path="influencers.png")

# Network visualization
plot_influence_network(tracker.graph, output_path="network.png")
```

---

## Research Testbed

TraceIQ includes reproducible experiments for studying AI-to-AI influence:

```bash
pip install -e ".[research]"

# Run experiments
python experiments/run_exp1_wrong_hint.py   # Wrong hint infection
python experiments/run_exp2_propagation.py  # Multi-hop propagation
python experiments/run_exp3_mitigation.py   # Mitigation policy

# Generate plots
python experiments/plot_all.py
```

See [experiments/README.md](experiments/README.md) for details.

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/quickstart.md](docs/quickstart.md) | Getting started guide |
| [docs/metrics.md](docs/metrics.md) | Detailed metric documentation |
| [docs/configuration.md](docs/configuration.md) | Configuration reference |
| [docs/architecture.md](docs/architecture.md) | System architecture |
| [docs/integration.md](docs/integration.md) | Integration patterns |
| [docs/cli.md](docs/cli.md) | CLI reference |
| [docs/THEORY.md](docs/THEORY.md) | Mathematical foundations |
| [MATH.md](MATH.md) | IEEE metric formulas |

---

## Dependencies

**Core:**
- pydantic >= 2.0
- numpy >= 1.24
- networkx >= 3.0
- click >= 8.0
- rich >= 13.0

**Optional:**
- sentence-transformers >= 2.2 (`[embedding]`)
- matplotlib >= 3.7 (`[plot]`)
- pandas >= 2.0, scipy >= 1.10 (`[research]`)

---

## Development

```bash
# Install dev dependencies
pip install -e ".[all,dev]"

# Run linter
ruff check src/ tests/

# Run tests
pytest -v

# Build documentation
pip install mkdocs mkdocs-material
mkdocs serve
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run tests (`pytest`)
4. Run linter (`ruff check src/ tests/`)
5. Commit your changes
6. Push to the branch
7. Open a Pull Request

## Citation

If you use TraceIQ in your research, please cite:

```bibtex
@software{traceiq,
  title = {TraceIQ: Measure AI-to-AI Influence in Multi-Agent Systems},
  year = {2024},
  url = {https://github.com/Anarv2104/TraceIQ}
}
```

---

*Built for AI safety researchers and multi-agent system developers*
