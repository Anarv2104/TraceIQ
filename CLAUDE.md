# CLAUDE.md - TraceIQ Project Context

## Project Overview

**TraceIQ** is a Python library for measuring AI-to-AI influence in multi-agent systems. It tracks how one agent's outputs influence another agent's behavior using semantic embeddings and provides tools for analysis, visualization, and security monitoring.

## Repository

- **GitHub**: https://github.com/Anarv2104/TraceIQ
- **PyPI**: traceiq (pending publication)
- **Version**: 0.3.0

## Project Status: COMPLETE

### v0.3.0 - IEEE Research Framework

| Phase | Status | Description |
|-------|--------|-------------|
| Core Implementation | ✅ Done | All source modules implemented |
| Storage Backends | ✅ Done | Memory and SQLite backends |
| Embeddings | ✅ Done | SentenceTransformer + Mock embedder |
| Scoring Engine | ✅ Done | Drift, influence, IQx, RWI, Z-score |
| Graph Analytics | ✅ Done | NetworkX + spectral radius |
| Plotting | ✅ Done | 10 plot types including IEEE metrics |
| CLI | ✅ Done | Extended with IEEE metric commands |
| Export | ✅ Done | CSV and JSONL export functions |
| IEEE Metrics | ✅ Done | 7 new metrics (IQx, RWI, PR, etc.) |
| Capability Registry | ✅ Done | Attack surface computation |
| Research Scripts | ✅ Done | Simulation, ablation, sensitivity |
| Tests | ✅ Done | 116 tests passing |
| Documentation | ✅ Done | MATH.md + updated docs |

## Architecture

```
TraceIQ/
├── src/traceiq/
│   ├── __init__.py          # Public API exports
│   ├── models.py             # Pydantic models (extended with IEEE fields)
│   ├── tracker.py            # Main InfluenceTracker class
│   ├── embeddings.py         # SentenceTransformerEmbedder, MockEmbedder
│   ├── scoring.py            # ScoringEngine (drift, IQx, RWI, Z-score)
│   ├── metrics.py            # NEW: Core IEEE metric computations
│   ├── capabilities.py       # NEW: CapabilityRegistry for attack surface
│   ├── graph.py              # InfluenceGraph (spectral radius, adjacency)
│   ├── plotting.py           # Extended plotting (IQx heatmap, PR over time)
│   ├── cli.py                # Click CLI (propagation-risk, alerts, etc.)
│   ├── export.py             # CSV/JSONL export functions
│   └── storage/
│       ├── base.py           # Abstract StorageBackend
│       ├── memory.py         # MemoryStorage (dict-based)
│       └── sqlite.py         # SQLiteStorage (extended schema)
├── research/                 # NEW: Research experiment scripts
│   ├── synthetic_simulation.py
│   ├── ablation_study.py
│   └── sensitivity_analysis.py
├── tests/                    # Pytest test suite
├── docs/                     # MkDocs documentation
├── examples/                 # Runnable examples
├── MATH.md                   # NEW: Mathematical framework documentation
└── pyproject.toml           # Package configuration (v0.3.0)
```

## IEEE Metrics Summary

| Metric | Formula | Purpose |
|--------|---------|---------|
| **Drift (L2)** | `D = \|\|s(t+) - s(t-)\|\|_2` | L2 norm state change |
| **IQx** | `IQx = D / (B + ε)` | Normalized influence quotient |
| **Accumulated Influence** | `AI = Σ IQx` | Sum over window |
| **Propagation Risk** | `PR = spectral_radius(W)` | Network instability |
| **Attack Surface** | `AS = Σ p(c)` | Capability-weighted risk |
| **RWI** | `RWI = IQx × AS` | Security-adjusted influence |
| **Z-score** | `Z = (IQx - μ) / (σ + ε)` | Anomaly detection |

## Key Algorithms

### Legacy Drift Detection
```python
drift_delta = 1 - cosine_similarity(current_embedding, baseline_embedding)
```

### IEEE Drift (L2)
```python
drift_l2 = np.linalg.norm(emb_after - emb_before)
```

### Influence Quotient (IQx)
```python
IQx = drift_l2 / (baseline_median + epsilon)
```

### Propagation Risk
```python
spectral_radius = max(abs(eigenvalues(adjacency_matrix)))
```

### Risk-Weighted Influence
```python
RWI = IQx * attack_surface
```

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

## Commands

```bash
# Install
pip install -e ".[all,dev]"

# Run tests
pytest -v

# Lint
ruff check src/ tests/

# Format
ruff format src/ tests/

# Build package
python -m build

# Run research simulation
python research/synthetic_simulation.py

# CLI commands (v0.3.0)
traceiq --help
traceiq propagation-risk --db traceiq.db
traceiq alerts --db traceiq.db --threshold 2.0
traceiq risky-agents --db traceiq.db
traceiq capabilities show
traceiq plot iqx-heatmap --db traceiq.db -o heatmap.png
```

## Example Usage (v0.3.0)

```python
from traceiq import InfluenceTracker, TrackerConfig

# Configure with IEEE metrics
config = TrackerConfig(
    storage_backend="sqlite",
    storage_path="research.db",
    baseline_window=10,
    epsilon=1e-6,
    anomaly_threshold=2.0,
    capability_weights={
        "execute_code": 1.0,
        "admin": 1.5,
    }
)

tracker = InfluenceTracker(config=config)

# Register agent capabilities
tracker.capabilities.register_agent("agent_0", ["execute_code", "admin"])
tracker.capabilities.register_agent("agent_1", ["file_read"])

# Track with full metrics
result = tracker.track_event(
    sender_id="agent_0",
    receiver_id="agent_1",
    sender_content="Execute this command...",
    receiver_content="Executing command...",
)

print(f"IQx: {result['IQx']}")
print(f"RWI: {result['RWI']}")
print(f"Z-score: {result['Z_score']}")
print(f"Alert: {result['alert']}")

# Get propagation risk
pr = tracker.get_propagation_risk()
print(f"Propagation Risk: {pr}")

# Get anomaly alerts
alerts = tracker.get_alerts()
print(f"Alerts: {len(alerts)}")
```

## Test Results

```
116 tests passing (110 passed, 6 skipped)
- test_metrics.py: 25 tests (L2 drift, IQx, PR, AS, RWI, Z-score)
- test_capabilities.py: 21 tests (registry, persistence, attack surface)
- test_scoring.py: 12 tests (cosine similarity, cold start, thresholds)
- test_storage_sqlite.py: 11 tests (CRUD, persistence, schema migration)
- test_export.py: 10 tests (CSV, JSONL output)
- test_plotting.py: 7 tests (visualization generation - skipped without matplotlib)
- test_pypi_safety.py: 16 tests (import safety, schema compat, determinism)
```

## Research Experiments

### Synthetic Simulation
```bash
python research/synthetic_simulation.py
```
- 5-agent simulation with biased injector
- Measures influence propagation
- Generates plots to `research/outputs/`

### Ablation Study
```bash
python research/ablation_study.py
```
- Tests baseline_window [3, 5, 10, 15, 20, 30]
- Measures detection rate vs window size

### Sensitivity Analysis
```bash
python research/sensitivity_analysis.py
```
- Epsilon sensitivity
- Anomaly threshold precision/recall
- Capability weight schemes

## Backward Compatibility

- v0.2.0 databases automatically migrated to v0.3.0 schema
- All v0.2.0 API remains functional
- New IEEE metric fields default to None for old records

## Contact

Repository: https://github.com/Anarv2104/TraceIQ
