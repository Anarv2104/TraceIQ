# Architecture Overview

This document describes TraceIQ's internal architecture and data flow.

## Project Structure

```
TraceIQ/
├── src/traceiq/
│   ├── __init__.py          # Public API exports
│   ├── models.py             # Pydantic data models
│   ├── tracker.py            # Main InfluenceTracker class
│   ├── embeddings.py         # Embedding backends (SentenceTransformer, Mock)
│   ├── scoring.py            # Drift & influence scoring engine
│   ├── metrics.py            # IEEE metric computations (IQx, RWI, Z-score)
│   ├── capabilities.py       # Agent capability registry
│   ├── graph.py              # NetworkX graph analytics
│   ├── plotting.py           # Matplotlib visualizations
│   ├── cli.py                # Click-based CLI
│   ├── export.py             # CSV/JSONL export
│   ├── validity.py           # Cold-start and validity checking
│   ├── risk.py               # Risk score computation
│   ├── policy.py             # Policy engine for mitigation
│   ├── schema.py             # TraceIQEvent schema
│   └── storage/
│       ├── base.py           # Abstract StorageBackend
│       ├── memory.py         # In-memory backend
│       └── sqlite.py         # SQLite backend
├── tests/                    # Pytest test suite
├── experiments/              # Research experiment scripts
├── docs/                     # Documentation
└── examples/                 # Runnable examples
```

## Module Descriptions

| Module | Purpose |
|--------|---------|
| `tracker.py` | Main `InfluenceTracker` class; orchestrates all components |
| `embeddings.py` | Converts text to vectors via sentence-transformers |
| `scoring.py` | Computes drift, influence, baseline management |
| `metrics.py` | Core IEEE metrics (L2 drift, IQx, PR, RWI, Z-score) |
| `graph.py` | Builds influence graph; computes spectral radius |
| `validity.py` | Checks cold-start, determines metric validity |
| `risk.py` | Computes bounded risk scores from metrics |
| `policy.py` | Applies mitigation policies (quarantine, block) |
| `storage/` | Persistence layer (memory or SQLite) |

## Data Flow

```
┌─────────────────┐
│  track_event()  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│    Embedder     │  Convert text → vectors
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ScoringEngine  │  Compute drift, influence, IQx, Z-score
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ check_validity  │  Cold-start detection, confidence
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  compute_risk   │  Bounded risk score [0, 1]
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  PolicyEngine   │  Apply allow/verify/quarantine/block
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Storage     │  Persist event + scores
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ InfluenceGraph  │  Update graph, edge weights
└─────────────────┘
```

## Key Design Decisions

1. **Pydantic models**: All data structures use Pydantic for validation
2. **Pluggable storage**: Abstract `StorageBackend` allows memory/SQLite swap
3. **Mock embedder**: Enables testing without heavy ML dependencies
4. **Validity gating**: Metrics marked invalid during cold-start prevent false alerts
5. **Bounded metrics**: IQx capped, risk scores bounded [0,1] for numerical stability
