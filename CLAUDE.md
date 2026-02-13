# CLAUDE.md - TraceIQ Project Context

## Project Overview

**TraceIQ** is a Python library for measuring AI-to-AI influence in multi-agent systems. It tracks how one agent's outputs influence another agent's behavior using semantic embeddings and provides tools for analysis and visualization.

## Repository

- **GitHub**: https://github.com/Anarv2104/TraceIQ
- **PyPI**: traceiq (pending publication)
- **Version**: 0.1.0

## Project Status: COMPLETE

### Completed Milestones

| Phase | Status | Description |
|-------|--------|-------------|
| Core Implementation | ✅ Done | All source modules implemented |
| Storage Backends | ✅ Done | Memory and SQLite backends |
| Embeddings | ✅ Done | SentenceTransformer + Mock embedder |
| Scoring Engine | ✅ Done | Drift and influence calculations |
| Graph Analytics | ✅ Done | NetworkX-based influence graphs |
| Plotting | ✅ Done | Matplotlib visualizations |
| CLI | ✅ Done | Click-based command line interface |
| Export | ✅ Done | CSV and JSONL export functions |
| Tests | ✅ Done | 40 tests passing |
| Linting | ✅ Done | Ruff passes |
| Documentation | ✅ Done | MkDocs with 7 pages |
| Examples | ✅ Done | 2 runnable examples |
| CI/CD | ✅ Done | GitHub Actions workflow |
| GitHub Push | ✅ Done | All code pushed to repo |

## Architecture

```
TraceIQ/
├── src/traceiq/
│   ├── __init__.py          # Public API: InfluenceTracker, TrackerConfig, models
│   ├── models.py             # Pydantic models (InteractionEvent, ScoreResult, etc.)
│   ├── tracker.py            # Main InfluenceTracker class
│   ├── embeddings.py         # SentenceTransformerEmbedder, MockEmbedder
│   ├── scoring.py            # ScoringEngine with drift/influence calculations
│   ├── graph.py              # InfluenceGraph (NetworkX DiGraph wrapper)
│   ├── plotting.py           # Matplotlib plotting functions
│   ├── cli.py                # Click CLI (traceiq command)
│   ├── export.py             # CSV/JSONL export functions
│   └── storage/
│       ├── base.py           # Abstract StorageBackend
│       ├── memory.py         # MemoryStorage (dict-based)
│       └── sqlite.py         # SQLiteStorage (persistent)
├── tests/                    # Pytest test suite
├── docs/                     # MkDocs documentation
├── examples/                 # Runnable examples
└── pyproject.toml           # Package configuration
```

## Key Algorithms

### Drift Detection
```python
drift_delta = 1 - cosine_similarity(current_embedding, baseline_embedding)
```
- Baseline = rolling mean of last N receiver embeddings
- Higher drift = receiver deviated more from typical behavior

### Influence Scoring
```python
baseline_shift = baseline_after - baseline_before
influence_score = cosine_similarity(sender_embedding, baseline_shift)
```
- Measures correlation between sender content and receiver's behavioral change
- Range: -1.0 to +1.0

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

## Commands

```bash
# Install
pip install -e ".[all,dev]"

# Run tests
pytest -v

# Lint
ruff check src/ tests/

# Build package
python -m build

# Serve docs locally
mkdocs serve

# Deploy docs to GitHub Pages
mkdocs gh-deploy

# Publish to PyPI
twine upload dist/*
```

## Test Results

```
40 tests passing
- test_scoring.py: 12 tests (cosine similarity, cold start, thresholds)
- test_storage_sqlite.py: 11 tests (CRUD, persistence)
- test_export.py: 10 tests (CSV, JSONL output)
- test_plotting.py: 7 tests (visualization generation)
```

## Real Agent Test Results

Tested with sentence-transformers (all-MiniLM-L6-v2):

**Test 1: Agent Debate (12 interactions)**
- 8 high-drift events detected
- 1 high-influence event
- Top influencer: skeptic_agent (+0.28)

**Test 2: Idea Adoption (9 interactions)**
- agent_zero correctly identified as source (+0.70)
- 6 high-drift events after baseline
- Influence chains detected

## Files Changed in This Session

1. Created entire project structure from scratch
2. Implemented all 13 source files
3. Created 4 test files with 40 tests
4. Created 7 documentation pages
5. Created 2 example scripts
6. Set up CI/CD with GitHub Actions
7. Pushed everything to GitHub

## Publication Checklist

- [x] LICENSE file (MIT)
- [x] README.md (comprehensive)
- [x] pyproject.toml (hatchling build)
- [x] All tests passing
- [x] Ruff linting passes
- [x] Package builds successfully
- [x] Documentation complete
- [x] GitHub repository created
- [ ] PyPI publication (pending user action)
- [ ] GitHub Pages deployment (pending user action)

## Notes for Future Development

1. **Embedding Cache**: LRU cache implemented with configurable size
2. **Storage**: Abstract base class allows easy addition of new backends
3. **Graph**: Uses NetworkX, can be extended with more algorithms
4. **Thresholds**: Configurable via TrackerConfig
5. **Mock Embedder**: Deterministic, useful for testing without sentence-transformers

## Contact

Repository: https://github.com/Anarv2104/TraceIQ
