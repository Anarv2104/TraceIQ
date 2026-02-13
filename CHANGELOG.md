# Changelog

All notable changes to TraceIQ will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2024-XX-XX

### Added

- **Truncation flags**: `sender_truncated` and `receiver_truncated` flags are added to score results when content exceeds `max_content_length`
- **Real batch embeddings**: `embed_batch()` now uses true batch encoding via `model.encode(list)` for better performance
- **Deterministic ordering**: All ranking outputs (`top_influencers`, `top_susceptible`, `top_influenced`) are now sorted by score descending, then agent_id ascending for consistent results
- **CLI validation**: The `ingest` command now validates that both `sender_content` and `receiver_content` are present in each JSONL record, with line number reporting on errors
- **Pre-commit hooks**: Added `.pre-commit-config.yaml` with ruff format and lint hooks
- **New tests**: Added comprehensive PyPI safety tests for imports, schema compatibility, caching, and deterministic behavior

### Changed

- **max_content_length default**: Increased from 512 to 5000 characters
- **Embedding cache key**: Cache key is now based on the hash of truncated content only, not full content
- **Schema migration**: Old databases with single `content` column now raise `RuntimeError` instead of auto-migrating

### Fixed

- **Susceptibility plot label**: Changed X-axis label from "Total Incoming Influence" to "Total Incoming Drift" to accurately reflect the metric

### Documentation

- Added detailed docstring in `scoring.py` explaining baseline definition and influence scoring algorithms

## [0.1.0] - 2024-XX-XX

### Added

- Initial release
- Core `InfluenceTracker` class for tracking AI-to-AI interactions
- Support for memory and SQLite storage backends
- SentenceTransformer-based embeddings with LRU caching
- Mock embedder for testing without heavy dependencies
- Drift and influence scoring with configurable thresholds
- NetworkX-based influence graph analytics
- Matplotlib plotting functions (drift over time, heatmaps, bar charts, network graphs)
- CLI with init, ingest, summary, export, and plot commands
- CSV and JSONL export functions
- Comprehensive test suite
- MkDocs documentation
