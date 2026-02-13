# Contributing to TraceIQ

Thank you for your interest in contributing to TraceIQ! This document provides guidelines and instructions for contributing.

## Development Setup

### Prerequisites

- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/Anarv2104/TraceIQ.git
cd TraceIQ

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode with all dependencies
pip install -e ".[all,dev]"

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

## Code Style

We use [ruff](https://github.com/astral-sh/ruff) for linting and formatting.

```bash
# Format code
ruff format .

# Check for issues
ruff check .

# Auto-fix issues
ruff check . --fix
```

Pre-commit hooks will run these checks automatically before each commit.

## Running Tests

```bash
# Run all tests
pytest -v

# Run specific test file
pytest tests/test_scoring.py -v

# Run with coverage
pytest --cov=traceiq --cov-report=html
```

## Pull Request Process

1. **Fork the repository** and create a branch from `main`
2. **Write tests** for any new functionality
3. **Update documentation** if needed
4. **Run tests and linting** to ensure they pass
5. **Submit a pull request** with a clear description of changes

### PR Title Format

Use conventional commit format:
- `feat: add new feature`
- `fix: resolve issue`
- `docs: update documentation`
- `test: add tests`
- `refactor: improve code structure`

## Code Guidelines

### General

- Keep functions focused and single-purpose
- Use type hints for all function signatures
- Write docstrings for public functions and classes
- Avoid adding unnecessary dependencies

### Testing

- Write tests for all new functionality
- Maintain test coverage above 80%
- Use fixtures from `conftest.py` when possible
- Test edge cases and error conditions

### Documentation

- Update README.md for user-facing changes
- Add docstrings following Google style
- Update CHANGELOG.md for notable changes

## Architecture

### Key Components

- `tracker.py`: Main `InfluenceTracker` class
- `models.py`: Pydantic data models
- `embeddings.py`: Embedding backends
- `scoring.py`: Drift and influence calculations
- `graph.py`: NetworkX-based analytics
- `storage/`: Storage backends (memory, SQLite)
- `plotting.py`: Matplotlib visualizations
- `cli.py`: Click-based CLI
- `export.py`: CSV/JSONL export functions

### Adding New Features

1. **Storage backends**: Implement `StorageBackend` abstract class
2. **Embedders**: Follow `SentenceTransformerEmbedder` interface
3. **CLI commands**: Add to `cli.py` using Click decorators

## Questions?

Open an issue on GitHub or start a discussion.
