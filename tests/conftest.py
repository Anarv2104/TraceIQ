"""Pytest configuration and fixtures."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from traceiq.embeddings import MockEmbedder
from traceiq.models import InteractionEvent, ScoreResult, TrackerConfig
from traceiq.scoring import ScoringEngine
from traceiq.storage import MemoryStorage, SQLiteStorage
from traceiq.tracker import InfluenceTracker


@pytest.fixture
def mock_embedder() -> MockEmbedder:
    """Create a mock embedder with fixed seed."""
    return MockEmbedder(embedding_dim=384, seed=42)


@pytest.fixture
def scoring_engine() -> ScoringEngine:
    """Create a scoring engine with default settings."""
    return ScoringEngine(
        baseline_window=5,
        drift_threshold=0.3,
        influence_threshold=0.5,
    )


@pytest.fixture
def memory_storage() -> MemoryStorage:
    """Create an in-memory storage backend."""
    return MemoryStorage()


@pytest.fixture
def sqlite_storage(tmp_path: Path) -> SQLiteStorage:
    """Create a SQLite storage backend with temp file."""
    db_path = tmp_path / "test.db"
    storage = SQLiteStorage(db_path)
    yield storage
    storage.close()


@pytest.fixture
def tracker() -> InfluenceTracker:
    """Create an InfluenceTracker with mock embedder."""
    config = TrackerConfig(
        storage_backend="memory",
        baseline_window=5,
        drift_threshold=0.3,
        influence_threshold=0.5,
        random_seed=42,
    )
    tracker = InfluenceTracker(config=config, use_mock_embedder=True)
    yield tracker
    tracker.close()


@pytest.fixture
def sqlite_tracker(tmp_path: Path) -> InfluenceTracker:
    """Create an InfluenceTracker with SQLite backend."""
    db_path = tmp_path / "test.db"
    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=str(db_path),
        baseline_window=5,
        drift_threshold=0.3,
        influence_threshold=0.5,
        random_seed=42,
    )
    tracker = InfluenceTracker(config=config, use_mock_embedder=True)
    yield tracker
    tracker.close()


@pytest.fixture
def sample_events() -> list[InteractionEvent]:
    """Create sample interaction events."""
    from datetime import datetime, timedelta, timezone
    from uuid import uuid4

    base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
    events = []

    for i in range(10):
        events.append(
            InteractionEvent(
                event_id=uuid4(),
                sender_id=f"agent_{i % 3}",
                receiver_id=f"agent_{(i + 1) % 3}",
                content=f"Message {i} from sender",
                timestamp=base_time + timedelta(minutes=i),
                metadata={"index": i},
            )
        )

    return events


@pytest.fixture
def sample_scores(sample_events: list[InteractionEvent]) -> list[ScoreResult]:
    """Create sample score results matching sample events."""
    scores = []
    for i, event in enumerate(sample_events):
        scores.append(
            ScoreResult(
                event_id=event.event_id,
                influence_score=0.3 + (i % 5) * 0.1,
                drift_delta=0.2 + (i % 4) * 0.1,
                receiver_baseline_drift=0.05,
                flags=["high_drift"] if i % 3 == 0 else [],
                cold_start=i == 0,
            )
        )
    return scores


@pytest.fixture
def random_embeddings() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create random embeddings for testing."""
    rng = np.random.default_rng(42)
    e1 = rng.random(384).astype(np.float32)
    e2 = rng.random(384).astype(np.float32)
    e3 = rng.random(384).astype(np.float32)

    # Normalize
    e1 = e1 / np.linalg.norm(e1)
    e2 = e2 / np.linalg.norm(e2)
    e3 = e3 / np.linalg.norm(e3)

    return e1, e2, e3
