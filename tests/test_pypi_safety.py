"""Tests for PyPI safety - import, schema, caching, determinism."""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

import pytest


class TestImportSafety:
    """Test that imports work without optional dependencies."""

    def test_import_traceiq_without_matplotlib(self) -> None:
        """Import traceiq should work without matplotlib installed."""
        # This test validates that the main package imports without matplotlib
        # The actual import happens at module level, this just validates it worked
        import traceiq

        assert hasattr(traceiq, "InfluenceTracker")
        assert hasattr(traceiq, "TrackerConfig")
        assert hasattr(traceiq, "__version__")

    def test_import_models(self) -> None:
        """Import models should work."""
        from traceiq.models import (
            InteractionEvent,
            ScoreResult,
            SummaryReport,
            TrackerConfig,
        )

        assert InteractionEvent is not None
        assert ScoreResult is not None
        assert SummaryReport is not None
        assert TrackerConfig is not None

    def test_import_tracker(self) -> None:
        """Import tracker should work."""
        from traceiq.tracker import InfluenceTracker

        assert InfluenceTracker is not None


class TestSchemaCompatibility:
    """Test database schema handling."""

    def test_new_db_has_correct_schema(self, tmp_path: Path) -> None:
        """A new database should have both content columns."""
        from traceiq.storage import SQLiteStorage

        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(db_path)

        cursor = storage._conn.cursor()
        cursor.execute("PRAGMA table_info(events)")
        columns = {row[1] for row in cursor.fetchall()}

        assert "sender_content" in columns
        assert "receiver_content" in columns
        assert "content" not in columns

        storage.close()

    def test_old_schema_raises_error(self, tmp_path: Path) -> None:
        """Opening a database with old schema should raise RuntimeError."""
        from traceiq.storage import SQLiteStorage

        db_path = tmp_path / "old.db"

        # Create a database with old schema (single content column)
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                sender_id TEXT NOT NULL,
                receiver_id TEXT NOT NULL,
                content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
        """)
        conn.commit()
        conn.close()

        # Now try to open with SQLiteStorage - should fail
        with pytest.raises(RuntimeError) as exc_info:
            SQLiteStorage(db_path)

        assert "older TraceIQ version" in str(exc_info.value)
        assert "incompatible" in str(exc_info.value)

    def test_insert_and_retrieve_event(self, tmp_path: Path) -> None:
        """Test basic CRUD operations on new schema."""
        from datetime import datetime, timezone
        from uuid import uuid4

        from traceiq.models import InteractionEvent
        from traceiq.storage import SQLiteStorage

        db_path = tmp_path / "crud.db"
        storage = SQLiteStorage(db_path)

        event = InteractionEvent(
            event_id=uuid4(),
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="Hello from sender",
            receiver_content="Hello from receiver",
            timestamp=datetime.now(timezone.utc),
            metadata={"test": True},
        )

        storage.store_event(event)
        retrieved = storage.get_event(event.event_id)

        assert retrieved is not None
        assert retrieved.sender_content == "Hello from sender"
        assert retrieved.receiver_content == "Hello from receiver"

        storage.close()


class TestDeterministicOrdering:
    """Test that outputs are deterministically ordered."""

    def test_summary_ordering_is_deterministic(self) -> None:
        """Same events should produce same ordering in summary."""
        from traceiq.models import TrackerConfig

        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=20,
            baseline_k=5,
            random_seed=42,
        )

        # Run twice and compare
        results1 = self._run_tracking_session(config)
        results2 = self._run_tracking_session(config)

        assert results1["top_influencers"] == results2["top_influencers"]
        assert results1["top_susceptible"] == results2["top_susceptible"]

    def _run_tracking_session(self, config) -> dict:
        """Run a tracking session and return summary data."""
        from traceiq.tracker import InfluenceTracker

        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Add events in same order
        interactions = [
            ("agent_a", "agent_b", "Message 1", "Response 1"),
            ("agent_b", "agent_c", "Message 2", "Response 2"),
            ("agent_a", "agent_c", "Message 3", "Response 3"),
            ("agent_c", "agent_a", "Message 4", "Response 4"),
            ("agent_b", "agent_a", "Message 5", "Response 5"),
        ]

        for sender, receiver, s_content, r_content in interactions:
            tracker.track_event(sender, receiver, s_content, r_content)

        summary = tracker.summary(top_n=5)
        tracker.close()

        return {
            "top_influencers": summary.top_influencers,
            "top_susceptible": summary.top_susceptible,
        }

    def test_graph_ranking_tiebreaker(self) -> None:
        """Agents with same score should be ordered by agent_id."""
        from uuid import uuid4

        from traceiq.graph import InfluenceGraph
        from traceiq.models import InteractionEvent, ScoreResult

        graph = InfluenceGraph()

        # Create events with same influence scores for different agents
        for sender in ["zebra", "alpha", "beta"]:
            event = InteractionEvent(
                event_id=uuid4(),
                sender_id=sender,
                receiver_id="receiver",
                sender_content="test",
                receiver_content="test",
            )
            score = ScoreResult(
                event_id=event.event_id,
                influence_score=0.5,  # Same score for all
                drift_delta=0.3,
                receiver_baseline_drift=0.1,
            )
            graph.add_interaction(event, score)

        top = graph.top_influencers(n=3)

        # With same scores, should be sorted by agent_id ascending
        agent_ids = [t[0] for t in top]
        assert agent_ids == ["alpha", "beta", "zebra"]


class TestEmbeddingCache:
    """Test embedding cache behavior."""

    def test_same_content_uses_cache(self) -> None:
        """Embedding same content twice should hit cache."""
        from traceiq.embeddings import MockEmbedder

        embedder = MockEmbedder(seed=42)

        # Embed same content twice
        content = "This is test content for caching"
        embedder.embed(content)
        embedder.embed(content)

        info = embedder.cache_info
        assert info["hits"] == 1
        assert info["misses"] == 1
        assert info["size"] == 1

    def test_different_content_misses_cache(self) -> None:
        """Embedding different content should miss cache."""
        from traceiq.embeddings import MockEmbedder

        embedder = MockEmbedder(seed=42)

        embedder.embed("Content A")
        embedder.embed("Content B")

        info = embedder.cache_info
        assert info["hits"] == 0
        assert info["misses"] == 2
        assert info["size"] == 2

    def test_cache_key_uses_truncated_content(self) -> None:
        """Cache key should be based on truncated content only."""
        from traceiq.embeddings import MockEmbedder

        embedder = MockEmbedder(seed=42, max_content_length=10)

        # These should have the same cache key (same first 10 chars)
        content1 = "0123456789AAAA"
        content2 = "0123456789BBBB"

        emb1 = embedder.embed(content1)
        emb2 = embedder.embed(content2)

        info = embedder.cache_info
        assert info["hits"] == 1  # Second should hit cache
        assert info["misses"] == 1

        # Embeddings should be identical
        import numpy as np

        assert np.allclose(emb1, emb2)


class TestExportContents:
    """Test that exports contain both content fields."""

    def test_csv_export_has_both_contents(self, tmp_path: Path) -> None:
        """CSV export should include sender_content and receiver_content."""
        from traceiq.models import TrackerConfig
        from traceiq.tracker import InfluenceTracker

        config = TrackerConfig(storage_backend="memory")
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        tracker.track_event(
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="Sender says hello",
            receiver_content="Receiver responds",
        )

        output_path = tmp_path / "export.csv"
        tracker.export_csv(output_path)
        tracker.close()

        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 1
        assert rows[0]["sender_content"] == "Sender says hello"
        assert rows[0]["receiver_content"] == "Receiver responds"

    def test_jsonl_export_has_both_contents(self, tmp_path: Path) -> None:
        """JSONL export should include sender_content and receiver_content."""
        from traceiq.models import TrackerConfig
        from traceiq.tracker import InfluenceTracker

        config = TrackerConfig(storage_backend="memory")
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        tracker.track_event(
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="Sender says hello",
            receiver_content="Receiver responds",
        )

        output_path = tmp_path / "export.jsonl"
        tracker.export_jsonl(output_path)
        tracker.close()

        with open(output_path, encoding="utf-8") as f:
            record = json.loads(f.readline())

        assert record["sender_content"] == "Sender says hello"
        assert record["receiver_content"] == "Receiver responds"


class TestScoringFlags:
    """Test that scoring flags are set correctly."""

    def test_cold_start_flag(self) -> None:
        """First interaction should have cold_start flag."""
        from traceiq.models import TrackerConfig
        from traceiq.tracker import InfluenceTracker

        config = TrackerConfig(storage_backend="memory")
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        result = tracker.track_event(
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="Hello",
            receiver_content="Hi",
        )
        tracker.close()

        assert result["cold_start"] is True

    def test_truncation_flags(self) -> None:
        """Truncated content should add truncation flags."""
        from traceiq.models import TrackerConfig
        from traceiq.tracker import InfluenceTracker

        config = TrackerConfig(
            storage_backend="memory",
            max_content_length=10,  # Very short to trigger truncation
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # First event to establish baseline
        tracker.track_event(
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="Short",
            receiver_content="Short",
        )

        # Second event with long content that will be truncated
        result = tracker.track_event(
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="This is a very long sender content that exceeds limit",
            receiver_content="This is a very long receiver content that exceeds limit",
        )
        tracker.close()

        assert "sender_truncated" in result["flags"]
        assert "receiver_truncated" in result["flags"]
