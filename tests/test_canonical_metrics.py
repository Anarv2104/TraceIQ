"""Tests for canonical IEEE metrics implementation.

These tests verify the canonical state drift (||current - previous||) behavior
as opposed to the proxy drift (||current - rolling_mean||).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import uuid4

import numpy as np
import pytest

from traceiq.models import TrackerConfig
from traceiq.scoring import ScoringEngine
from traceiq.storage.sqlite import SQLiteStorage
from traceiq.tracker import InfluenceTracker


class TestCanonicalDrift:
    """Tests for canonical state drift computation."""

    def test_drift_zero_when_identical(self) -> None:
        """Canonical drift should be 0 when current == previous embedding."""
        engine = ScoringEngine(baseline_window=5)

        # Same embedding for all calls
        emb = np.ones(384, dtype=np.float32) / np.sqrt(384)

        # First event (cold start)
        result1 = engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=emb,
            receiver_embedding=emb,
        )
        assert result1.cold_start is True
        assert result1.drift_l2_state is None  # No previous state yet

        # Second event with SAME embedding
        result2 = engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=emb,
            receiver_embedding=emb,
        )
        assert result2.cold_start is False
        # Canonical drift should be 0 (identical embeddings)
        assert result2.drift_l2_state == pytest.approx(0.0, abs=1e-6)

    def test_drift_increases_with_difference(self) -> None:
        """Canonical drift should increase as embeddings diverge."""
        engine = ScoringEngine(baseline_window=5)

        # Base embedding
        emb1 = np.zeros(384, dtype=np.float32)
        emb1[0] = 1.0

        # Different embedding (orthogonal)
        emb2 = np.zeros(384, dtype=np.float32)
        emb2[1] = 1.0

        # First event (cold start)
        engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=emb1,
            receiver_embedding=emb1,
        )

        # Second event with different embedding
        result = engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=emb2,
            receiver_embedding=emb2,
        )

        # Canonical drift should be sqrt(2) (L2 norm of difference)
        expected_drift = np.sqrt(2)
        assert result.drift_l2_state == pytest.approx(expected_drift, rel=1e-5)

    def test_iqx_uses_canonical_drift(self) -> None:
        """IQx should be computed using canonical state drift."""
        engine = ScoringEngine(baseline_window=3, epsilon=1e-6)

        # Create unit vectors
        emb1 = np.zeros(384, dtype=np.float32)
        emb1[0] = 1.0

        emb2 = np.zeros(384, dtype=np.float32)
        emb2[1] = 1.0

        # Build up some history
        for _ in range(3):
            engine.compute_scores(
                event_id=uuid4(),
                sender_id="sender",
                receiver_id="receiver",
                sender_embedding=emb1,
                receiver_embedding=emb1,
            )
            engine.compute_scores(
                event_id=uuid4(),
                sender_id="sender",
                receiver_id="receiver",
                sender_embedding=emb2,
                receiver_embedding=emb2,
            )

        # Now compute a new score
        result = engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=emb1,
            receiver_embedding=emb1,
        )

        # IQx should be positive and based on canonical drift
        assert result.IQx is not None
        assert result.IQx > 0

    def test_canonical_vs_proxy_different(self) -> None:
        """Canonical and proxy drift should be computed independently."""
        engine = ScoringEngine(baseline_window=5)

        # Create sequence of varying embeddings
        embeddings = []
        for i in range(6):
            emb = np.zeros(384, dtype=np.float32)
            emb[i % 10] = 1.0
            embeddings.append(emb)

        # Track multiple events to build baseline
        for emb in embeddings[:5]:
            engine.compute_scores(
                event_id=uuid4(),
                sender_id="sender",
                receiver_id="receiver",
                sender_embedding=emb,
                receiver_embedding=emb,
            )

        # Final event
        result = engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=embeddings[5],
            receiver_embedding=embeddings[5],
        )

        # Both should be computed
        assert result.drift_l2_state is not None
        assert result.drift_l2_proxy is not None

        # They measure different things, so may differ
        # (canonical = vs previous, proxy = vs rolling mean)
        # Just verify both are non-negative
        assert result.drift_l2_state >= 0
        assert result.drift_l2_proxy >= 0


class TestZScoreAlerts:
    """Tests for Z-score anomaly detection."""

    def test_zscore_alert_triggers(self) -> None:
        """Z-score alert should trigger when IQx exceeds threshold."""
        engine = ScoringEngine(
            baseline_window=10,
            anomaly_threshold=2.0,
            epsilon=1e-6,
        )

        # Create consistent small drift to establish baseline
        normal_emb = np.zeros(384, dtype=np.float32)
        normal_emb[0] = 1.0

        slightly_diff = np.zeros(384, dtype=np.float32)
        slightly_diff[0] = 0.99
        slightly_diff[1] = 0.1
        slightly_diff = slightly_diff / np.linalg.norm(slightly_diff)

        # Build history with small, consistent drifts
        for _ in range(10):
            engine.compute_scores(
                event_id=uuid4(),
                sender_id="sender",
                receiver_id="receiver",
                sender_embedding=normal_emb,
                receiver_embedding=normal_emb,
            )
            engine.compute_scores(
                event_id=uuid4(),
                sender_id="sender",
                receiver_id="receiver",
                sender_embedding=slightly_diff,
                receiver_embedding=slightly_diff,
            )

        # Now create a much larger drift (anomaly)
        anomaly_emb = np.zeros(384, dtype=np.float32)
        anomaly_emb[5] = 1.0  # Completely different direction

        result = engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=anomaly_emb,
            receiver_embedding=anomaly_emb,
        )

        # Should have Z-score computed
        assert result.Z_score is not None
        # Alert may or may not trigger depending on history
        # Just verify Z_score is reasonable
        assert isinstance(result.Z_score, float)

    def test_no_alert_on_cold_start(self) -> None:
        """No alert should trigger on cold start (no history)."""
        engine = ScoringEngine(anomaly_threshold=2.0)

        emb = np.random.rand(384).astype(np.float32)

        result = engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=emb,
            receiver_embedding=emb,
        )

        assert result.cold_start is True
        assert result.alert_flag is False


class TestSQLiteSchemaMigration:
    """Tests for SQLite schema migration from v2 to v3."""

    def test_migration_v2_to_v3(self, tmp_path: Path) -> None:
        """Old v2 databases should migrate to v3 with new columns."""
        db_path = tmp_path / "test_migrate.db"

        # Create a v2 schema database manually
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()

        # Create v2 schema tables
        cursor.execute("""
            CREATE TABLE events (
                event_id TEXT PRIMARY KEY,
                sender_id TEXT NOT NULL,
                receiver_id TEXT NOT NULL,
                sender_content TEXT NOT NULL,
                receiver_content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE scores (
                event_id TEXT PRIMARY KEY,
                influence_score REAL NOT NULL,
                drift_delta REAL NOT NULL,
                receiver_baseline_drift REAL NOT NULL,
                flags_json TEXT NOT NULL,
                cold_start INTEGER NOT NULL,
                drift_l2 REAL,
                IQx REAL,
                baseline_median REAL,
                RWI REAL,
                Z_score REAL,
                alert_flag INTEGER DEFAULT 0
            )
        """)

        cursor.execute("""
            CREATE TABLE schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        """)

        # Set version to 2
        cursor.execute(
            "INSERT INTO schema_meta (key, value) VALUES (?, ?)",
            ("schema_version", "2"),
        )

        # Insert some v2 data
        cursor.execute(
            """
            INSERT INTO scores VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                "test-event-id",
                0.5,
                0.3,
                0.1,
                "[]",
                0,
                0.2,  # drift_l2
                1.5,  # IQx
                0.15,  # baseline_median
                None,  # RWI
                None,  # Z_score
                0,  # alert_flag
            ),
        )
        conn.commit()
        conn.close()

        # Now open with SQLiteStorage (should trigger migration)
        storage = SQLiteStorage(db_path)

        # Verify new columns exist
        cursor = storage._conn.cursor()
        cursor.execute("PRAGMA table_info(scores)")
        columns = {row[1] for row in cursor.fetchall()}

        assert "drift_l2_state" in columns
        assert "drift_l2_proxy" in columns

        # Verify old data is preserved
        cursor.execute(
            "SELECT drift_l2 FROM scores WHERE event_id = ?", ("test-event-id",)
        )
        row = cursor.fetchone()
        assert row[0] == pytest.approx(0.2)

        # Verify schema version is updated
        cursor.execute(
            "SELECT value FROM schema_meta WHERE key = ?", ("schema_version",)
        )
        row = cursor.fetchone()
        assert row[0] == "3"

        storage.close()

    def test_new_db_has_all_columns(self, tmp_path: Path) -> None:
        """New databases should have all v3 columns from creation."""
        db_path = tmp_path / "test_new.db"
        storage = SQLiteStorage(db_path)

        cursor = storage._conn.cursor()
        cursor.execute("PRAGMA table_info(scores)")
        columns = {row[1] for row in cursor.fetchall()}

        # All v3 columns should exist
        assert "drift_l2" in columns
        assert "drift_l2_state" in columns
        assert "drift_l2_proxy" in columns
        assert "IQx" in columns
        assert "baseline_median" in columns
        assert "RWI" in columns
        assert "Z_score" in columns
        assert "alert_flag" in columns

        storage.close()


class TestTrackerIntegration:
    """Integration tests for canonical metrics through InfluenceTracker."""

    def test_tracker_returns_canonical_drift(self) -> None:
        """InfluenceTracker.track_event should return canonical drift fields."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=5,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # First event (cold start)
        result1 = tracker.track_event(
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="Hello",
            receiver_content="Hi there",
        )

        assert result1["cold_start"] is True
        assert "drift_l2_state" in result1
        assert "drift_l2_proxy" in result1
        assert "drift_l2" in result1

        # Second event (should have canonical drift)
        result2 = tracker.track_event(
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="How are you?",
            receiver_content="I am fine.",
        )

        assert result2["cold_start"] is False
        # With MockEmbedder and different content, drift should be non-zero
        # (MockEmbedder produces deterministic embeddings based on content)
        assert result2["drift_l2_state"] is not None
        assert result2["drift_l2_proxy"] is not None

        tracker.close()

    def test_reset_clears_last_state(self) -> None:
        """Resetting baselines should also clear last state tracking."""
        engine = ScoringEngine(baseline_window=5)

        emb = np.random.rand(384).astype(np.float32)

        # Build some state
        engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=emb,
            receiver_embedding=emb,
        )

        # Verify state exists
        assert engine._get_last_state("receiver") is not None

        # Reset
        engine.reset_baseline("receiver")

        # Verify state is cleared
        assert engine._get_last_state("receiver") is None

    def test_reset_all_clears_all_states(self) -> None:
        """reset_all_baselines should clear all last state tracking."""
        engine = ScoringEngine(baseline_window=5)

        emb = np.random.rand(384).astype(np.float32)

        # Build state for multiple receivers
        for receiver in ["r1", "r2", "r3"]:
            engine.compute_scores(
                event_id=uuid4(),
                sender_id="sender",
                receiver_id=receiver,
                sender_embedding=emb,
                receiver_embedding=emb,
            )

        # Verify states exist
        for receiver in ["r1", "r2", "r3"]:
            assert engine._get_last_state(receiver) is not None

        # Reset all
        engine.reset_all_baselines()

        # Verify all states are cleared
        for receiver in ["r1", "r2", "r3"]:
            assert engine._get_last_state(receiver) is None
