"""Tests for scoring module."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from traceiq.scoring import ScoringEngine, cosine_similarity


class TestCosineSimilarity:
    """Tests for cosine_similarity function."""

    def test_identical_vectors(self) -> None:
        """Identical vectors should have similarity 1.0."""
        v = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_orthogonal_vectors(self) -> None:
        """Orthogonal vectors should have similarity 0.0."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(0.0)

    def test_opposite_vectors(self) -> None:
        """Opposite vectors should have similarity -1.0."""
        v1 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == pytest.approx(-1.0)

    def test_zero_vector(self) -> None:
        """Zero vectors should return 0.0."""
        v1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        v2 = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert cosine_similarity(v1, v2) == 0.0
        assert cosine_similarity(v2, v1) == 0.0

    def test_normalized_vectors(self, random_embeddings) -> None:
        """Test with normalized random vectors."""
        e1, e2, _ = random_embeddings
        sim = cosine_similarity(e1, e2)
        assert -1.0 <= sim <= 1.0


class TestScoringEngine:
    """Tests for ScoringEngine class."""

    def test_cold_start(self, scoring_engine: ScoringEngine) -> None:
        """First event for a receiver should be cold start."""
        sender_emb = np.random.rand(384).astype(np.float32)
        receiver_emb = np.random.rand(384).astype(np.float32)

        result = scoring_engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender_1",
            receiver_id="receiver_1",
            sender_embedding=sender_emb,
            receiver_embedding=receiver_emb,
        )

        assert result.cold_start is True
        assert result.drift_delta == 0.0
        assert result.influence_score == 0.0
        assert result.flags == []

    def test_non_cold_start_after_baseline(
        self, scoring_engine: ScoringEngine
    ) -> None:
        """Second event should not be cold start."""
        sender_emb = np.random.rand(384).astype(np.float32)
        receiver_emb1 = np.random.rand(384).astype(np.float32)
        receiver_emb2 = np.random.rand(384).astype(np.float32)

        # First event (cold start)
        scoring_engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender_1",
            receiver_id="receiver_1",
            sender_embedding=sender_emb,
            receiver_embedding=receiver_emb1,
        )

        # Second event (should have baseline now)
        result = scoring_engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender_1",
            receiver_id="receiver_1",
            sender_embedding=sender_emb,
            receiver_embedding=receiver_emb2,
        )

        assert result.cold_start is False
        assert result.drift_delta >= 0.0

    def test_drift_threshold_flags(self) -> None:
        """Test high_drift flag when drift exceeds threshold."""
        engine = ScoringEngine(
            baseline_window=5,
            drift_threshold=0.1,  # Low threshold for testing
            influence_threshold=0.5,
        )

        # Create baseline
        base_emb = np.zeros(384, dtype=np.float32)
        base_emb[0] = 1.0  # Unit vector along first axis

        engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=base_emb,
            receiver_embedding=base_emb,
        )

        # Create very different embedding to trigger high drift
        diff_emb = np.zeros(384, dtype=np.float32)
        diff_emb[1] = 1.0  # Unit vector along second axis (orthogonal)

        result = engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=diff_emb,
            receiver_embedding=diff_emb,
        )

        # Drift should be high (1 - 0 = 1.0)
        assert result.drift_delta > 0.1
        assert "high_drift" in result.flags

    def test_influence_threshold_flags(self) -> None:
        """Test high_influence flag when influence exceeds threshold."""
        engine = ScoringEngine(
            baseline_window=5,
            drift_threshold=0.3,
            influence_threshold=0.3,  # Lower threshold for testing
        )

        sender_emb = np.zeros(384, dtype=np.float32)
        sender_emb[0] = 1.0

        receiver_emb1 = np.zeros(384, dtype=np.float32)
        receiver_emb1[1] = 1.0

        # Create baseline
        engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=sender_emb,
            receiver_embedding=receiver_emb1,
        )

        # Second event where receiver shifts toward sender direction
        receiver_emb2 = np.zeros(384, dtype=np.float32)
        receiver_emb2[0] = 0.7
        receiver_emb2[1] = 0.7
        receiver_emb2 = receiver_emb2 / np.linalg.norm(receiver_emb2)

        result = engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=sender_emb,
            receiver_embedding=receiver_emb2,
        )

        # Baseline shift should align somewhat with sender
        assert result.influence_score != 0.0

    def test_baseline_window_rolling(self) -> None:
        """Test that baseline uses rolling window."""
        engine = ScoringEngine(baseline_window=3)

        # Create sequence of embeddings
        for i in range(5):
            emb = np.zeros(384, dtype=np.float32)
            emb[i % 10] = 1.0
            engine.compute_scores(
                event_id=uuid4(),
                sender_id="sender",
                receiver_id="receiver",
                sender_embedding=emb,
                receiver_embedding=emb,
            )

        # Check baseline is maintained
        baseline = engine._get_baseline("receiver")
        assert baseline is not None
        assert len(engine._receiver_baselines["receiver"]) == 3

    def test_reset_baseline(self, scoring_engine: ScoringEngine) -> None:
        """Test baseline reset functionality."""
        emb = np.random.rand(384).astype(np.float32)
        scoring_engine.compute_scores(
            event_id=uuid4(),
            sender_id="sender",
            receiver_id="receiver",
            sender_embedding=emb,
            receiver_embedding=emb,
        )

        assert scoring_engine._get_baseline("receiver") is not None

        scoring_engine.reset_baseline("receiver")
        assert scoring_engine._get_baseline("receiver") is None

    def test_reset_all_baselines(self, scoring_engine: ScoringEngine) -> None:
        """Test reset all baselines."""
        emb = np.random.rand(384).astype(np.float32)

        for receiver in ["r1", "r2", "r3"]:
            scoring_engine.compute_scores(
                event_id=uuid4(),
                sender_id="sender",
                receiver_id=receiver,
                sender_embedding=emb,
                receiver_embedding=emb,
            )

        scoring_engine.reset_all_baselines()

        for receiver in ["r1", "r2", "r3"]:
            assert scoring_engine._get_baseline(receiver) is None
