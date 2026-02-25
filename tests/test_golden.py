"""Golden tests with fixed expected outputs for regression detection."""

from __future__ import annotations

import numpy as np
import pytest

from traceiq import InfluenceTracker, TrackerConfig
from traceiq.metrics import compute_drift_l2, compute_IQx, compute_propagation_risk


class TestGoldenMetrics:
    """Golden tests for metric computations."""

    def test_drift_l2_orthogonal_vectors(self) -> None:
        """Orthogonal unit vectors -> drift = sqrt(2)."""
        emb1 = np.zeros(384, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(384, dtype=np.float32)
        emb2[1] = 1.0

        drift = compute_drift_l2(emb1, emb2)
        assert drift == pytest.approx(np.sqrt(2), rel=1e-5)

    def test_drift_l2_identical_is_zero(self) -> None:
        """Identical embeddings -> drift = 0."""
        emb = np.ones(384, dtype=np.float32) / np.sqrt(384)
        drift = compute_drift_l2(emb, emb)
        assert drift == pytest.approx(0.0, abs=1e-9)

    def test_drift_l2_opposite_vectors(self) -> None:
        """Opposite unit vectors -> drift = 2."""
        emb1 = np.zeros(384, dtype=np.float32)
        emb1[0] = 1.0
        emb2 = np.zeros(384, dtype=np.float32)
        emb2[0] = -1.0

        drift = compute_drift_l2(emb1, emb2)
        assert drift == pytest.approx(2.0, rel=1e-5)

    def test_iqx_is_capped_at_25(self) -> None:
        """IQx should be capped at 25.0."""
        iqx = compute_IQx(drift=100.0, baseline_median=0.01)
        assert iqx == 25.0

    def test_iqx_with_zero_drift(self) -> None:
        """IQx with zero drift should be 0."""
        iqx = compute_IQx(drift=0.0, baseline_median=1.0)
        assert iqx == pytest.approx(0.0, abs=1e-9)

    def test_iqx_normalization(self) -> None:
        """IQx should normalize drift by baseline."""
        # Drift = 2 * baseline should give IQx ~= 2
        iqx = compute_IQx(drift=2.0, baseline_median=1.0)
        assert iqx == pytest.approx(2.0, rel=1e-3)

    def test_iqx_baseline_floor(self) -> None:
        """IQx should use baseline floor for very small baselines."""
        # With baseline_floor=0.05, small baselines are floored
        iqx1 = compute_IQx(drift=0.1, baseline_median=0.001, baseline_floor=0.05)
        iqx2 = compute_IQx(drift=0.1, baseline_median=0.05, baseline_floor=0.05)
        # Both should give similar results due to floor
        assert iqx1 == pytest.approx(iqx2, rel=0.1)

    def test_pr_identity_matrix(self) -> None:
        """Identity matrix spectral radius = 1.0."""
        matrix = np.eye(3, dtype=np.float64)
        pr = compute_propagation_risk(matrix)
        assert pr == pytest.approx(1.0, rel=1e-5)

    def test_pr_zero_matrix(self) -> None:
        """Zero matrix spectral radius = 0.0."""
        matrix = np.zeros((3, 3), dtype=np.float64)
        pr = compute_propagation_risk(matrix)
        assert pr == pytest.approx(0.0, abs=1e-9)

    def test_pr_empty_matrix(self) -> None:
        """Empty matrix spectral radius = 0.0."""
        matrix = np.array([], dtype=np.float64)
        pr = compute_propagation_risk(matrix)
        assert pr == 0.0

    def test_pr_single_element(self) -> None:
        """1x1 matrix spectral radius = |element|."""
        matrix = np.array([[0.5]], dtype=np.float64)
        pr = compute_propagation_risk(matrix)
        assert pr == pytest.approx(0.5, rel=1e-5)

    def test_pr_chain_topology(self) -> None:
        """Chain A->B->C has spectral radius = 0 (nilpotent)."""
        # A->B->C with unit weights forms a nilpotent matrix
        matrix = np.array(
            [
                [0, 1, 0],  # A -> B
                [0, 0, 1],  # B -> C
                [0, 0, 0],  # C -> nothing
            ],
            dtype=np.float64,
        )
        pr = compute_propagation_risk(matrix)
        # Nilpotent matrix has all eigenvalues = 0
        assert pr == pytest.approx(0.0, abs=1e-5)

    def test_pr_cycle_topology(self) -> None:
        """Cycle A->B->C->A with unit weights has spectral radius = 1."""
        matrix = np.array(
            [
                [0, 1, 0],  # A -> B
                [0, 0, 1],  # B -> C
                [1, 0, 0],  # C -> A (forms cycle)
            ],
            dtype=np.float64,
        )
        pr = compute_propagation_risk(matrix)
        # Permutation matrix has eigenvalues on unit circle
        assert pr == pytest.approx(1.0, rel=1e-5)


class TestGoldenTrackerSequence:
    """Golden tests for tracker determinism."""

    def test_deterministic_outputs(self) -> None:
        """Same seed produces identical outputs."""

        def run_tracker():
            config = TrackerConfig(
                storage_backend="memory",
                baseline_window=20,
                baseline_k=5,
                random_seed=42,
            )
            tracker = InfluenceTracker(config=config, use_mock_embedder=True)
            results = []
            for i in range(10):
                r = tracker.track_event(
                    sender_id="sender",
                    receiver_id="receiver",
                    sender_content=f"Msg {i}",
                    receiver_content=f"Resp {i}",
                )
                results.append(
                    (r["influence_score"], r["drift_delta"], r["cold_start"])
                )
            tracker.close()
            return results

        r1, r2 = run_tracker(), run_tracker()
        for (a1, b1, c1), (a2, b2, c2) in zip(r1, r2, strict=True):
            assert a1 == pytest.approx(a2)
            assert b1 == pytest.approx(b2)
            assert c1 == c2

    def test_cold_start_valid_false_no_alerts(self) -> None:
        """During cold start: valid=False, alert=False."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=30,
            baseline_k=20,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        for i in range(25):
            r = tracker.track_event(
                sender_id="sender",
                receiver_id="receiver",
                sender_content=f"Msg {i}",
                receiver_content=f"Resp {i}",
            )
            if i < 20:
                assert r["valid"] is False
                assert r["alert"] is False

        tracker.close()

    def test_event_ids_are_unique(self) -> None:
        """All event IDs should be unique."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=20,
            baseline_k=5,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        event_ids = []
        for i in range(50):
            r = tracker.track_event(
                sender_id=f"sender_{i % 3}",
                receiver_id=f"receiver_{i % 3}",
                sender_content=f"Msg {i}",
                receiver_content=f"Resp {i}",
            )
            event_ids.append(r.get("event_id"))

        tracker.close()

        # All event IDs should be unique
        assert len(event_ids) == len(set(event_ids))

    def test_scores_monotonic_event_count(self) -> None:
        """Number of stored scores should equal number of events."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=20,
            baseline_k=5,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        for i in range(30):
            tracker.track_event(
                sender_id="sender",
                receiver_id="receiver",
                sender_content=f"Msg {i}",
                receiver_content=f"Resp {i}",
            )

        # Use tracker methods (not storage directly)
        events = tracker.get_events()
        scores = tracker.get_scores()

        tracker.close()

        assert len(events) == 30
        assert len(scores) == 30


class TestGoldenZScore:
    """Golden tests for Z-score computation."""

    def test_z_score_at_mean_is_zero(self) -> None:
        """Z-score at the mean should be ~0."""
        from traceiq.metrics import compute_z_score_robust

        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        median = 3.0  # Median of [1,2,3,4,5]

        z = compute_z_score_robust(median, values)
        assert z == pytest.approx(0.0, abs=0.1)

    def test_z_score_above_mean_positive(self) -> None:
        """Z-score above mean should be positive."""
        from traceiq.metrics import compute_z_score_robust

        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        z = compute_z_score_robust(10.0, values)  # Well above median
        assert z > 0

    def test_z_score_below_mean_negative(self) -> None:
        """Z-score below mean should be negative."""
        from traceiq.metrics import compute_z_score_robust

        values = [1.0, 2.0, 3.0, 4.0, 5.0]

        z = compute_z_score_robust(-10.0, values)  # Well below median
        assert z < 0

    def test_z_score_requires_min_samples(self) -> None:
        """Z-score with insufficient samples should be 0."""
        from traceiq.metrics import compute_z_score_robust

        values = [1.0]  # Only 1 sample

        z = compute_z_score_robust(5.0, values)
        assert z == 0.0


class TestGoldenEdgeCases:
    """Golden tests for edge cases."""

    def test_same_sender_receiver(self) -> None:
        """Events where sender == receiver should still be tracked."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=20,
            baseline_k=5,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Self-influence (agent talking to itself)
        r = tracker.track_event(
            sender_id="agent",
            receiver_id="agent",
            sender_content="Self message",
            receiver_content="Self response",
        )

        tracker.close()

        assert r is not None
        assert "event_id" in r

    def test_empty_content_strings(self) -> None:
        """Events with empty content should still be tracked."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=20,
            baseline_k=5,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        r = tracker.track_event(
            sender_id="sender",
            receiver_id="receiver",
            sender_content="",
            receiver_content="",
        )

        tracker.close()

        assert r is not None

    def test_unicode_content(self) -> None:
        """Events with unicode content should be handled."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=20,
            baseline_k=5,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Use valid unicode characters (Chinese + Greek letters)
        r = tracker.track_event(
            sender_id="sender",
            receiver_id="receiver",
            sender_content="Hello \u4e16\u754c",  # "Hello World" in Chinese
            receiver_content="Response \u03b1\u03b2\u03b3",  # Greek letters
        )

        tracker.close()

        assert r is not None
