"""Performance sanity tests (optional, run with pytest -m perf)."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from traceiq import InfluenceTracker, TrackerConfig


@pytest.mark.perf
class TestPerformanceSanity:
    """Performance sanity with loose thresholds."""

    def test_10k_events_under_60s(self, tmp_path: Path) -> None:
        """10k events complete in under 60 seconds."""
        config = TrackerConfig(
            storage_backend="sqlite",
            storage_path=str(tmp_path / "perf.db"),
            baseline_window=10,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        start = time.time()
        for i in range(10_000):
            tracker.track_event(
                sender_id=f"agent_{i % 10}",
                receiver_id=f"agent_{(i + 1) % 10}",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
            )
        elapsed = time.time() - start
        tracker.close()

        assert elapsed < 60, f"10k events took {elapsed:.2f}s"
        print(f"\n[Perf] 10k events: {elapsed:.2f}s ({10000/elapsed:.0f} events/sec)")

    def test_1k_events_memory_storage(self, tmp_path: Path) -> None:
        """1k events with memory storage complete in under 10 seconds."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=10,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        start = time.time()
        for i in range(1_000):
            tracker.track_event(
                sender_id=f"agent_{i % 5}",
                receiver_id=f"agent_{(i + 1) % 5}",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
            )
        elapsed = time.time() - start
        tracker.close()

        assert elapsed < 10, f"1k events took {elapsed:.2f}s"
        rate = 1000 / elapsed
        print(f"\n[Perf] 1k events (memory): {elapsed:.2f}s ({rate:.0f} events/sec)")

    def test_graph_operations_scale(self, tmp_path: Path) -> None:
        """Graph operations should scale reasonably with event count."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=10,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Create many events across multiple agents
        num_agents = 20
        num_events = 500

        for i in range(num_events):
            tracker.track_event(
                sender_id=f"agent_{i % num_agents}",
                receiver_id=f"agent_{(i + 1) % num_agents}",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
            )

        # Time graph operations
        start = time.time()
        _ = tracker.graph.top_influencers(n=10)
        _ = tracker.graph.top_susceptible(n=10)
        _ = tracker.graph.build_adjacency_matrix()
        _ = tracker.graph.compute_spectral_radius()
        graph_time = time.time() - start

        tracker.close()

        # Graph operations should be fast
        assert graph_time < 5, f"Graph operations took {graph_time:.2f}s"
        print(f"\n[Perf] Graph ops: {graph_time:.3f}s")

    def test_windowed_pr_computation(self, tmp_path: Path) -> None:
        """Windowed PR computation should be fast."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=10,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Create events
        for i in range(200):
            tracker.track_event(
                sender_id=f"agent_{i % 10}",
                receiver_id=f"agent_{(i + 1) % 10}",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
            )

        events = tracker.get_events()
        scores = tracker.get_scores()

        # Time windowed PR computation
        start = time.time()
        for ws in [10, 20, 50, 100]:
            _ = tracker.graph.compute_windowed_pr(events, scores, window_size=ws)
        pr_time = time.time() - start

        tracker.close()

        assert pr_time < 5, f"Windowed PR took {pr_time:.2f}s"
        print(f"\n[Perf] Windowed PR (4 sizes): {pr_time:.3f}s")


@pytest.mark.perf
class TestExportPerformance:
    """Export operation performance tests."""

    def test_csv_export_1k_events(self, tmp_path: Path) -> None:
        """CSV export of 1k events should be fast."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=10,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        for i in range(1_000):
            tracker.track_event(
                sender_id=f"agent_{i % 10}",
                receiver_id=f"agent_{(i + 1) % 10}",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
            )

        csv_path = tmp_path / "export.csv"

        start = time.time()
        tracker.export_csv(str(csv_path))
        export_time = time.time() - start

        tracker.close()

        assert export_time < 5, f"CSV export took {export_time:.2f}s"
        assert csv_path.exists()
        print(f"\n[Perf] CSV export (1k events): {export_time:.3f}s")
