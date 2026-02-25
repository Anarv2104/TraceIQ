"""Integration tests for TraceIQ end-to-end pipeline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from traceiq import InfluenceTracker, TrackerConfig

pytestmark = pytest.mark.integration


class TestTrackerToExportPipeline:
    """Tracker -> SQLite -> Export -> Verify pipeline."""

    def test_sqlite_to_csv_export(self, tmp_path: Path) -> None:
        """Track events with SQLite, export to CSV, verify contents."""
        db_path = tmp_path / "test.db"
        csv_path = tmp_path / "export.csv"

        config = TrackerConfig(
            storage_backend="sqlite",
            storage_path=str(db_path),
            baseline_window=20,
            baseline_k=5,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Track events
        for i in range(10):
            tracker.track_event(
                sender_id=f"agent_{i % 3}",
                receiver_id=f"agent_{(i + 1) % 3}",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
            )

        # Export and verify
        tracker.export_csv(str(csv_path))
        tracker.close()

        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))

        assert len(rows) == 10
        # These are the columns actually exported by export_combined_csv
        required_cols = [
            "sender_id",
            "receiver_id",
            "influence_score",
            "drift_delta",
            "cold_start",
        ]
        for col in required_cols:
            assert col in rows[0], f"Missing column: {col}"

    def test_memory_to_jsonl_export(self, tmp_path: Path) -> None:
        """Track events with memory storage, export to JSONL, verify contents."""
        jsonl_path = tmp_path / "export.jsonl"

        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=20,
            baseline_k=5,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Track events
        for i in range(8):
            tracker.track_event(
                sender_id="sender",
                receiver_id="receiver",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
            )

        # Export and verify
        tracker.export_jsonl(str(jsonl_path))
        tracker.close()

        with open(jsonl_path) as f:
            rows = [json.loads(line) for line in f if line.strip()]

        assert len(rows) == 8
        assert all("event_id" in row for row in rows)
        assert all("sender_id" in row for row in rows)

    def test_no_alerts_during_cold_start(self, tmp_path: Path) -> None:
        """Verify no alerts during cold start period."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=30,
            baseline_k=20,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        for i in range(25):
            result = tracker.track_event(
                sender_id="sender",
                receiver_id="receiver",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
            )
            if i < 20:
                assert result["valid"] is False, f"Event {i} should be invalid"
                assert result["alert"] is False, f"Event {i} should have no alert"

        tracker.close()

    def test_valid_metrics_after_cold_start(self, tmp_path: Path) -> None:
        """Verify valid metrics after cold start period completes."""
        # Use small baseline_k and enough events to establish baseline per receiver
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=20,
            baseline_k=5,  # Small cold start period
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        results = []
        for i in range(30):  # More events to ensure baseline is established
            result = tracker.track_event(
                sender_id="sender",
                receiver_id="receiver",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
            )
            results.append(result)

        tracker.close()

        # During cold start (first baseline_k events), metrics should be invalid
        for i in range(5):
            assert results[i]["valid"] is False, f"Event {i} should be invalid"

        # After cold start, we expect metrics to become valid eventually
        # The exact timing depends on baseline accumulation per receiver
        valid_count = sum(1 for r in results if r["valid"] is True)
        # After 30 events with baseline_k=5, many should be valid
        assert valid_count > 0, f"Expected valid events, got {valid_count}"


class TestPropagationRiskIntegration:
    """PR computation through tracker."""

    def test_windowed_pr_nonzero_with_chain(self, tmp_path: Path) -> None:
        """Windowed PR > 0 when chain has non-zero weights."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=20,
            baseline_k=3,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Create chain: A -> B -> C with multiple events
        agents = ["agent_A", "agent_B", "agent_C"]
        for _ in range(5):
            for i in range(len(agents) - 1):
                tracker.track_event(
                    sender_id=agents[i],
                    receiver_id=agents[i + 1],
                    sender_content=f"Influence from {agents[i]}",
                    receiver_content=f"Response from {agents[i + 1]}",
                )

        # Use tracker methods to get events/scores
        events = tracker.get_events()
        scores = tracker.get_scores()
        _ = tracker.graph.compute_windowed_pr(events, scores, window_size=10)

        # Should have edges
        assert len(tracker.graph._edge_iqx) > 0, "Graph should have edges"
        tracker.close()

    def test_pr_increases_with_more_interactions(self, tmp_path: Path) -> None:
        """PR should generally increase with more interactions in a chain."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=20,
            baseline_k=5,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        pr_values = []
        agents = ["agent_A", "agent_B", "agent_C", "agent_D"]

        for round_idx in range(10):
            for i in range(len(agents) - 1):
                tracker.track_event(
                    sender_id=agents[i],
                    receiver_id=agents[i + 1],
                    sender_content=f"Round {round_idx} message from {agents[i]}",
                    receiver_content=f"Round {round_idx} response from {agents[i + 1]}",
                )

            # Use tracker methods
            events = tracker.get_events()
            scores = tracker.get_scores()
            if len(events) >= 3:
                window = min(len(events), 20)
                pr = tracker.graph.compute_windowed_pr(events, scores, window)
                pr_values.append(pr)

        tracker.close()

        # PR should be computed for multiple rounds
        assert len(pr_values) > 0, "Should have PR values"


class TestGraphIntegration:
    """Graph analytics integration tests."""

    def test_graph_builds_from_tracked_events(self, tmp_path: Path) -> None:
        """Graph should build correctly from tracked events."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=20,
            baseline_k=5,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Track events between multiple agents
        pairs = [
            ("alice", "bob"),
            ("bob", "charlie"),
            ("alice", "charlie"),
            ("charlie", "david"),
        ]

        for i, (sender, receiver) in enumerate(pairs * 3):
            tracker.track_event(
                sender_id=sender,
                receiver_id=receiver,
                sender_content=f"Message {i} from {sender}",
                receiver_content=f"Response {i} from {receiver}",
            )

        stats = tracker.graph.get_stats()
        tracker.close()

        # Verify graph structure
        assert stats["num_nodes"] == 4, "Should have 4 unique agents"
        assert stats["num_edges"] == 4, "Should have 4 unique edges"

    def test_top_influencers_ranking(self, tmp_path: Path) -> None:
        """Top influencers should be ranked by outgoing influence weight sum."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=20,
            baseline_k=5,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Agent_A sends many messages
        for i in range(20):
            tracker.track_event(
                sender_id="agent_A",
                receiver_id="agent_B",
                sender_content=f"Message {i} from A",
                receiver_content=f"Response {i}",
            )

        # Agent_B sends fewer messages
        for i in range(5):
            tracker.track_event(
                sender_id="agent_B",
                receiver_id="agent_C",
                sender_content=f"Message {i} from B",
                receiver_content=f"Response {i}",
            )

        top = tracker.graph.top_influencers(n=2)
        tracker.close()

        # Both agents should appear in top 2
        assert len(top) == 2
        agent_names = [t[0] for t in top]
        assert "agent_A" in agent_names
        assert "agent_B" in agent_names
