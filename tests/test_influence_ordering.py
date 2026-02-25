"""Influence ordering with synthetic ground truth."""

from traceiq import InfluenceTracker, TrackerConfig


class TestInfluenceRanking:
    """Test relative influence rankings."""

    def test_high_volume_sender_has_more_total_iqx(self) -> None:
        """High volume sender should have higher total IQx (sum, not average)."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=10,
            baseline_k=3,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # High volume sender - 20 interactions
        for i in range(20):
            tracker.track_event("high", "target", f"msg {i}", f"resp {i}")

        # Low volume sender - 3 interactions
        for i in range(3):
            tracker.track_event("low", "target", f"msg {i}", f"resp {i}")

        # Use top_iqx_influencers which sums IQx (volume matters)
        top = tracker.graph.top_iqx_influencers(n=2)
        tracker.close()

        # High volume sender should have more total IQx
        assert len(top) >= 2
        assert top[0][0] == "high"

    def test_chain_agents_detected(self) -> None:
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=10,
            baseline_k=3,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # A -> B -> C chain
        for i in range(10):
            tracker.track_event("A", "B", f"A to B {i}", f"B resp {i}")
            tracker.track_event("B", "C", f"B to C {i}", f"C resp {i}")

        stats = tracker.graph.get_stats()
        tracker.close()

        # Graph should have A, B, C nodes
        assert stats["num_nodes"] >= 3

    def test_isolated_agent_not_in_top(self) -> None:
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=10,
            baseline_k=3,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Active sender
        for i in range(15):
            tracker.track_event("active", "target", f"msg {i}", f"resp {i}")

        # Isolated agent (only receives, never sends)
        for i in range(5):
            tracker.track_event("other", "isolated", f"msg {i}", f"resp {i}")

        top = tracker.graph.top_influencers(n=3)
        tracker.close()

        # Isolated should not be top influencer (they never send)
        top_ids = [agent_id for agent_id, _ in top]
        assert "isolated" not in top_ids or top_ids.index("isolated") > 0
