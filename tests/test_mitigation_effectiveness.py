"""Mitigation guard effectiveness tests."""

from traceiq import InfluenceTracker, TrackerConfig


class TestAlertInfrastructure:
    """Test alert detection works."""

    def test_alerts_trackable(self) -> None:
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=10,
            baseline_k=3,
            anomaly_threshold=2.0,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Warmup
        for i in range(10):
            tracker.track_event("normal", "target", f"msg {i}", f"resp {i}")

        alerts = tracker.get_alerts()
        tracker.close()

        assert isinstance(alerts, list)

    def test_result_contains_alert_field(self) -> None:
        config = TrackerConfig(
            storage_backend="memory",
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        result = tracker.track_event("sender", "receiver", "msg", "resp")
        tracker.close()

        assert "alert" in result
        assert isinstance(result["alert"], bool)

    def test_valid_field_in_result(self) -> None:
        config = TrackerConfig(
            storage_backend="memory",
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        result = tracker.track_event("sender", "receiver", "msg", "resp")
        tracker.close()

        assert "valid" in result
        assert isinstance(result["valid"], bool)

    def test_cold_start_no_alerts(self) -> None:
        """During cold start period, alerts should not fire."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=30,
            baseline_k=20,
            anomaly_threshold=1.0,  # Very sensitive
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Only track a few events (below baseline_k)
        results = []
        for i in range(5):
            r = tracker.track_event("sender", "receiver", f"msg {i}", f"resp {i}")
            results.append(r)

        # All should be invalid (cold start) and no alerts
        alerts = tracker.get_alerts(valid_only=True)
        tracker.close()

        # During cold start, valid_only alerts should be empty
        assert len(alerts) == 0


class TestRiskScoring:
    """Test risk score computation."""

    def test_risk_score_in_result(self) -> None:
        config = TrackerConfig(
            storage_backend="memory",
            enable_risk_scoring=True,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        result = tracker.track_event("sender", "receiver", "msg", "resp")
        tracker.close()

        assert "risk_score" in result
        assert "risk_level" in result

    def test_risk_level_categories(self) -> None:
        config = TrackerConfig(
            storage_backend="memory",
            enable_risk_scoring=True,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        result = tracker.track_event("sender", "receiver", "msg", "resp")
        tracker.close()

        valid_levels = {"unknown", "low", "medium", "high", "critical"}
        assert result["risk_level"] in valid_levels
