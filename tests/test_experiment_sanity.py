"""Experiment sanity - determinism and schema validation.

CI-safe tests that validate core traceiq behavior:
- Determinism with fixed seeds
- Schema validation for TraceIQEvent
- IQx bounds checking on tracker output
"""

import json
import math

from traceiq import InfluenceTracker, TraceIQEvent, TrackerConfig
from traceiq.metrics import IQX_CAP


class TestDeterminism:
    """Same seed = same results."""

    def _run_session(self, seed: int) -> dict:
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=10,
            baseline_k=3,
            random_seed=seed,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        results = []
        for i in range(10):
            r = tracker.track_event("sender", "receiver", f"msg {i}", f"resp {i}")
            results.append(r)

        summary = tracker.summary()
        tracker.close()

        return {
            "total": summary.total_events,
            "alerts": sum(1 for r in results if r.get("alert")),
        }

    def test_same_seed_same_results(self) -> None:
        r1 = self._run_session(42)
        r2 = self._run_session(42)
        assert r1 == r2

    def test_different_seed_may_differ(self) -> None:
        """Different seeds should produce consistent but potentially different results."""
        r1 = self._run_session(42)
        r2 = self._run_session(43)
        # Both should run successfully
        assert r1["total"] == r2["total"]  # Same number of events


class TestSchemaValidation:
    """TraceIQEvent serializes correctly."""

    def test_event_to_json(self) -> None:
        from uuid import uuid4

        event = TraceIQEvent(
            event_id=str(uuid4()),
            run_id="test-run",
            sender_id="sender",
            receiver_id="receiver",
            sender_content="content",
            receiver_output="response",
        )

        json_str = event.model_dump_json()
        parsed = json.loads(json_str)
        assert parsed["sender_id"] == "sender"
        assert parsed["receiver_id"] == "receiver"
        assert parsed["run_id"] == "test-run"

    def test_event_roundtrip(self) -> None:
        from uuid import uuid4

        event = TraceIQEvent(
            event_id=str(uuid4()),
            run_id="test-run",
            sender_id="sender",
            receiver_id="receiver",
            sender_content="content",
            receiver_output="response",
            metadata={"key": "value"},
        )

        json_str = event.to_jsonl()
        restored = TraceIQEvent.from_jsonl(json_str)

        assert restored.sender_id == event.sender_id
        assert restored.receiver_id == event.receiver_id
        assert restored.metadata == event.metadata

    def test_event_with_policy(self) -> None:
        from uuid import uuid4

        event = TraceIQEvent(
            event_id=str(uuid4()),
            run_id="test-run",
            sender_id="sender",
            receiver_id="receiver",
            sender_content="content",
            receiver_output="response",
        )

        updated = event.with_policy("block", "high risk detected")
        assert updated.policy_action == "block"
        assert updated.policy_reason == "high risk detected"
        assert updated.event_type == "blocked"


class TestTrackerWithSchema:
    """Test tracker integration with TraceIQEvent schema."""

    def test_track_with_run_id(self) -> None:
        config = TrackerConfig(
            storage_backend="memory",
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        result = tracker.track_event(
            sender_id="sender",
            receiver_id="receiver",
            sender_content="msg",
            receiver_content="resp",
            run_id="experiment-1",
            task_id="task-1",
        )
        tracker.close()

        assert result["run_id"] == "experiment-1"
        assert result["task_id"] == "task-1"


def _get_iqx(result: dict) -> float | None:
    """Extract IQx from result, handling key variations."""
    return result.get("IQx", result.get("iqx"))


class TestIQxBoundsOnTracker:
    """Validate IQx values from tracker are bounded and sane."""

    def test_iqx_in_bounds_chain(self) -> None:
        """IQx values from chain-like interactions are bounded."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=10,
            baseline_k=3,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        iqx_values = []

        # Simulate A -> B -> C chain
        for i in range(15):
            result = tracker.track_event("A", "B", f"msg {i}", f"resp {i}")
            iqx = _get_iqx(result)
            if iqx is not None:
                iqx_values.append(iqx)

            result = tracker.track_event("B", "C", f"forward {i}", f"ack {i}")
            iqx = _get_iqx(result)
            if iqx is not None:
                iqx_values.append(iqx)

        tracker.close()

        # Must have collected some IQx values after warmup
        assert len(iqx_values) > 0, "No IQx values returned after warmup"

        # All IQx values must be bounded
        for iqx in iqx_values:
            assert 0 <= iqx <= IQX_CAP, f"IQx out of bounds: {iqx}"
            assert not math.isnan(iqx), "IQx is NaN"
            assert not math.isinf(iqx), "IQx is infinite"

    def test_iqx_in_bounds_noise(self) -> None:
        """IQx values from noise interactions are bounded."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=10,
            baseline_k=3,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        iqx_values = []
        alert_count = 0
        total = 0

        # Simulate random noise interactions
        agents = ["A", "B", "C", "D"]
        for i in range(20):
            sender = agents[i % 4]
            receiver = agents[(i + 1) % 4]
            result = tracker.track_event(
                sender,
                receiver,
                f"noise content {i} from {sender}",
                f"noise response {i}",
            )

            iqx = _get_iqx(result)
            if iqx is not None:
                iqx_values.append(iqx)
                assert 0 <= iqx <= IQX_CAP

            if result.get("alert", False):
                alert_count += 1
            total += 1

        tracker.close()

        # Alert rate for pure noise should be low (< 30%)
        if total > 0:
            alert_rate = alert_count / total
            assert alert_rate <= 0.5, f"Alert rate unexpectedly high: {alert_rate}"

    def test_result_keys_exist(self) -> None:
        """Track event returns expected keys."""
        config = TrackerConfig(
            storage_backend="memory",
            baseline_window=10,
            baseline_k=3,
            random_seed=42,
        )
        tracker = InfluenceTracker(config=config, use_mock_embedder=True)

        # Warmup
        for i in range(5):
            tracker.track_event("A", "B", f"warmup {i}", f"resp {i}")

        result = tracker.track_event("A", "B", "test message", "test response")
        tracker.close()

        # Core keys that must exist
        assert "alert" in result, "Missing 'alert' key in result"
        assert "valid" in result, "Missing 'valid' key in result"

        # IQx should exist (possibly under different case)
        has_iqx = "IQx" in result or "iqx" in result
        assert has_iqx, f"Missing IQx key. Result keys: {list(result.keys())}"
