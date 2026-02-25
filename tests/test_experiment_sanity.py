"""Experiment sanity - determinism and schema validation."""

import json

from traceiq import InfluenceTracker, TraceIQEvent, TrackerConfig


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
