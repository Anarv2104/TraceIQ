"""Tests for the TraceIQEvent schema module."""

import json
import time

from traceiq.schema import TraceIQEvent, compute_state_quality


class TestComputeStateQuality:
    """Tests for compute_state_quality function."""

    def test_high_quality_with_both_states(self):
        """Both state_before and state_after present -> high quality."""
        quality = compute_state_quality(
            receiver_output="output",
            receiver_state_before="before",
            receiver_state_after="after",
        )
        assert quality == "high"

    def test_medium_quality_with_input_view(self):
        """Only receiver_input_view present -> medium quality."""
        quality = compute_state_quality(
            receiver_output="output",
            receiver_input_view="retrieved chunks",
        )
        assert quality == "medium"

    def test_low_quality_output_only(self):
        """Only receiver_output present -> low quality."""
        quality = compute_state_quality(
            receiver_output="output",
        )
        assert quality == "low"

    def test_high_beats_medium(self):
        """Full state tracking beats partial context."""
        quality = compute_state_quality(
            receiver_output="output",
            receiver_input_view="chunks",
            receiver_state_before="before",
            receiver_state_after="after",
        )
        assert quality == "high"


class TestTraceIQEvent:
    """Tests for TraceIQEvent model."""

    def test_required_fields(self):
        """Test creation with required fields only."""
        event = TraceIQEvent(
            run_id="run_001",
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="hello",
            receiver_output="hi there",
        )
        assert event.run_id == "run_001"
        assert event.sender_id == "agent_a"
        assert event.receiver_id == "agent_b"
        assert event.sender_content == "hello"
        assert event.receiver_output == "hi there"
        assert event.event_id is not None
        assert event.ts > 0

    def test_default_values(self):
        """Test that defaults are applied correctly."""
        event = TraceIQEvent(
            run_id="run_001",
            sender_id="a",
            receiver_id="b",
            sender_content="x",
            receiver_output="y",
        )
        assert event.task_id is None
        assert event.receiver_input_view is None
        assert event.receiver_state_before is None
        assert event.receiver_state_after is None
        assert event.event_type == "applied"
        assert event.policy_action is None
        assert event.policy_reason is None
        assert event.metadata == {}

    def test_auto_state_quality_low(self):
        """Test auto-computed state quality (low)."""
        event = TraceIQEvent(
            run_id="run_001",
            sender_id="a",
            receiver_id="b",
            sender_content="x",
            receiver_output="y",
        )
        assert event.state_quality == "low"

    def test_auto_state_quality_medium(self):
        """Test auto-computed state quality (medium)."""
        event = TraceIQEvent(
            run_id="run_001",
            sender_id="a",
            receiver_id="b",
            sender_content="x",
            receiver_output="y",
            receiver_input_view="context chunks",
        )
        assert event.state_quality == "medium"

    def test_auto_state_quality_high(self):
        """Test auto-computed state quality (high)."""
        event = TraceIQEvent(
            run_id="run_001",
            sender_id="a",
            receiver_id="b",
            sender_content="x",
            receiver_output="y",
            receiver_state_before="state1",
            receiver_state_after="state2",
        )
        assert event.state_quality == "high"

    def test_explicit_state_quality_respected(self):
        """Test that explicit state_quality is respected.

        When user explicitly sets state_quality, it should be kept
        (the auto-compute only runs if state_quality is "low").
        """
        event = TraceIQEvent(
            run_id="run_001",
            sender_id="a",
            receiver_id="b",
            sender_content="x",
            receiver_output="y",
            state_quality="high",  # Explicit override
        )
        # Explicit "high" is preserved
        assert event.state_quality == "high"

    def test_to_jsonl(self):
        """Test JSONL serialization."""
        event = TraceIQEvent(
            run_id="run_001",
            sender_id="a",
            receiver_id="b",
            sender_content="hello",
            receiver_output="world",
        )
        jsonl = event.to_jsonl()
        assert isinstance(jsonl, str)
        data = json.loads(jsonl)
        assert data["run_id"] == "run_001"
        assert data["sender_id"] == "a"
        assert data["receiver_id"] == "b"

    def test_from_jsonl(self):
        """Test JSONL deserialization."""
        original = TraceIQEvent(
            run_id="run_001",
            sender_id="a",
            receiver_id="b",
            sender_content="hello",
            receiver_output="world",
            event_type="blocked",
            policy_action="quarantine",
        )
        jsonl = original.to_jsonl()
        restored = TraceIQEvent.from_jsonl(jsonl)

        assert restored.run_id == original.run_id
        assert restored.sender_id == original.sender_id
        assert restored.receiver_id == original.receiver_id
        assert restored.event_type == "blocked"
        assert restored.policy_action == "quarantine"

    def test_with_policy(self):
        """Test with_policy method."""
        event = TraceIQEvent(
            run_id="run_001",
            sender_id="a",
            receiver_id="b",
            sender_content="x",
            receiver_output="y",
        )
        updated = event.with_policy("quarantine", "high_risk")

        assert updated.policy_action == "quarantine"
        assert updated.policy_reason == "high_risk"
        assert updated.event_type == "blocked"
        # Original unchanged
        assert event.policy_action is None

    def test_with_policy_allow(self):
        """Test with_policy for allow action."""
        event = TraceIQEvent(
            run_id="run_001",
            sender_id="a",
            receiver_id="b",
            sender_content="x",
            receiver_output="y",
        )
        updated = event.with_policy("allow", "low_risk")

        assert updated.policy_action == "allow"
        assert updated.event_type == "applied"

    def test_with_policy_custom_event_type(self):
        """Test with_policy with custom event_type."""
        event = TraceIQEvent(
            run_id="run_001",
            sender_id="a",
            receiver_id="b",
            sender_content="x",
            receiver_output="y",
        )
        updated = event.with_policy("verify", "needs_review", event_type="attempted")

        assert updated.policy_action == "verify"
        assert updated.event_type == "attempted"

    def test_unique_event_ids(self):
        """Test that event IDs are unique."""
        events = [
            TraceIQEvent(
                run_id="run",
                sender_id="a",
                receiver_id="b",
                sender_content="x",
                receiver_output="y",
            )
            for _ in range(100)
        ]
        ids = [e.event_id for e in events]
        assert len(set(ids)) == 100

    def test_timestamp_is_recent(self):
        """Test that timestamp is current."""
        before = time.time()
        event = TraceIQEvent(
            run_id="run",
            sender_id="a",
            receiver_id="b",
            sender_content="x",
            receiver_output="y",
        )
        after = time.time()

        assert before <= event.ts <= after
