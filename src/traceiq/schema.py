"""Extended event schema for production-grade TraceIQ.

This module provides the TraceIQEvent model which extends the basic InteractionEvent
with additional fields for run tracking, state quality assessment, and mitigation policies.
"""

from __future__ import annotations

import json
import time
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


class TraceIQEvent(BaseModel):
    """Extended event model for production influence tracking.

    This model captures full context for each interaction including:
    - Run and task tracking for experiment organization
    - State quality indicators for metric confidence
    - Mitigation tracking for policy enforcement

    Attributes:
        event_id: Unique identifier for this event
        ts: Unix timestamp when event occurred
        run_id: Identifier for the experiment/session run
        task_id: Optional task identifier within a run
        sender_id: ID of the sending agent
        receiver_id: ID of the receiving agent
        sender_content: Content sent by the sender
        receiver_output: Receiver's response/output
        receiver_input_view: What the receiver actually saw (for RAG/tools)
        receiver_state_before: Receiver's state before processing
        receiver_state_after: Receiver's state after processing
        state_quality: Quality of state tracking (affects metric confidence)
        event_type: Whether event was attempted, applied, or blocked
        policy_action: Action taken by policy engine
        policy_reason: Reason for policy action
        metadata: Additional arbitrary metadata
    """

    # Required fields
    event_id: str = Field(default_factory=lambda: str(uuid4()))
    ts: float = Field(default_factory=time.time)
    run_id: str
    task_id: str | None = None
    sender_id: str
    receiver_id: str
    sender_content: str
    receiver_output: str

    # Optional state tracking
    receiver_input_view: str | None = Field(
        default=None,
        description="What the receiver actually saw (e.g., retrieved chunks for RAG)",
    )
    receiver_state_before: str | None = Field(
        default=None,
        description="Receiver's state/memory before processing",
    )
    receiver_state_after: str | None = Field(
        default=None,
        description="Receiver's state/memory after processing",
    )
    state_quality: Literal["low", "medium", "high"] = Field(
        default="low",
        description="Quality of state tracking (auto-computed if not set)",
    )

    # Mitigation tracking
    event_type: Literal["attempted", "applied", "blocked"] = Field(
        default="applied",
        description="Whether event was attempted, applied to state, or blocked",
    )
    policy_action: Literal["allow", "verify", "quarantine", "block"] | None = Field(
        default=None,
        description="Action taken by policy engine",
    )
    policy_reason: str | None = Field(
        default=None,
        description="Reason for the policy action",
    )

    # Metadata
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": False}  # Allow mutation for policy updates

    def model_post_init(self, __context: Any) -> None:
        """Auto-compute state_quality if not explicitly set."""
        # Only auto-compute if state_quality wasn't explicitly provided
        # We check if it's still the default "low" and if state fields exist
        if self.state_quality == "low":
            computed = compute_state_quality(
                receiver_output=self.receiver_output,
                receiver_input_view=self.receiver_input_view,
                receiver_state_before=self.receiver_state_before,
                receiver_state_after=self.receiver_state_after,
            )
            # Use object.__setattr__ to bypass frozen validation if enabled
            object.__setattr__(self, "state_quality", computed)

    def to_jsonl(self) -> str:
        """Serialize event to JSONL format.

        Returns:
            JSON string representation of the event
        """
        return self.model_dump_json()

    @classmethod
    def from_jsonl(cls, line: str) -> TraceIQEvent:
        """Deserialize event from JSONL format.

        Args:
            line: JSON string representation of an event

        Returns:
            TraceIQEvent instance
        """
        data = json.loads(line)
        return cls(**data)

    def with_policy(
        self,
        action: Literal["allow", "verify", "quarantine", "block"],
        reason: str,
        event_type: Literal["attempted", "applied", "blocked"] | None = None,
    ) -> TraceIQEvent:
        """Create a copy with policy information applied.

        Args:
            action: The policy action to apply
            reason: Reason for the policy action
            event_type: Override event type (auto-set based on action if None)

        Returns:
            New TraceIQEvent with policy fields set
        """
        if event_type is None:
            # Auto-determine event_type from action
            if action in ("quarantine", "block"):
                event_type = "blocked"
            else:
                event_type = "applied"

        return self.model_copy(
            update={
                "policy_action": action,
                "policy_reason": reason,
                "event_type": event_type,
            }
        )


def compute_state_quality(
    receiver_output: str,
    receiver_input_view: str | None = None,
    receiver_state_before: str | None = None,
    receiver_state_after: str | None = None,
) -> Literal["low", "medium", "high"]:
    """Compute state quality based on available state information.

    State quality determines confidence in metrics:
    - "high": Both state_before and state_after exist (full state tracking)
    - "medium": receiver_input_view exists (partial context)
    - "low": Only receiver_output exists (minimal tracking)

    Args:
        receiver_output: The receiver's output (always required)
        receiver_input_view: What the receiver saw (RAG chunks, tool outputs)
        receiver_state_before: Receiver's state before processing
        receiver_state_after: Receiver's state after processing

    Returns:
        State quality level: "low", "medium", or "high"
    """
    if receiver_state_before is not None and receiver_state_after is not None:
        return "high"
    elif receiver_input_view is not None:
        return "medium"
    else:
        return "low"
