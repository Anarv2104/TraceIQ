"""Mitigation policy engine for influence events.

This module provides the PolicyEngine class that applies mitigation policies
based on risk scores. Policies determine whether events are allowed, verified,
quarantined, or blocked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

from traceiq.weights import clamp_weight

if TYPE_CHECKING:
    from traceiq.risk import RiskResult
    from traceiq.schema import TraceIQEvent


@dataclass
class PolicyDecision:
    """Result of policy evaluation.

    Attributes:
        action: The policy action to take
        reason: Human-readable reason for the action
        trust_adjustment: Change to apply to sender's trust score
    """

    action: Literal["allow", "verify", "quarantine", "block"]
    reason: str
    trust_adjustment: float = 0.0


@dataclass
class TrustState:
    """Trust state for an agent.

    Attributes:
        score: Current trust score in [0, 1]
        violations: Number of policy violations
        last_action: Most recent policy action taken
    """

    score: float = 1.0
    violations: int = 0
    last_action: Literal["allow", "verify", "quarantine", "block"] | None = None


class PolicyEngine:
    """Policy engine for applying mitigation policies to events.

    The PolicyEngine evaluates risk scores and applies appropriate policies:
    - "allow": Event proceeds normally
    - "verify": Event requires additional verification
    - "quarantine": Event is isolated for review
    - "block": Event is rejected entirely

    It also maintains per-agent trust scores that decay on violations
    and can influence future policy decisions.

    Trust Decay Logic:
    - correct=True → trust + small gain (trust_gain)
    - correct=False → trust - larger decay (trust_decay_rate * 1.5)
    - repeated high risk → decay (trust_decay_rate * 0.5)
    - policy_action in (quarantine, block) → decay (trust_decay_rate)

    Trust scores scale edge weights in propagation graph via scale_edge_weight().

    Attributes:
        enable_trust_decay: Whether to apply trust decay on violations
        trust_decay_rate: Amount to decay trust per violation
        trust_gain: Amount to increase trust for correct outcomes
        trust_scores: Per-agent trust state
    """

    def __init__(
        self,
        enable_trust_decay: bool = True,
        trust_decay_rate: float = 0.1,
        trust_gain: float = 0.02,
        min_trust: float = 0.0,
        max_trust: float = 1.0,
    ) -> None:
        """Initialize the policy engine.

        Args:
            enable_trust_decay: Whether to decay trust on violations (default: True)
            trust_decay_rate: Amount to decay trust per violation (default: 0.1)
            trust_gain: Amount to increase trust for correct outcomes (default: 0.02)
            min_trust: Minimum trust score (default: 0.0)
            max_trust: Maximum trust score (default: 1.0)
        """
        self.enable_trust_decay = enable_trust_decay
        self.trust_decay_rate = trust_decay_rate
        self.trust_gain = trust_gain
        self.min_trust = min_trust
        self.max_trust = max_trust
        self._trust_states: dict[str, TrustState] = {}

    def evaluate_risk(self, risk_result: RiskResult) -> PolicyDecision:
        """Evaluate risk and determine policy action.

        Args:
            risk_result: Risk computation result

        Returns:
            PolicyDecision with action and reason
        """
        if risk_result.risk_level == "unknown":
            # Unknown risk - allow but note uncertainty
            return PolicyDecision(
                action="allow",
                reason="risk_unknown_cold_start",
                trust_adjustment=0.0,
            )

        if risk_result.risk_level == "critical":
            return PolicyDecision(
                action="quarantine",
                reason=f"risk_level=critical (score={risk_result.risk_score:.3f})",
                trust_adjustment=-self.trust_decay_rate * 2,
            )

        if risk_result.risk_level == "high":
            return PolicyDecision(
                action="verify",
                reason=f"risk_level=high (score={risk_result.risk_score:.3f})",
                trust_adjustment=-self.trust_decay_rate,
            )

        if risk_result.risk_level == "medium":
            return PolicyDecision(
                action="allow",
                reason=f"risk_level=medium (score={risk_result.risk_score:.3f})",
                trust_adjustment=0.0,
            )

        # Low risk
        return PolicyDecision(
            action="allow",
            reason=f"risk_level=low (score={risk_result.risk_score:.3f})",
            trust_adjustment=0.0,
        )

    def apply_policy(
        self,
        event: TraceIQEvent,
        risk_result: RiskResult,
    ) -> TraceIQEvent:
        """Apply policy to an event based on risk assessment.

        This method:
        1. Evaluates the risk and determines the appropriate action
        2. Updates the sender's trust score if trust decay is enabled
        3. Returns a new event with policy fields set

        Args:
            event: The event to apply policy to
            risk_result: Risk computation result

        Returns:
            Updated TraceIQEvent with policy_action, policy_reason, and event_type
        """
        # Get policy decision
        decision = self.evaluate_risk(risk_result)

        # Apply trust decay if enabled
        if self.enable_trust_decay and decision.trust_adjustment != 0:
            self._adjust_trust(event.sender_id, decision.trust_adjustment)
            # Track violation
            if decision.action in ("quarantine", "block"):
                state = self._get_trust_state(event.sender_id)
                state.violations += 1
                state.last_action = decision.action

        # Determine event type based on action
        if decision.action in ("quarantine", "block"):
            event_type: Literal["attempted", "applied", "blocked"] = "blocked"
        else:
            event_type = "applied"

        # Return updated event
        return event.model_copy(
            update={
                "policy_action": decision.action,
                "policy_reason": decision.reason,
                "event_type": event_type,
            }
        )

    def get_trust(self, agent_id: str) -> float:
        """Get current trust score for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Trust score in [min_trust, max_trust]
        """
        return self._get_trust_state(agent_id).score

    def set_trust(self, agent_id: str, score: float) -> None:
        """Set trust score for an agent.

        Args:
            agent_id: Agent identifier
            score: New trust score (will be clamped to valid range)
        """
        state = self._get_trust_state(agent_id)
        state.score = clamp_weight(score, self.min_trust, self.max_trust)

    def reset_trust(self, agent_id: str) -> None:
        """Reset trust score for an agent to maximum.

        Args:
            agent_id: Agent identifier
        """
        if agent_id in self._trust_states:
            del self._trust_states[agent_id]

    def get_trust_state(self, agent_id: str) -> TrustState:
        """Get full trust state for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            TrustState with score, violations, and last_action
        """
        return self._get_trust_state(agent_id)

    def get_all_trust_states(self) -> dict[str, TrustState]:
        """Get all trust states.

        Returns:
            Dict mapping agent_id to TrustState
        """
        return dict(self._trust_states)

    def get_low_trust_agents(self, threshold: float = 0.5) -> list[tuple[str, float]]:
        """Get agents with trust below threshold.

        Args:
            threshold: Trust threshold (default: 0.5)

        Returns:
            List of (agent_id, trust_score) tuples, sorted by trust ascending
        """
        low_trust = [
            (agent_id, state.score)
            for agent_id, state in self._trust_states.items()
            if state.score < threshold
        ]
        return sorted(low_trust, key=lambda x: x[1])

    def _get_trust_state(self, agent_id: str) -> TrustState:
        """Get or create trust state for an agent."""
        if agent_id not in self._trust_states:
            self._trust_states[agent_id] = TrustState(score=self.max_trust)
        return self._trust_states[agent_id]

    def _adjust_trust(self, agent_id: str, adjustment: float) -> None:
        """Adjust trust score for an agent.

        Args:
            agent_id: Agent identifier
            adjustment: Amount to add to trust (negative for decay)
        """
        state = self._get_trust_state(agent_id)
        new_score = state.score + adjustment
        state.score = clamp_weight(new_score, self.min_trust, self.max_trust)

    def _decay_trust(self, agent_id: str, decay: float | None = None) -> None:
        """Decay trust score for an agent.

        Args:
            agent_id: Agent identifier
            decay: Decay amount (uses trust_decay_rate if None)
        """
        if decay is None:
            decay = self.trust_decay_rate
        self._adjust_trust(agent_id, -decay)

    def update_trust_from_outcome(
        self,
        agent_id: str,
        risk_level: str,
        policy_action: str,
        correct: bool | None = None,
    ) -> None:
        """Update trust based on outcomes, not just actions.

        This method enables outcome-based trust adjustment when the actual
        correctness of an agent's output is known (e.g., from metadata.correct).

        Trust decay logic:
        - correct=True → trust + small gain
        - correct=False → trust - larger decay
        - repeated high risk → decay
        - policy_action in (quarantine, block) → decay

        Args:
            agent_id: Agent identifier
            risk_level: Risk level from scoring ("low", "medium", "high", "critical")
            policy_action: Policy action taken ("allow", "verify", "quarantine", "block")
            correct: Whether the outcome was correct (from metadata if available)
        """
        if not self.enable_trust_decay:
            return

        current = self.get_trust(agent_id)

        if correct is True:
            # Reward for correct outcomes
            new_trust = min(self.max_trust, current + self.trust_gain)
        elif correct is False:
            # Penalize for incorrect outcomes (larger decay)
            new_trust = max(self.min_trust, current - self.trust_decay_rate * 1.5)
        elif risk_level in ("high", "critical"):
            # Decay for high risk (smaller decay)
            new_trust = max(self.min_trust, current - self.trust_decay_rate * 0.5)
        elif policy_action in ("quarantine", "block"):
            # Decay for blocked actions
            new_trust = max(self.min_trust, current - self.trust_decay_rate)
        else:
            new_trust = current

        self.set_trust(agent_id, new_trust)

    def scale_edge_weight(self, sender_id: str, weight: float) -> float:
        """Scale edge weight by sender's trust score.

        Trust should scale edge weights in propagation graph. Lower trust
        means lower weight in the adjacency matrix.

        Args:
            sender_id: Sender agent identifier
            weight: Original edge weight

        Returns:
            Scaled weight (weight * trust_score)
        """
        return weight * self.get_trust(sender_id)


def should_block_event(
    risk_result: RiskResult,
    trust_score: float = 1.0,
    trust_threshold: float = 0.3,
) -> bool:
    """Quick check if an event should be blocked.

    This is a convenience function for simple blocking decisions
    without using the full PolicyEngine.

    Args:
        risk_result: Risk computation result
        trust_score: Sender's trust score (default: 1.0)
        trust_threshold: Minimum trust to allow (default: 0.3)

    Returns:
        True if event should be blocked

    Examples:
        >>> from traceiq.risk import RiskResult
        >>> risk = RiskResult(risk_score=0.9, risk_level="critical", components={})
        >>> should_block_event(risk)
        True

        >>> risk = RiskResult(risk_score=0.1, risk_level="low", components={})
        >>> should_block_event(risk)
        False
    """
    # Block if trust is too low
    if trust_score < trust_threshold:
        return True

    # Block critical risk
    if risk_result.risk_level == "critical":
        return True

    return False
