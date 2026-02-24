"""Tests for the policy module."""

import pytest

from traceiq.policy import PolicyDecision, PolicyEngine, TrustState, should_block_event
from traceiq.risk import RiskResult
from traceiq.schema import TraceIQEvent


class TestPolicyDecision:
    """Tests for PolicyDecision dataclass."""

    def test_creation(self):
        """Test creating a policy decision."""
        decision = PolicyDecision(
            action="quarantine",
            reason="high risk detected",
            trust_adjustment=-0.1,
        )
        assert decision.action == "quarantine"
        assert decision.reason == "high risk detected"
        assert decision.trust_adjustment == -0.1


class TestTrustState:
    """Tests for TrustState dataclass."""

    def test_default_values(self):
        """Test default trust state values."""
        state = TrustState()
        assert state.score == 1.0
        assert state.violations == 0
        assert state.last_action is None

    def test_custom_values(self):
        """Test custom trust state values."""
        state = TrustState(score=0.5, violations=2, last_action="quarantine")
        assert state.score == 0.5
        assert state.violations == 2
        assert state.last_action == "quarantine"


class TestPolicyEngine:
    """Tests for PolicyEngine class."""

    def test_initialization(self):
        """Test policy engine initialization."""
        engine = PolicyEngine()
        assert engine.enable_trust_decay is True
        assert engine.trust_decay_rate == 0.1

    def test_initialization_custom(self):
        """Test custom initialization."""
        engine = PolicyEngine(
            enable_trust_decay=False,
            trust_decay_rate=0.2,
            min_trust=0.1,
            max_trust=0.9,
        )
        assert engine.enable_trust_decay is False
        assert engine.trust_decay_rate == 0.2
        assert engine.min_trust == 0.1
        assert engine.max_trust == 0.9

    def test_evaluate_risk_unknown(self):
        """Unknown risk level returns allow action."""
        engine = PolicyEngine()
        risk = RiskResult(risk_score=None, risk_level="unknown", components={})
        decision = engine.evaluate_risk(risk)
        assert decision.action == "allow"
        assert "cold_start" in decision.reason

    def test_evaluate_risk_critical(self):
        """Critical risk level returns quarantine action."""
        engine = PolicyEngine()
        risk = RiskResult(risk_score=0.9, risk_level="critical", components={})
        decision = engine.evaluate_risk(risk)
        assert decision.action == "quarantine"
        assert "critical" in decision.reason

    def test_evaluate_risk_high(self):
        """High risk level returns verify action."""
        engine = PolicyEngine()
        risk = RiskResult(risk_score=0.7, risk_level="high", components={})
        decision = engine.evaluate_risk(risk)
        assert decision.action == "verify"
        assert "high" in decision.reason

    def test_evaluate_risk_medium(self):
        """Medium risk level returns allow action."""
        engine = PolicyEngine()
        risk = RiskResult(risk_score=0.3, risk_level="medium", components={})
        decision = engine.evaluate_risk(risk)
        assert decision.action == "allow"

    def test_evaluate_risk_low(self):
        """Low risk level returns allow action."""
        engine = PolicyEngine()
        risk = RiskResult(risk_score=0.1, risk_level="low", components={})
        decision = engine.evaluate_risk(risk)
        assert decision.action == "allow"

    def test_apply_policy(self):
        """Test applying policy to an event."""
        engine = PolicyEngine()
        event = TraceIQEvent(
            run_id="run_001",
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="hello",
            receiver_output="world",
        )
        risk = RiskResult(risk_score=0.9, risk_level="critical", components={})

        updated = engine.apply_policy(event, risk)
        assert updated.policy_action == "quarantine"
        assert updated.event_type == "blocked"
        assert "critical" in updated.policy_reason

    def test_apply_policy_allow(self):
        """Test applying allow policy."""
        engine = PolicyEngine()
        event = TraceIQEvent(
            run_id="run_001",
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="hello",
            receiver_output="world",
        )
        risk = RiskResult(risk_score=0.1, risk_level="low", components={})

        updated = engine.apply_policy(event, risk)
        assert updated.policy_action == "allow"
        assert updated.event_type == "applied"

    def test_trust_decay_on_quarantine(self):
        """Trust decays on quarantine actions."""
        engine = PolicyEngine(enable_trust_decay=True, trust_decay_rate=0.1)
        event = TraceIQEvent(
            run_id="run",
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="x",
            receiver_output="y",
        )

        # Initial trust
        assert engine.get_trust("agent_a") == 1.0

        # Apply critical risk policy
        risk = RiskResult(risk_score=0.9, risk_level="critical", components={})
        engine.apply_policy(event, risk)

        # Trust should have decayed
        assert engine.get_trust("agent_a") < 1.0

    def test_trust_no_decay_on_allow(self):
        """Trust does not decay on allow actions."""
        engine = PolicyEngine(enable_trust_decay=True)
        event = TraceIQEvent(
            run_id="run",
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="x",
            receiver_output="y",
        )

        # Apply low risk policy
        risk = RiskResult(risk_score=0.1, risk_level="low", components={})
        engine.apply_policy(event, risk)

        # Trust unchanged
        assert engine.get_trust("agent_a") == 1.0

    def test_trust_decay_disabled(self):
        """Trust decay can be disabled."""
        engine = PolicyEngine(enable_trust_decay=False)
        event = TraceIQEvent(
            run_id="run",
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="x",
            receiver_output="y",
        )

        # Apply critical risk policy
        risk = RiskResult(risk_score=0.9, risk_level="critical", components={})
        engine.apply_policy(event, risk)

        # Trust unchanged because decay is disabled
        assert engine.get_trust("agent_a") == 1.0

    def test_get_trust_default(self):
        """Default trust is max_trust."""
        engine = PolicyEngine(max_trust=0.9)
        assert engine.get_trust("new_agent") == 0.9

    def test_set_trust(self):
        """Test setting trust score."""
        engine = PolicyEngine()
        engine.set_trust("agent_a", 0.5)
        assert engine.get_trust("agent_a") == 0.5

    def test_set_trust_clamped(self):
        """Trust is clamped to valid range."""
        engine = PolicyEngine(min_trust=0.1, max_trust=0.9)
        engine.set_trust("agent_a", 1.5)
        assert engine.get_trust("agent_a") == 0.9

        engine.set_trust("agent_a", -0.5)
        assert engine.get_trust("agent_a") == 0.1

    def test_reset_trust(self):
        """Test resetting trust score."""
        engine = PolicyEngine()
        engine.set_trust("agent_a", 0.3)
        assert engine.get_trust("agent_a") == 0.3

        engine.reset_trust("agent_a")
        assert engine.get_trust("agent_a") == 1.0

    def test_get_trust_state(self):
        """Test getting full trust state."""
        engine = PolicyEngine()
        event = TraceIQEvent(
            run_id="run",
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="x",
            receiver_output="y",
        )

        # Apply critical risk
        risk = RiskResult(risk_score=0.9, risk_level="critical", components={})
        engine.apply_policy(event, risk)

        state = engine.get_trust_state("agent_a")
        assert state.score < 1.0
        assert state.violations == 1
        assert state.last_action == "quarantine"

    def test_get_all_trust_states(self):
        """Test getting all trust states."""
        engine = PolicyEngine()
        engine.set_trust("agent_a", 0.5)
        engine.set_trust("agent_b", 0.8)

        states = engine.get_all_trust_states()
        assert len(states) == 2
        assert states["agent_a"].score == 0.5
        assert states["agent_b"].score == 0.8

    def test_get_low_trust_agents(self):
        """Test getting low trust agents."""
        engine = PolicyEngine()
        engine.set_trust("agent_a", 0.3)
        engine.set_trust("agent_b", 0.8)
        engine.set_trust("agent_c", 0.4)

        low_trust = engine.get_low_trust_agents(threshold=0.5)
        assert len(low_trust) == 2
        # Sorted by trust ascending
        assert low_trust[0] == ("agent_a", 0.3)
        assert low_trust[1] == ("agent_c", 0.4)

    def test_violations_tracked(self):
        """Violations are tracked on quarantine/block."""
        engine = PolicyEngine()
        event = TraceIQEvent(
            run_id="run",
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="x",
            receiver_output="y",
        )

        risk = RiskResult(risk_score=0.9, risk_level="critical", components={})

        # Multiple violations
        for _ in range(3):
            engine.apply_policy(event, risk)

        state = engine.get_trust_state("agent_a")
        assert state.violations == 3


class TestTrustOutcomeUpdate:
    """Tests for outcome-based trust update."""

    def test_trust_gain_on_correct(self):
        """Trust increases when metadata.correct=True."""
        engine = PolicyEngine(enable_trust_decay=True, trust_gain=0.1)
        engine.set_trust("agent_a", 0.5)

        engine.update_trust_from_outcome(
            agent_id="agent_a",
            risk_level="low",
            policy_action="allow",
            correct=True,
        )

        assert engine.get_trust("agent_a") > 0.5

    def test_trust_decay_on_incorrect(self):
        """Trust decreases when metadata.correct=False."""
        engine = PolicyEngine(enable_trust_decay=True, trust_decay_rate=0.1)
        engine.set_trust("agent_a", 0.8)

        engine.update_trust_from_outcome(
            agent_id="agent_a",
            risk_level="low",
            policy_action="allow",
            correct=False,
        )

        assert engine.get_trust("agent_a") < 0.8

    def test_trust_decay_on_high_risk(self):
        """Trust decreases on high risk level."""
        engine = PolicyEngine(enable_trust_decay=True, trust_decay_rate=0.1)
        engine.set_trust("agent_a", 0.8)

        engine.update_trust_from_outcome(
            agent_id="agent_a",
            risk_level="high",
            policy_action="verify",
            correct=None,  # Unknown outcome
        )

        assert engine.get_trust("agent_a") < 0.8

    def test_no_change_on_low_risk_allow(self):
        """Trust unchanged on low risk allow."""
        engine = PolicyEngine(enable_trust_decay=True)
        engine.set_trust("agent_a", 0.8)

        engine.update_trust_from_outcome(
            agent_id="agent_a",
            risk_level="low",
            policy_action="allow",
            correct=None,
        )

        assert engine.get_trust("agent_a") == 0.8


class TestEdgeWeightScaling:
    """Tests for trust-based edge weight scaling."""

    def test_scale_edge_weight_full_trust(self):
        """Full trust keeps weight unchanged."""
        engine = PolicyEngine()
        # Default trust is 1.0
        scaled = engine.scale_edge_weight("agent_a", 0.5)
        assert scaled == 0.5

    def test_scale_edge_weight_low_trust(self):
        """Low trust scales down weight."""
        engine = PolicyEngine()
        engine.set_trust("agent_a", 0.5)

        scaled = engine.scale_edge_weight("agent_a", 0.8)
        assert scaled == pytest.approx(0.4)

    def test_scale_edge_weight_zero_trust(self):
        """Zero trust zeros the weight."""
        engine = PolicyEngine()
        engine.set_trust("agent_a", 0.0)

        scaled = engine.scale_edge_weight("agent_a", 0.8)
        assert scaled == 0.0


class TestShouldBlockEvent:
    """Tests for should_block_event function."""

    def test_block_critical_risk(self):
        """Critical risk should be blocked."""
        risk = RiskResult(risk_score=0.9, risk_level="critical", components={})
        assert should_block_event(risk) is True

    def test_allow_low_risk(self):
        """Low risk should not be blocked."""
        risk = RiskResult(risk_score=0.1, risk_level="low", components={})
        assert should_block_event(risk) is False

    def test_block_low_trust(self):
        """Low trust should cause blocking."""
        risk = RiskResult(risk_score=0.1, risk_level="low", components={})
        assert should_block_event(risk, trust_score=0.2, trust_threshold=0.3) is True

    def test_allow_high_trust(self):
        """High trust should not cause blocking (for non-critical risk)."""
        risk = RiskResult(risk_score=0.5, risk_level="medium", components={})
        assert should_block_event(risk, trust_score=0.9) is False
