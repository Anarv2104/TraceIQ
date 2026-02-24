"""Tests for the risk module."""

import pytest

from traceiq.risk import (
    DEFAULT_RISK_THRESHOLDS,
    RiskResult,
    aggregate_risk_scores,
    classify_risk_level,
    compute_risk_score,
)


class TestRiskResult:
    """Tests for RiskResult dataclass."""

    def test_valid_result(self):
        """Test creating a valid risk result."""
        result = RiskResult(risk_score=0.5, risk_level="medium", components={"a": 1})
        assert result.risk_score == 0.5
        assert result.risk_level == "medium"
        assert result.components == {"a": 1}
        assert result.is_valid is True
        assert bool(result) is True

    def test_unknown_result(self):
        """Test creating an unknown risk result."""
        result = RiskResult(risk_score=None, risk_level="unknown", components={})
        assert result.risk_score is None
        assert result.risk_level == "unknown"
        assert result.is_valid is False
        assert bool(result) is False

    def test_zero_risk(self):
        """Test that zero risk evaluates to False."""
        result = RiskResult(risk_score=0.0, risk_level="low", components={})
        assert bool(result) is False
        assert result.is_valid is True

    def test_requires_attention(self):
        """Test requires_attention property."""
        assert RiskResult(0.9, "critical", {}).requires_attention is True
        assert RiskResult(0.7, "high", {}).requires_attention is True
        assert RiskResult(0.3, "medium", {}).requires_attention is False
        assert RiskResult(0.1, "low", {}).requires_attention is False
        assert RiskResult(None, "unknown", {}).requires_attention is False


class TestComputeRiskScore:
    """Tests for compute_risk_score function."""

    def test_invalid_returns_unknown(self):
        """Invalid metrics return unknown risk level."""
        result = compute_risk_score(robust_z=3.0, drift=0.5, valid=False)
        assert result.risk_score is None
        assert result.risk_level == "unknown"
        assert "invalid_metrics" in str(result.components.get("reason", ""))

    def test_missing_z_score_returns_unknown(self):
        """Missing Z-score returns unknown risk level."""
        result = compute_risk_score(robust_z=None, drift=0.5, valid=True)
        assert result.risk_score is None
        assert result.risk_level == "unknown"
        assert "missing_z_score" in str(result.components.get("reason", ""))

    def test_risk_score_in_range(self):
        """Risk score is always in [0, 1]."""
        # Test various combinations
        test_cases = [
            (0.0, 0.0),
            (1.0, 0.5),
            (3.0, 0.5),
            (5.0, 1.0),
            (10.0, 2.0),
        ]
        for z, drift in test_cases:
            result = compute_risk_score(robust_z=z, drift=drift)
            assert 0 <= result.risk_score <= 1, f"Failed for z={z}, drift={drift}"

    def test_risk_level_thresholds(self):
        """Test risk level classification based on thresholds."""
        # Default thresholds: (0.2, 0.5, 0.8)
        # Low Z and drift should give low risk
        result = compute_risk_score(robust_z=0.5, drift=0.1)
        assert result.risk_level == "low"

        # High Z and drift should give higher risk
        result = compute_risk_score(robust_z=5.0, drift=2.0)
        assert result.risk_level in ("high", "critical")

    def test_negative_z_score_uses_absolute(self):
        """Negative Z-score is treated as positive for risk."""
        result_pos = compute_risk_score(robust_z=3.0, drift=0.5)
        result_neg = compute_risk_score(robust_z=-3.0, drift=0.5)
        # Should be similar since we use abs(z)
        assert result_pos.risk_score == pytest.approx(result_neg.risk_score, rel=0.1)

    def test_drift_none_defaults_to_zero(self):
        """Missing drift defaults to 0."""
        result = compute_risk_score(robust_z=2.0, drift=None)
        assert result.risk_score is not None
        # With zero drift, risk should be lower
        result_with_drift = compute_risk_score(robust_z=2.0, drift=1.0)
        assert result.risk_score < result_with_drift.risk_score

    def test_pr_amplification(self):
        """PR > 0 amplifies risk."""
        result_no_pr = compute_risk_score(robust_z=2.0, drift=0.5, pr_window=0.0)
        result_with_pr = compute_risk_score(robust_z=2.0, drift=0.5, pr_window=1.0)
        assert result_with_pr.risk_score > result_no_pr.risk_score

    def test_alignment_affects_risk(self):
        """Positive alignment increases risk, negative reduces it."""
        result_pos = compute_risk_score(robust_z=2.0, drift=0.5, alignment=1.0)
        result_neg = compute_risk_score(robust_z=2.0, drift=0.5, alignment=-0.5)
        assert result_neg.risk_score < result_pos.risk_score

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        result = compute_risk_score(
            robust_z=2.0,
            drift=0.5,
            thresholds=(0.1, 0.3, 0.5),
        )
        # With lower thresholds, same risk score maps to higher level
        assert result.risk_level in ("medium", "high", "critical")

    def test_components_included(self):
        """Components dict includes all factors."""
        result = compute_risk_score(
            robust_z=2.0, drift=0.5, alignment=0.8, pr_window=0.5
        )
        assert "robust_z" in result.components
        assert "drift" in result.components
        assert "alignment" in result.components
        assert "pr_window" in result.components
        assert "z_component" in result.components
        assert "drift_component" in result.components
        assert "risk_core" in result.components


class TestClassifyRiskLevel:
    """Tests for classify_risk_level function."""

    def test_none_returns_unknown(self):
        """None score returns unknown."""
        assert classify_risk_level(None) == "unknown"

    def test_low_risk(self):
        """Score < 0.2 is low."""
        assert classify_risk_level(0.0) == "low"
        assert classify_risk_level(0.1) == "low"
        assert classify_risk_level(0.19) == "low"

    def test_medium_risk(self):
        """Score in [0.2, 0.5) is medium."""
        assert classify_risk_level(0.2) == "medium"
        assert classify_risk_level(0.3) == "medium"
        assert classify_risk_level(0.49) == "medium"

    def test_high_risk(self):
        """Score in [0.5, 0.8) is high."""
        assert classify_risk_level(0.5) == "high"
        assert classify_risk_level(0.6) == "high"
        assert classify_risk_level(0.79) == "high"

    def test_critical_risk(self):
        """Score >= 0.8 is critical."""
        assert classify_risk_level(0.8) == "critical"
        assert classify_risk_level(0.9) == "critical"
        assert classify_risk_level(1.0) == "critical"

    def test_custom_thresholds(self):
        """Test with custom thresholds."""
        # Very tight thresholds
        assert classify_risk_level(0.15, (0.1, 0.2, 0.3)) == "medium"
        assert classify_risk_level(0.25, (0.1, 0.2, 0.3)) == "high"
        assert classify_risk_level(0.35, (0.1, 0.2, 0.3)) == "critical"


class TestAggregateRiskScores:
    """Tests for aggregate_risk_scores function."""

    def test_mean_aggregation(self):
        """Test mean aggregation."""
        result = aggregate_risk_scores([0.2, 0.4, 0.6], method="mean")
        assert result == pytest.approx(0.4)

    def test_max_aggregation(self):
        """Test max aggregation."""
        result = aggregate_risk_scores([0.2, 0.4, 0.6], method="max")
        assert result == 0.6

    def test_p95_aggregation(self):
        """Test 95th percentile aggregation."""
        result = aggregate_risk_scores([0.1, 0.2, 0.3, 0.4, 0.9], method="p95")
        assert result > 0.4  # Should be close to 0.9

    def test_filters_none_values(self):
        """None values are filtered out."""
        result = aggregate_risk_scores([None, 0.2, None, 0.4, None], method="mean")
        assert result == pytest.approx(0.3)

    def test_all_none_returns_none(self):
        """All None values returns None."""
        result = aggregate_risk_scores([None, None, None])
        assert result is None

    def test_empty_list_returns_none(self):
        """Empty list returns None."""
        result = aggregate_risk_scores([])
        assert result is None

    def test_single_value(self):
        """Single value returns that value."""
        assert aggregate_risk_scores([0.5], method="mean") == 0.5
        assert aggregate_risk_scores([0.5], method="max") == 0.5

    def test_invalid_method(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError):
            aggregate_risk_scores([0.5], method="invalid")


class TestExposureAffectsRisk:
    """Tests for exposure term in risk scoring."""

    def test_exposure_increases_risk(self):
        """Higher exposure should increase risk score."""
        result_low_exp = compute_risk_score(robust_z=2.0, drift=0.5, exposure=0.0)
        result_high_exp = compute_risk_score(robust_z=2.0, drift=0.5, exposure=10.0)
        assert result_high_exp.risk_score > result_low_exp.risk_score

    def test_exposure_in_components(self):
        """Exposure should be tracked in components."""
        result = compute_risk_score(robust_z=2.0, drift=0.5, exposure=5.0)
        assert "exposure" in result.components
        assert result.components["exposure"] == 5.0
        assert "exposure_factor" in result.components

    def test_zero_exposure(self):
        """Zero exposure should not cause errors."""
        result = compute_risk_score(robust_z=2.0, drift=0.5, exposure=0.0)
        assert result.risk_score is not None
        assert 0 <= result.risk_score <= 1

    def test_large_exposure(self):
        """Large exposure should be bounded by sigmoid."""
        result = compute_risk_score(robust_z=2.0, drift=0.5, exposure=100.0)
        assert result.risk_score is not None
        assert result.risk_score <= 1.0


class TestDefaultConstants:
    """Tests for default constants."""

    def test_default_thresholds(self):
        """Test default risk thresholds are reasonable."""
        assert DEFAULT_RISK_THRESHOLDS == (0.2, 0.5, 0.8)
        assert all(0 < t < 1 for t in DEFAULT_RISK_THRESHOLDS)
        assert (
            DEFAULT_RISK_THRESHOLDS[0]
            < DEFAULT_RISK_THRESHOLDS[1]
            < DEFAULT_RISK_THRESHOLDS[2]
        )
