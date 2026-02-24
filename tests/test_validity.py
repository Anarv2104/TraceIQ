"""Tests for the validity module."""

import pytest

from traceiq.validity import (
    DEFAULT_BASELINE_K,
    ValidityResult,
    check_validity,
    compute_effective_threshold,
    should_alert,
)


class TestValidityResult:
    """Tests for ValidityResult dataclass."""

    def test_valid_result(self):
        """Test creating a valid result."""
        result = ValidityResult(valid=True, invalid_reason=None, confidence="high")
        assert result.valid is True
        assert result.invalid_reason is None
        assert result.confidence == "high"
        assert bool(result) is True

    def test_invalid_result(self):
        """Test creating an invalid result."""
        result = ValidityResult(
            valid=False, invalid_reason="cold_start", confidence="low"
        )
        assert result.valid is False
        assert result.invalid_reason == "cold_start"
        assert result.confidence == "low"
        assert bool(result) is False


class TestCheckValidity:
    """Tests for check_validity function."""

    def test_cold_start_invalid(self):
        """Baseline < K => invalid with cold_start reason."""
        result = check_validity(
            baseline_samples=5,
            state_quality="medium",
            baseline_k=20,
        )
        assert result.valid is False
        assert result.invalid_reason == "cold_start"
        assert result.confidence == "medium"

    def test_sufficient_samples_valid(self):
        """Baseline >= K => valid."""
        result = check_validity(
            baseline_samples=25,
            state_quality="high",
            baseline_k=20,
        )
        assert result.valid is True
        assert result.invalid_reason is None
        assert result.confidence == "high"

    def test_exactly_k_samples(self):
        """Baseline == K => valid (edge case)."""
        result = check_validity(
            baseline_samples=20,
            state_quality="medium",
            baseline_k=20,
        )
        assert result.valid is True

    def test_one_less_than_k(self):
        """Baseline == K-1 => invalid (edge case)."""
        result = check_validity(
            baseline_samples=19,
            state_quality="medium",
            baseline_k=20,
        )
        assert result.valid is False

    def test_zero_samples(self):
        """Zero samples => invalid."""
        result = check_validity(
            baseline_samples=0,
            state_quality="low",
        )
        assert result.valid is False
        assert result.invalid_reason == "cold_start"

    def test_default_baseline_k(self):
        """Test default baseline_k is used."""
        # With default baseline_k=20
        result = check_validity(baseline_samples=19, state_quality="medium")
        assert result.valid is False

        result = check_validity(baseline_samples=20, state_quality="medium")
        assert result.valid is True

    def test_confidence_mapping(self):
        """Test state_quality is mapped to confidence."""
        for quality in ["low", "medium", "high"]:
            result = check_validity(
                baseline_samples=25,
                state_quality=quality,
            )
            assert result.confidence == quality

    def test_no_variance_invalid(self):
        """MAD < mad_floor => valid=False, invalid_reason='no_variance'."""
        result = check_validity(
            baseline_samples=25,
            state_quality="high",
            mad_value=0.0,  # Zero variance
        )
        assert result.valid is False
        assert result.invalid_reason == "no_variance"

    def test_no_variance_with_small_mad(self):
        """Very small MAD below floor is also invalid."""
        result = check_validity(
            baseline_samples=25,
            state_quality="high",
            mad_value=1e-8,  # Below default floor of 1e-6
        )
        assert result.valid is False
        assert result.invalid_reason == "no_variance"

    def test_sufficient_variance_valid(self):
        """MAD above floor with enough samples is valid."""
        result = check_validity(
            baseline_samples=25,
            state_quality="high",
            mad_value=0.1,  # Above floor
        )
        assert result.valid is True
        assert result.invalid_reason is None

    def test_cold_start_takes_precedence(self):
        """Cold start check happens before variance check."""
        result = check_validity(
            baseline_samples=5,  # Cold start
            state_quality="high",
            mad_value=0.0,  # Also no variance
        )
        # Cold start should be the reason, not no_variance
        assert result.valid is False
        assert result.invalid_reason == "cold_start"

    def test_custom_mad_floor(self):
        """Test with custom MAD floor."""
        result = check_validity(
            baseline_samples=25,
            state_quality="high",
            mad_value=0.001,
            mad_floor=0.01,  # Higher floor
        )
        assert result.valid is False
        assert result.invalid_reason == "no_variance"


class TestShouldAlert:
    """Tests for should_alert function."""

    def test_no_alert_on_cold_start(self):
        """Alert must be False when invalid, regardless of Z-score."""
        validity = ValidityResult(
            valid=False, invalid_reason="cold_start", confidence="low"
        )
        # Even with extremely high Z-score, should not alert
        assert should_alert(validity, z_score=100.0) is False
        assert should_alert(validity, z_score=5.0) is False
        assert should_alert(validity, z_score=10.0) is False

    def test_alert_when_valid_above_threshold(self):
        """Alert when valid and Z-score exceeds threshold."""
        validity = ValidityResult(valid=True, invalid_reason=None, confidence="high")
        assert should_alert(validity, z_score=3.0, anomaly_threshold=2.0) is True

    def test_no_alert_when_valid_below_threshold(self):
        """No alert when valid but Z-score below threshold."""
        validity = ValidityResult(valid=True, invalid_reason=None, confidence="high")
        assert should_alert(validity, z_score=1.5, anomaly_threshold=2.0) is False

    def test_no_alert_with_none_z_score(self):
        """No alert when Z-score is None."""
        validity = ValidityResult(valid=True, invalid_reason=None, confidence="high")
        assert should_alert(validity, z_score=None) is False

    def test_alert_with_negative_z_score(self):
        """Alert on negative Z-score with large absolute value."""
        validity = ValidityResult(valid=True, invalid_reason=None, confidence="high")
        assert should_alert(validity, z_score=-3.0, anomaly_threshold=2.0) is True

    def test_default_threshold(self):
        """Test default anomaly threshold of 2.0."""
        validity = ValidityResult(valid=True, invalid_reason=None, confidence="high")
        assert should_alert(validity, z_score=2.5) is True
        assert should_alert(validity, z_score=1.5) is False


class TestComputeEffectiveThreshold:
    """Tests for compute_effective_threshold function."""

    def test_high_confidence_no_adjustment(self):
        """High confidence keeps threshold unchanged."""
        threshold = compute_effective_threshold(2.0, "high")
        assert threshold == pytest.approx(2.0)

    def test_medium_confidence_increases_threshold(self):
        """Medium confidence increases threshold by 25%."""
        threshold = compute_effective_threshold(2.0, "medium")
        assert threshold == pytest.approx(2.5)

    def test_low_confidence_increases_threshold_more(self):
        """Low confidence increases threshold by 50%."""
        threshold = compute_effective_threshold(2.0, "low")
        assert threshold == pytest.approx(3.0)

    def test_different_base_thresholds(self):
        """Test with different base thresholds."""
        assert compute_effective_threshold(1.0, "high") == pytest.approx(1.0)
        assert compute_effective_threshold(1.0, "medium") == pytest.approx(1.25)
        assert compute_effective_threshold(1.0, "low") == pytest.approx(1.5)

        assert compute_effective_threshold(4.0, "high") == pytest.approx(4.0)
        assert compute_effective_threshold(4.0, "medium") == pytest.approx(5.0)
        assert compute_effective_threshold(4.0, "low") == pytest.approx(6.0)


class TestDefaultConstants:
    """Tests for default constant values."""

    def test_default_baseline_k(self):
        """Test DEFAULT_BASELINE_K is reasonable."""
        assert DEFAULT_BASELINE_K == 20
        assert DEFAULT_BASELINE_K > 0
