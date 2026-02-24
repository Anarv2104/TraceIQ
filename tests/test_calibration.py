"""Tests for risk calibration functionality."""

import numpy as np
import pytest

from traceiq.risk import (
    RiskThresholds,
    assign_risk_level,
    calibrate_thresholds,
)


class TestRiskThresholds:
    """Tests for RiskThresholds dataclass."""

    def test_creation(self):
        """Test creating risk thresholds."""
        thresholds = RiskThresholds(medium=0.3, high=0.6, critical=0.85)
        assert thresholds.medium == 0.3
        assert thresholds.high == 0.6
        assert thresholds.critical == 0.85

    def test_as_tuple(self):
        """Test conversion to tuple."""
        thresholds = RiskThresholds(medium=0.2, high=0.5, critical=0.8)
        assert thresholds.as_tuple() == (0.2, 0.5, 0.8)


class TestCalibrateThresholds:
    """Tests for calibrate_thresholds function."""

    def test_sufficient_samples(self):
        """Test calibration with sufficient samples."""
        # Generate scores with known percentiles
        scores = list(np.linspace(0, 1, 100))
        thresholds = calibrate_thresholds(scores)

        # Check percentile-based thresholds
        assert thresholds.medium == pytest.approx(0.80, rel=0.1)
        assert thresholds.high == pytest.approx(0.95, rel=0.1)
        assert thresholds.critical == pytest.approx(0.99, rel=0.1)

    def test_insufficient_samples_fallback(self):
        """Test fallback with insufficient samples."""
        scores = [0.1, 0.2, 0.3, 0.4, 0.5]  # Only 5 samples
        thresholds = calibrate_thresholds(scores)

        # Should use conservative defaults
        assert thresholds.medium == 0.3
        assert thresholds.high == 0.6
        assert thresholds.critical == 0.85

    def test_empty_list(self):
        """Test with empty list."""
        thresholds = calibrate_thresholds([])
        assert thresholds.medium == 0.3  # Conservative default

    def test_custom_percentiles(self):
        """Test with custom percentiles."""
        scores = list(np.linspace(0, 1, 100))
        thresholds = calibrate_thresholds(scores, percentiles=(50, 75, 90))

        assert thresholds.medium == pytest.approx(0.50, rel=0.1)
        assert thresholds.high == pytest.approx(0.75, rel=0.1)
        assert thresholds.critical == pytest.approx(0.90, rel=0.1)

    def test_skewed_distribution(self):
        """Test with skewed distribution."""
        # Most scores low, few high
        scores = [0.1] * 90 + [0.8, 0.9, 0.95, 0.99] + [0.2] * 6
        thresholds = calibrate_thresholds(scores)

        # Should reflect the actual distribution
        assert thresholds.medium < 0.5  # Most values are low
        assert thresholds.critical > 0.8  # Only top 1% is high


class TestAssignRiskLevel:
    """Tests for assign_risk_level function."""

    def test_none_returns_unknown(self):
        """None score returns unknown."""
        thresholds = RiskThresholds(medium=0.3, high=0.6, critical=0.85)
        assert assign_risk_level(None, thresholds) == "unknown"

    def test_low_risk(self):
        """Score below medium threshold is low."""
        thresholds = RiskThresholds(medium=0.3, high=0.6, critical=0.85)
        assert assign_risk_level(0.1, thresholds) == "low"
        assert assign_risk_level(0.29, thresholds) == "low"

    def test_medium_risk(self):
        """Score at or above medium but below high is medium."""
        thresholds = RiskThresholds(medium=0.3, high=0.6, critical=0.85)
        assert assign_risk_level(0.3, thresholds) == "medium"
        assert assign_risk_level(0.5, thresholds) == "medium"
        assert assign_risk_level(0.59, thresholds) == "medium"

    def test_high_risk(self):
        """Score at or above high but below critical is high."""
        thresholds = RiskThresholds(medium=0.3, high=0.6, critical=0.85)
        assert assign_risk_level(0.6, thresholds) == "high"
        assert assign_risk_level(0.7, thresholds) == "high"
        assert assign_risk_level(0.84, thresholds) == "high"

    def test_critical_risk(self):
        """Score at or above critical is critical."""
        thresholds = RiskThresholds(medium=0.3, high=0.6, critical=0.85)
        assert assign_risk_level(0.85, thresholds) == "critical"
        assert assign_risk_level(0.95, thresholds) == "critical"
        assert assign_risk_level(1.0, thresholds) == "critical"


class TestCalibrationCurveMonotonicity:
    """Tests for calibration curve properties."""

    def test_risk_increases_with_score(self):
        """Higher risk bins should have higher observed failure rates.

        This is a property test rather than exact validation.
        """
        # Generate synthetic data where failures correlate with risk
        np.random.seed(42)
        n_samples = 1000
        risk_scores = np.random.beta(2, 5, n_samples)  # Skewed toward low

        # Outcomes correlated with risk (higher risk = more likely to fail)
        outcomes = np.random.random(n_samples) < (risk_scores**2)

        # Bin and compute observed rates
        n_bins = 5
        bins = np.linspace(0, 1, n_bins + 1)
        bin_rates = []

        for i in range(n_bins):
            mask = (risk_scores >= bins[i]) & (risk_scores < bins[i + 1])
            if mask.sum() > 0:
                rate = outcomes[mask].mean()
                bin_rates.append(rate)

        # Check monotonicity (allow some noise)
        # Higher bins should generally have higher failure rates
        if len(bin_rates) >= 3:
            # At least first half should be lower than second half on average
            first_half_avg = np.mean(bin_rates[: len(bin_rates) // 2])
            second_half_avg = np.mean(bin_rates[len(bin_rates) // 2 :])
            assert first_half_avg < second_half_avg

    def test_well_calibrated_predictions(self):
        """Well-calibrated model should have diagonal calibration curve."""
        np.random.seed(42)
        n_samples = 1000

        # Perfectly calibrated: P(outcome=True | risk=r) = r
        risk_scores = np.random.uniform(0, 1, n_samples)
        outcomes = np.random.random(n_samples) < risk_scores

        # Bin and compute rates
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        for i in range(n_bins):
            mask = (risk_scores >= bins[i]) & (risk_scores < bins[i + 1])
            if mask.sum() > 10:  # Need enough samples
                observed_rate = outcomes[mask].mean()
                expected_rate = bin_centers[i]
                # Should be close to diagonal (within 0.2)
                assert abs(observed_rate - expected_rate) < 0.2
