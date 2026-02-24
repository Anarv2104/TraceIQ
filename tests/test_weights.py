"""Tests for the weights module."""

import math

import pytest

from traceiq.weights import (
    alignment_weight,
    bounded_iqx_weight,
    clamp_weight,
    combine_weights,
    confidence_weight,
    sigmoid_weight,
)


class TestClampWeight:
    """Tests for clamp_weight function."""

    def test_value_in_range(self):
        """Value in range is unchanged."""
        assert clamp_weight(0.5) == 0.5
        assert clamp_weight(0.0) == 0.0
        assert clamp_weight(1.0) == 1.0

    def test_clamp_above(self):
        """Value above max is clamped."""
        assert clamp_weight(1.5) == 1.0
        assert clamp_weight(100.0) == 1.0

    def test_clamp_below(self):
        """Value below min is clamped."""
        assert clamp_weight(-0.5) == 0.0
        assert clamp_weight(-100.0) == 0.0

    def test_custom_bounds(self):
        """Test with custom bounds."""
        assert clamp_weight(50, lo=0, hi=25) == 25
        assert clamp_weight(-10, lo=-5, hi=5) == -5
        assert clamp_weight(0.5, lo=-1, hi=1) == 0.5


class TestSigmoidWeight:
    """Tests for sigmoid_weight function."""

    def test_at_center(self):
        """Value at center returns 0.5."""
        assert sigmoid_weight(1.0, center=1.0) == pytest.approx(0.5)
        assert sigmoid_weight(0.0, center=0.0) == pytest.approx(0.5)

    def test_bounded_output(self):
        """Output is always in [0, 1]."""
        for x in [-100, -10, -1, 0, 1, 10, 100]:
            result = sigmoid_weight(x)
            assert 0 <= result <= 1

    def test_above_center_greater_than_half(self):
        """Values above center are > 0.5."""
        assert sigmoid_weight(2.0, center=1.0) > 0.5
        assert sigmoid_weight(3.0, center=1.0) > 0.5

    def test_below_center_less_than_half(self):
        """Values below center are < 0.5."""
        assert sigmoid_weight(0.0, center=1.0) < 0.5
        assert sigmoid_weight(-1.0, center=1.0) < 0.5

    def test_monotonically_increasing(self):
        """Sigmoid is monotonically increasing."""
        values = [sigmoid_weight(x) for x in range(-5, 6)]
        for i in range(len(values) - 1):
            assert values[i] < values[i + 1]

    def test_scale_affects_steepness(self):
        """Smaller scale = steeper transition."""
        # With smaller scale, deviation from center causes bigger change
        small_scale = sigmoid_weight(1.5, center=1.0, scale=0.1)
        large_scale = sigmoid_weight(1.5, center=1.0, scale=1.0)
        # Both > 0.5, but small scale is closer to 1.0
        assert small_scale > large_scale

    def test_extreme_values_no_overflow(self):
        """Extreme values don't cause overflow."""
        assert sigmoid_weight(1000) == pytest.approx(1.0)
        assert sigmoid_weight(-1000) == pytest.approx(0.0)


class TestConfidenceWeight:
    """Tests for confidence_weight function."""

    def test_high_confidence(self):
        """High confidence returns 1.0."""
        assert confidence_weight("high") == 1.0

    def test_medium_confidence(self):
        """Medium confidence returns 0.8."""
        assert confidence_weight("medium") == 0.8

    def test_low_confidence(self):
        """Low confidence returns 0.5."""
        assert confidence_weight("low") == 0.5

    def test_all_values_positive(self):
        """All confidence weights are positive."""
        for quality in ["low", "medium", "high"]:
            assert confidence_weight(quality) > 0


class TestBoundedIqxWeight:
    """Tests for bounded_iqx_weight function."""

    def test_zero_iqx(self):
        """Zero IQx gives small positive weight."""
        weight = bounded_iqx_weight(0.0)
        assert 0 < weight < 0.5

    def test_moderate_iqx(self):
        """Moderate IQx around center gives ~0.5."""
        weight = bounded_iqx_weight(2.0, sigmoid_center=2.0)
        assert weight == pytest.approx(0.5)

    def test_high_iqx(self):
        """High IQx gives weight close to 1."""
        weight = bounded_iqx_weight(10.0)
        assert weight > 0.9

    def test_extreme_iqx_capped(self):
        """Extreme IQx is capped before sigmoid."""
        weight_100 = bounded_iqx_weight(100.0)
        weight_1000 = bounded_iqx_weight(1000.0)
        # Both should be essentially 1.0 due to capping and sigmoid
        assert weight_100 == pytest.approx(weight_1000, rel=1e-3)

    def test_output_always_bounded(self):
        """Output is always in [0, 1]."""
        for iqx in [0, 0.1, 1, 5, 10, 25, 50, 100]:
            weight = bounded_iqx_weight(iqx)
            assert 0 <= weight <= 1

    def test_custom_cap(self):
        """Test with custom cap value."""
        weight = bounded_iqx_weight(100.0, cap=5.0)
        # Should be same as weight for 5.0
        expected = bounded_iqx_weight(5.0, cap=5.0)
        assert weight == pytest.approx(expected)


class TestAlignmentWeight:
    """Tests for alignment_weight function."""

    def test_positive_alignment_unchanged(self):
        """Positive alignment is unchanged."""
        assert alignment_weight(0.8) == 0.8
        assert alignment_weight(1.0) == 1.0

    def test_negative_alignment_floored(self):
        """Negative alignment is floored to 0."""
        assert alignment_weight(-0.5) == 0.0
        assert alignment_weight(-1.0) == 0.0

    def test_zero_alignment(self):
        """Zero alignment stays zero."""
        assert alignment_weight(0.0) == 0.0

    def test_custom_floor(self):
        """Test with custom floor value."""
        assert alignment_weight(-0.5, floor=0.1) == 0.1
        assert alignment_weight(0.5, floor=0.1) == 0.5


class TestCombineWeights:
    """Tests for combine_weights function."""

    def test_product_method(self):
        """Test product combination."""
        result = combine_weights(0.5, 0.8, method="product")
        assert result == pytest.approx(0.4)

    def test_mean_method(self):
        """Test mean combination."""
        result = combine_weights(0.5, 0.8, method="mean")
        assert result == pytest.approx(0.65)

    def test_min_method(self):
        """Test min combination."""
        result = combine_weights(0.5, 0.8, method="min")
        assert result == 0.5

    def test_max_method(self):
        """Test max combination."""
        result = combine_weights(0.5, 0.8, method="max")
        assert result == 0.8

    def test_empty_weights(self):
        """Empty weights returns 0."""
        assert combine_weights(method="product") == 0.0
        assert combine_weights(method="mean") == 0.0

    def test_single_weight(self):
        """Single weight is returned as-is for most methods."""
        assert combine_weights(0.7, method="mean") == 0.7
        assert combine_weights(0.7, method="min") == 0.7
        assert combine_weights(0.7, method="max") == 0.7
        assert combine_weights(0.7, method="product") == 0.7

    def test_invalid_method(self):
        """Invalid method raises ValueError."""
        with pytest.raises(ValueError):
            combine_weights(0.5, method="invalid")
