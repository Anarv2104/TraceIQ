"""Validity gating for cold-start safety.

This module provides functions to check whether computed metrics are valid
based on baseline sample count and state quality. This prevents alerting
on unreliable metrics during the cold-start period.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Default configuration
DEFAULT_BASELINE_K = 20  # Minimum samples before valid metrics
DEFAULT_WINDOW_SIZE = 50  # Default window for windowed computations
DEFAULT_MAD_FLOOR = 1e-6  # Minimum MAD for valid variance


@dataclass
class ValidityResult:
    """Result of validity check for metric computation.

    Attributes:
        valid: Whether metrics should be considered valid
        invalid_reason: Reason for invalidity (if valid=False)
        confidence: Confidence level based on state quality
    """

    valid: bool
    invalid_reason: str | None
    confidence: Literal["low", "medium", "high"]

    def __bool__(self) -> bool:
        """Allow using ValidityResult in boolean contexts."""
        return self.valid


def check_validity(
    baseline_samples: int,
    state_quality: Literal["low", "medium", "high"],
    mad_value: float | None = None,
    baseline_k: int = DEFAULT_BASELINE_K,
    mad_floor: float = DEFAULT_MAD_FLOOR,
) -> ValidityResult:
    """Check if computed metrics should be considered valid.

    This function implements validity gating to prevent false alerts during
    cold-start periods. Metrics are invalid if we don't have enough baseline
    samples to reliably compute IQx and Z-scores, or if there is no variance.

    STRICT RULES:
    1. If baseline_samples < baseline_k: valid=False, invalid_reason="cold_start"
    2. If mad_value is not None and mad_value < mad_floor: valid=False, invalid_reason="no_variance"
    3. Map state_quality to confidence: "high" -> "high", "medium" -> "medium", "low" -> "low"

    Args:
        baseline_samples: Number of baseline samples accumulated so far
        state_quality: Quality of state tracking ("low", "medium", "high")
        mad_value: Median Absolute Deviation value (optional, for variance check)
        baseline_k: Minimum samples required for valid metrics (default: 20)
        mad_floor: Minimum MAD for valid variance (default: 1e-6)

    Returns:
        ValidityResult with validity status, reason, and confidence level

    Examples:
        >>> result = check_validity(5, "medium")
        >>> result.valid
        False
        >>> result.invalid_reason
        'cold_start'

        >>> result = check_validity(25, "high")
        >>> result.valid
        True
        >>> result.confidence
        'high'

        >>> result = check_validity(25, "medium", mad_value=0.0)
        >>> result.valid
        False
        >>> result.invalid_reason
        'no_variance'
    """
    # Map state_quality to confidence
    confidence: Literal["low", "medium", "high"] = state_quality

    # Check if we have enough baseline samples
    if baseline_samples < baseline_k:
        return ValidityResult(
            valid=False,
            invalid_reason="cold_start",
            confidence=confidence,
        )

    # Check for no variance (if MAD is provided)
    if mad_value is not None and mad_value < mad_floor:
        return ValidityResult(
            valid=False,
            invalid_reason="no_variance",
            confidence=confidence,
        )

    # All checks passed
    return ValidityResult(
        valid=True,
        invalid_reason=None,
        confidence=confidence,
    )


def should_alert(
    validity: ValidityResult,
    z_score: float | None,
    anomaly_threshold: float = 2.0,
) -> bool:
    """Determine if an alert should be raised based on validity and Z-score.

    IMPORTANT: This function implements the critical rule that alerts should
    NOT be raised during cold-start periods, regardless of Z-score values.

    Args:
        validity: Validity check result
        z_score: Computed Z-score (or None if not computed)
        anomaly_threshold: Threshold for Z-score alerting

    Returns:
        True if an alert should be raised, False otherwise

    Examples:
        >>> validity = ValidityResult(valid=False, invalid_reason="cold_start", confidence="low")
        >>> should_alert(validity, z_score=5.0)
        False  # No alert on cold start, even with high Z-score

        >>> validity = ValidityResult(valid=True, invalid_reason=None, confidence="high")
        >>> should_alert(validity, z_score=3.0)
        True  # Alert when valid and Z-score exceeds threshold
    """
    # Never alert if metrics are invalid
    if not validity.valid:
        return False

    # Never alert if Z-score is not computed
    if z_score is None:
        return False

    # Alert if Z-score exceeds threshold
    return abs(z_score) > anomaly_threshold


def compute_effective_threshold(
    base_threshold: float,
    confidence: Literal["low", "medium", "high"],
) -> float:
    """Adjust alert threshold based on confidence level.

    Higher confidence allows tighter (lower) thresholds, while lower
    confidence requires more extreme values to trigger alerts.

    Args:
        base_threshold: Base Z-score threshold for alerts
        confidence: Confidence level from state quality

    Returns:
        Adjusted threshold value

    Examples:
        >>> compute_effective_threshold(2.0, "high")
        2.0
        >>> compute_effective_threshold(2.0, "medium")
        2.5
        >>> compute_effective_threshold(2.0, "low")
        3.0
    """
    multipliers = {
        "high": 1.0,
        "medium": 1.25,
        "low": 1.5,
    }
    return base_threshold * multipliers[confidence]
