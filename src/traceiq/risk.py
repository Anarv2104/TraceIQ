"""Risk scoring for influence events.

This module provides the core risk scoring functionality that combines
multiple metrics into a single bounded risk score. This is the primary
output metric for production use.

Risk Scoring Formula:
    risk_core = sigmoid(a * robust_z) * sigmoid(b * drift) * max(0, alignment)
    risk_with_pr = risk_core * (1 + gamma * pr_window)
    risk_final = risk_with_pr * (1.0 + 0.5 * exposure_factor)
    risk_final = clamp(risk_final, 0, 1)

Calibration:
    Thresholds should be calibrated from historical data using percentiles.
    Default thresholds are conservative fallbacks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

from traceiq.weights import alignment_weight, clamp_weight, sigmoid_weight

# Default risk thresholds for level classification
DEFAULT_RISK_THRESHOLDS = (0.2, 0.5, 0.8)

# Default exposure scale for sigmoid transformation
DEFAULT_EXPOSURE_SCALE = 5.0


@dataclass
class RiskThresholds:
    """Calibrated thresholds based on percentiles.

    Attributes:
        medium: Threshold for medium risk (>= p80)
        high: Threshold for high risk (>= p95)
        critical: Threshold for critical risk (>= p99)
    """

    medium: float  # >= p80
    high: float  # >= p95
    critical: float  # >= p99

    def as_tuple(self) -> tuple[float, float, float]:
        """Convert to threshold tuple format."""
        return (self.medium, self.high, self.critical)


@dataclass
class RiskResult:
    """Result of risk score computation.

    Attributes:
        risk_score: Computed risk score in [0, 1], or None if invalid
        risk_level: Categorical risk level
        components: Breakdown of risk factor contributions
    """

    risk_score: float | None
    risk_level: Literal["unknown", "low", "medium", "high", "critical"]
    components: dict[str, Any] = field(default_factory=dict)

    def __bool__(self) -> bool:
        """Return True if risk is known and non-zero."""
        return self.risk_score is not None and self.risk_score > 0

    @property
    def is_valid(self) -> bool:
        """Check if risk score was computed successfully."""
        return self.risk_score is not None

    @property
    def requires_attention(self) -> bool:
        """Check if risk level warrants attention."""
        return self.risk_level in ("high", "critical")


def compute_risk_score(
    robust_z: float | None,
    drift: float | None,
    alignment: float = 1.0,
    pr_window: float = 0.0,
    exposure: float = 0.0,
    valid: bool = True,
    a: float = 1.0,
    b: float = 1.0,
    gamma: float = 0.5,
    exposure_scale: float = DEFAULT_EXPOSURE_SCALE,
    thresholds: tuple[float, float, float] = DEFAULT_RISK_THRESHOLDS,
) -> RiskResult:
    """Compute bounded risk score from component metrics.

    The risk score combines:
    - Robust Z-score (anomaly magnitude)
    - Drift (state change magnitude)
    - Alignment (direction of influence)
    - Propagation Risk (network instability)
    - Exposure (downstream consumption count or out_degree)

    Formula:
        risk_core = sigmoid(a * robust_z) * sigmoid(b * drift) * max(0, alignment)
        risk_with_pr = risk_core * (1 + gamma * pr_window)
        exposure_factor = sigmoid(exposure / exposure_scale)
        risk_final = risk_with_pr * (1.0 + 0.5 * exposure_factor)
        risk_final = clamp(risk_final, 0, 1)

    Args:
        robust_z: Robust Z-score (or None if not computed)
        drift: L2 drift value (or None if not computed)
        alignment: Alignment score, typically in [-1, 1] (default: 1.0)
        pr_window: Windowed propagation risk (default: 0.0)
        exposure: Downstream consumption count or out_degree (default: 0.0)
        valid: Whether metrics are valid (False during cold start)
        a: Coefficient for Z-score sigmoid (default: 1.0)
        b: Coefficient for drift sigmoid (default: 1.0)
        gamma: Coefficient for PR amplification (default: 0.5)
        exposure_scale: Scale for exposure sigmoid transformation (default: 5.0)
        thresholds: (low, medium, high) thresholds for risk levels

    Returns:
        RiskResult with computed risk score and level

    Examples:
        >>> result = compute_risk_score(robust_z=3.0, drift=0.5)
        >>> result.risk_score
        0.2189...
        >>> result.risk_level
        'medium'

        >>> result = compute_risk_score(robust_z=None, drift=0.5, valid=False)
        >>> result.risk_score is None
        True
        >>> result.risk_level
        'unknown'
    """
    # Handle invalid metrics
    if not valid or robust_z is None:
        return RiskResult(
            risk_score=None,
            risk_level="unknown",
            components={
                "valid": valid,
                "robust_z": robust_z,
                "drift": drift,
                "reason": "invalid_metrics" if not valid else "missing_z_score",
            },
        )

    # Default drift to 0 if not provided
    if drift is None:
        drift = 0.0

    # Compute component sigmoids
    z_component = sigmoid_weight(abs(robust_z), center=2.0, scale=a)
    drift_component = sigmoid_weight(drift, center=0.5, scale=b)
    alignment_component = alignment_weight(alignment, floor=0.0)

    # Compute core risk (product of components)
    risk_core = z_component * drift_component * alignment_component

    # Apply PR amplification
    # PR values > 1 indicate potential amplification, scale impact
    pr_factor = 1.0 + gamma * max(0.0, pr_window)
    risk_with_pr = risk_core * pr_factor

    # Apply exposure term
    # Higher exposure (more downstream consumers) increases risk
    # Use additive amplification: exposure=0 -> factor=1.0, exposure>0 -> factor>1.0
    exposure_factor = sigmoid_weight(exposure / exposure_scale, center=1.0, scale=0.5)
    # Scale so that exposure=0 gives ~1.0, and high exposure gives up to ~1.5
    exposure_multiplier = 1.0 + 0.5 * exposure_factor
    risk_final = risk_with_pr * exposure_multiplier

    # Clamp to [0, 1]
    risk_final = clamp_weight(risk_final, 0.0, 1.0)

    # Determine risk level based on thresholds
    low_thresh, med_thresh, high_thresh = thresholds
    if risk_final < low_thresh:
        risk_level = "low"
    elif risk_final < med_thresh:
        risk_level = "medium"
    elif risk_final < high_thresh:
        risk_level = "high"
    else:
        risk_level = "critical"

    return RiskResult(
        risk_score=risk_final,
        risk_level=risk_level,
        components={
            "robust_z": robust_z,
            "drift": drift,
            "alignment": alignment,
            "pr_window": pr_window,
            "exposure": exposure,
            "z_component": z_component,
            "drift_component": drift_component,
            "alignment_component": alignment_component,
            "pr_factor": pr_factor,
            "exposure_factor": exposure_factor,
            "risk_core": risk_core,
            "risk_with_pr": risk_with_pr,
        },
    )


def classify_risk_level(
    risk_score: float | None,
    thresholds: tuple[float, float, float] = DEFAULT_RISK_THRESHOLDS,
) -> Literal["unknown", "low", "medium", "high", "critical"]:
    """Classify a risk score into a categorical level.

    Args:
        risk_score: Risk score in [0, 1], or None
        thresholds: (low, medium, high) boundary thresholds

    Returns:
        Risk level category

    Examples:
        >>> classify_risk_level(0.1)
        'low'
        >>> classify_risk_level(0.6)
        'high'
        >>> classify_risk_level(None)
        'unknown'
    """
    if risk_score is None:
        return "unknown"

    low_thresh, med_thresh, high_thresh = thresholds
    if risk_score < low_thresh:
        return "low"
    elif risk_score < med_thresh:
        return "medium"
    elif risk_score < high_thresh:
        return "high"
    else:
        return "critical"


def aggregate_risk_scores(
    scores: list[float | None],
    method: Literal["mean", "max", "p95"] = "mean",
) -> float | None:
    """Aggregate multiple risk scores into a single value.

    Args:
        scores: List of risk scores (None values are filtered)
        method: Aggregation method (default: "mean")

    Returns:
        Aggregated risk score, or None if no valid scores

    Examples:
        >>> aggregate_risk_scores([0.2, 0.4, 0.6])
        0.4
        >>> aggregate_risk_scores([0.2, 0.4, 0.6], method="max")
        0.6
        >>> aggregate_risk_scores([None, None])
        None
    """
    # Filter None values
    valid_scores = [s for s in scores if s is not None]

    if not valid_scores:
        return None

    if method == "mean":
        return sum(valid_scores) / len(valid_scores)
    elif method == "max":
        return max(valid_scores)
    elif method == "p95":
        import numpy as np

        return float(np.percentile(valid_scores, 95))
    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def calibrate_thresholds(
    risk_scores: list[float],
    percentiles: tuple[float, float, float] = (80, 95, 99),
) -> RiskThresholds:
    """Calibrate risk thresholds from historical data.

    IMPORTANT: Do NOT use fixed thresholds in production.
    Calibrate per run/environment using this function.

    Args:
        risk_scores: List of historical risk scores
        percentiles: (p80, p95, p99) percentiles for thresholds

    Returns:
        RiskThresholds with calibrated values

    Examples:
        >>> scores = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        >>> thresholds = calibrate_thresholds(scores)
        >>> thresholds.medium
        0.82
        >>> thresholds.high
        0.955
    """
    import numpy as np

    # Filter None values
    valid_scores = [s for s in risk_scores if s is not None]

    if len(valid_scores) < 10:
        # Fallback to conservative defaults when insufficient data
        return RiskThresholds(medium=0.3, high=0.6, critical=0.85)

    p80, p95, p99 = np.percentile(valid_scores, percentiles)
    return RiskThresholds(
        medium=float(p80),
        high=float(p95),
        critical=float(p99),
    )


def assign_risk_level(
    risk_score: float | None,
    thresholds: RiskThresholds,
) -> Literal["unknown", "low", "medium", "high", "critical"]:
    """Assign risk level using calibrated thresholds.

    Args:
        risk_score: Risk score in [0, 1], or None
        thresholds: Calibrated RiskThresholds

    Returns:
        Risk level category
    """
    if risk_score is None:
        return "unknown"
    if risk_score >= thresholds.critical:
        return "critical"
    if risk_score >= thresholds.high:
        return "high"
    if risk_score >= thresholds.medium:
        return "medium"
    return "low"
