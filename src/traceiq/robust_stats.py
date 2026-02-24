"""Robust statistical functions for anomaly detection.

This module re-exports and extends the MAD-based (Median Absolute Deviation)
functions from metrics.py. These functions are robust to outliers, making
them suitable for anomaly detection in adversarial settings.
"""

from __future__ import annotations

# Re-export from metrics module
from traceiq.metrics import (
    MAD_CONSTANT,
    compute_z_score_robust,
    rolling_mad,
    rolling_mean,
    rolling_median,
    rolling_std,
)

__all__ = [
    "MAD_CONSTANT",
    "rolling_mad",
    "rolling_mean",
    "rolling_median",
    "rolling_std",
    "compute_z_score_robust",
    "robust_z_score",
    "is_anomaly_robust",
]


def robust_z_score(
    x: float,
    window: list[float],
    eps: float = 1e-6,
) -> float:
    """Compute robust Z-score using MAD (alias for compute_z_score_robust).

    This function provides a cleaner name for the robust Z-score computation
    and is the preferred interface for new code.

    The robust Z-score uses Median Absolute Deviation (MAD) instead of
    standard deviation, making it resistant to outliers that could
    otherwise inflate the variance and mask true anomalies.

    Formula: Z_robust = MAD_CONSTANT * (x - median) / (MAD + eps)

    Where MAD_CONSTANT = 0.6745 makes MAD consistent with standard deviation
    for normally distributed data.

    Args:
        x: Current value to compute Z-score for
        window: Historical values to compute statistics from
        eps: Small constant for numerical stability (default: 1e-6)

    Returns:
        Robust Z-score (number of MAD-scaled deviations from median)

    Examples:
        >>> values = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> robust_z_score(3.0, values)  # At median
        0.0
        >>> robust_z_score(10.0, values)  # Anomaly
        4.7215...

    See Also:
        compute_z_score_robust: The underlying implementation
        is_anomaly_robust: Convenience function for anomaly detection
    """
    return compute_z_score_robust(x, window, eps)


def is_anomaly_robust(
    x: float,
    window: list[float],
    threshold: float = 2.0,
    eps: float = 1e-6,
) -> bool:
    """Check if a value is an anomaly using robust statistics.

    This is a convenience function that combines Z-score computation
    with threshold checking.

    Args:
        x: Value to check
        window: Historical values
        threshold: Z-score threshold for anomaly detection (default: 2.0)
        eps: Numerical stability constant (default: 1e-6)

    Returns:
        True if |Z| > threshold, False otherwise

    Examples:
        >>> values = [1.0, 2.0, 3.0, 4.0, 5.0]
        >>> is_anomaly_robust(3.0, values)  # Normal
        False
        >>> is_anomaly_robust(10.0, values)  # Anomaly
        True
    """
    # Need at least 2 values for meaningful statistics
    if len(window) < 2:
        return False

    z = robust_z_score(x, window, eps)
    return abs(z) > threshold
