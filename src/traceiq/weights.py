"""Bounded weight transformations for risk scoring.

This module provides functions to transform and bound weights used in
risk scoring and adjacency matrix construction. All transformations
ensure outputs are bounded to prevent numerical instability.
"""

from __future__ import annotations

import math
from typing import Literal


def clamp_weight(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a value to a bounded range.

    Args:
        x: Input value to clamp
        lo: Lower bound (default: 0.0)
        hi: Upper bound (default: 1.0)

    Returns:
        Clamped value in [lo, hi]

    Examples:
        >>> clamp_weight(1.5)
        1.0
        >>> clamp_weight(-0.5)
        0.0
        >>> clamp_weight(0.5)
        0.5
        >>> clamp_weight(50, lo=0, hi=25)
        25
    """
    return max(lo, min(x, hi))


def sigmoid_weight(
    x: float,
    center: float = 1.0,
    scale: float = 0.5,
) -> float:
    """Transform value to (0, 1) using sigmoid function.

    The sigmoid function provides a smooth, bounded transformation that:
    - Approaches 0 for large negative inputs
    - Approaches 1 for large positive inputs
    - Equals 0.5 at the center point

    Args:
        x: Input value to transform
        center: Center point where sigmoid = 0.5 (default: 1.0)
        scale: Controls steepness of transition (default: 0.5)

    Returns:
        Transformed value in (0, 1)

    Examples:
        >>> sigmoid_weight(1.0, center=1.0)  # At center
        0.5
        >>> sigmoid_weight(3.0, center=1.0, scale=0.5)  # Above center
        0.9820...
        >>> sigmoid_weight(-1.0, center=1.0, scale=0.5)  # Below center
        0.0179...
    """
    # Prevent overflow for extreme values
    z = (x - center) / scale
    if z > 500:
        return 1.0
    if z < -500:
        return 0.0
    return 1.0 / (1.0 + math.exp(-z))


def confidence_weight(quality: Literal["low", "medium", "high"]) -> float:
    """Convert state quality to a confidence weight.

    This weight is used to scale metrics based on how much state
    information is available. Higher quality state tracking yields
    more confident (higher weight) metrics.

    Args:
        quality: State quality level

    Returns:
        Confidence weight in [0.5, 1.0]

    Examples:
        >>> confidence_weight("high")
        1.0
        >>> confidence_weight("medium")
        0.8
        >>> confidence_weight("low")
        0.5
    """
    weights = {
        "low": 0.5,
        "medium": 0.8,
        "high": 1.0,
    }
    return weights[quality]


def bounded_iqx_weight(
    iqx: float,
    cap: float = 25.0,
    sigmoid_center: float = 2.0,
    sigmoid_scale: float = 1.0,
) -> float:
    """Transform IQx to a bounded [0, 1] weight for graph edges.

    This function applies multiple safeguards:
    1. Caps extreme IQx values
    2. Applies sigmoid transformation for smooth bounding
    3. Ensures output is strictly in [0, 1]

    Args:
        iqx: Input IQx value (can be any non-negative float)
        cap: Maximum IQx value before capping (default: 25.0)
        sigmoid_center: Sigmoid center point (default: 2.0)
        sigmoid_scale: Sigmoid scale parameter (default: 1.0)

    Returns:
        Bounded weight in [0, 1]

    Examples:
        >>> bounded_iqx_weight(0.0)  # No influence
        0.1192...
        >>> bounded_iqx_weight(2.0)  # Moderate influence
        0.5
        >>> bounded_iqx_weight(100.0)  # Extreme (capped)
        0.9999...
    """
    # Cap extreme values
    capped = min(iqx, cap)
    # Apply sigmoid transformation
    return sigmoid_weight(capped, center=sigmoid_center, scale=sigmoid_scale)


def alignment_weight(
    alignment: float,
    floor: float = 0.0,
) -> float:
    """Convert alignment score to a non-negative weight.

    Alignment scores can be negative (counter-alignment). This function
    ensures the output is non-negative for use in risk calculations.

    Args:
        alignment: Alignment score (typically in [-1, 1])
        floor: Minimum output value (default: 0.0)

    Returns:
        Non-negative weight

    Examples:
        >>> alignment_weight(0.8)
        0.8
        >>> alignment_weight(-0.5)
        0.0
        >>> alignment_weight(-0.5, floor=0.1)
        0.1
    """
    return max(floor, alignment)


def combine_weights(
    *weights: float,
    method: Literal["product", "mean", "min", "max"] = "product",
) -> float:
    """Combine multiple weights into a single value.

    Args:
        *weights: Variable number of weight values
        method: Combination method (default: "product")

    Returns:
        Combined weight value

    Examples:
        >>> combine_weights(0.5, 0.8, method="product")
        0.4
        >>> combine_weights(0.5, 0.8, method="mean")
        0.65
        >>> combine_weights(0.5, 0.8, method="min")
        0.5
    """
    if not weights:
        return 0.0

    if method == "product":
        result = 1.0
        for w in weights:
            result *= w
        return result
    elif method == "mean":
        return sum(weights) / len(weights)
    elif method == "min":
        return min(weights)
    elif method == "max":
        return max(weights)
    else:
        raise ValueError(f"Unknown combination method: {method}")
