"""Statistical analysis utilities for TraceIQ research experiments.

This module provides functions for statistical hypothesis testing, confidence
interval computation, and effect size calculation to support rigorous
scientific validation of TraceIQ metrics.

Functions:
    - confidence_interval: Compute mean and t-distribution CI
    - bootstrap_ci: Compute bootstrap confidence interval
    - t_test_independent: Two-sample independent t-test
    - t_test_paired: Paired t-test for matched samples
    - mann_whitney_u: Non-parametric rank test
    - effect_size_cohens_d: Cohen's d effect size
    - effect_size_glass_delta: Glass's delta (asymmetric effect size)
    - shapiro_normality: Test for normality
    - levene_variance: Test for equal variances
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np


# Try to import scipy, fall back gracefully
try:
    import scipy.stats as stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class CIResult:
    """Result of confidence interval computation."""

    mean: float
    ci_low: float
    ci_high: float
    confidence: float
    n: int


@dataclass
class TTestResult:
    """Result of a t-test."""

    t_statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    degrees_freedom: float
    mean_diff: float


@dataclass
class MannWhitneyResult:
    """Result of Mann-Whitney U test."""

    u_statistic: float
    p_value: float
    significant: bool
    confidence_level: float
    effect_size_r: float  # r = Z / sqrt(N)


@dataclass
class EffectSizeResult:
    """Result of effect size computation."""

    effect_size: float
    interpretation: Literal["negligible", "small", "medium", "large"]


@dataclass
class NormalityResult:
    """Result of normality test."""

    statistic: float
    p_value: float
    is_normal: bool  # True if p > 0.05


def _require_scipy() -> None:
    """Raise ImportError if scipy is not available."""
    if not HAS_SCIPY:
        raise ImportError(
            "scipy is required for statistical analysis. "
            "Install with: pip install scipy"
        )


def confidence_interval(
    data: list[float] | np.ndarray,
    confidence: float = 0.95,
) -> CIResult:
    """Compute mean and confidence interval using t-distribution.

    Uses Student's t-distribution for accurate small-sample inference.

    Args:
        data: Sample data
        confidence: Confidence level (default: 0.95 for 95% CI)

    Returns:
        CIResult with mean, CI bounds, and sample size

    Example:
        >>> result = confidence_interval([1.2, 1.4, 1.1, 1.5, 1.3])
        >>> print(f"Mean: {result.mean:.2f} [{result.ci_low:.2f}, {result.ci_high:.2f}]")
    """
    _require_scipy()

    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)

    if n < 2:
        mean = float(arr[0]) if n == 1 else 0.0
        return CIResult(mean=mean, ci_low=mean, ci_high=mean, confidence=confidence, n=n)

    mean = float(np.mean(arr))
    se = float(stats.sem(arr))
    h = se * stats.t.ppf((1 + confidence) / 2, n - 1)

    return CIResult(
        mean=mean,
        ci_low=mean - h,
        ci_high=mean + h,
        confidence=confidence,
        n=n,
    )


def bootstrap_ci(
    data: list[float] | np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int = 42,
) -> CIResult:
    """Compute bootstrap confidence interval.

    Non-parametric method that makes no distributional assumptions.
    Preferred for small samples or non-normal data.

    Args:
        data: Sample data
        n_bootstrap: Number of bootstrap resamples
        confidence: Confidence level
        seed: Random seed for reproducibility

    Returns:
        CIResult with mean and bootstrap percentile CI

    Example:
        >>> result = bootstrap_ci([1.2, 1.4, 1.1, 1.5, 1.3], n_bootstrap=5000)
        >>> print(f"Mean: {result.mean:.2f} [{result.ci_low:.2f}, {result.ci_high:.2f}]")
    """
    arr = np.asarray(data, dtype=np.float64)
    n = len(arr)

    if n < 2:
        mean = float(arr[0]) if n == 1 else 0.0
        return CIResult(mean=mean, ci_low=mean, ci_high=mean, confidence=confidence, n=n)

    rng = np.random.default_rng(seed)

    # Generate bootstrap samples
    boot_means = np.array([
        np.mean(rng.choice(arr, size=n, replace=True)) for _ in range(n_bootstrap)
    ])

    alpha = 1 - confidence
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100

    return CIResult(
        mean=float(np.mean(arr)),
        ci_low=float(np.percentile(boot_means, lower_percentile)),
        ci_high=float(np.percentile(boot_means, upper_percentile)),
        confidence=confidence,
        n=n,
    )


def t_test_independent(
    group1: list[float] | np.ndarray,
    group2: list[float] | np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = True,
) -> TTestResult:
    """Two-sample independent t-test.

    Tests whether the means of two independent groups differ significantly.
    Use equal_var=False for Welch's t-test when variances may differ.

    Args:
        group1: First group data
        group2: Second group data
        alpha: Significance level (default: 0.05)
        equal_var: Assume equal variances (True) or use Welch's test (False)

    Returns:
        TTestResult with test statistic, p-value, and significance

    Example:
        >>> control = [85, 88, 90, 82, 87]
        >>> treatment = [92, 95, 89, 94, 91]
        >>> result = t_test_independent(control, treatment)
        >>> print(f"p = {result.p_value:.4f}, significant: {result.significant}")
    """
    _require_scipy()

    arr1 = np.asarray(group1, dtype=np.float64)
    arr2 = np.asarray(group2, dtype=np.float64)

    t_stat, p_value = stats.ttest_ind(arr1, arr2, equal_var=equal_var)

    # Compute degrees of freedom
    n1, n2 = len(arr1), len(arr2)
    if equal_var:
        df = n1 + n2 - 2
    else:
        # Welch-Satterthwaite approximation
        v1 = np.var(arr1, ddof=1)
        v2 = np.var(arr2, ddof=1)
        num = (v1 / n1 + v2 / n2) ** 2
        denom = (v1 / n1) ** 2 / (n1 - 1) + (v2 / n2) ** 2 / (n2 - 1)
        df = num / denom if denom > 0 else n1 + n2 - 2

    return TTestResult(
        t_statistic=float(t_stat),
        p_value=float(p_value),
        significant=p_value < alpha,
        confidence_level=1 - alpha,
        degrees_freedom=float(df),
        mean_diff=float(np.mean(arr1) - np.mean(arr2)),
    )


def t_test_paired(
    before: list[float] | np.ndarray,
    after: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> TTestResult:
    """Paired t-test for matched samples.

    Tests whether the mean difference between paired observations is zero.
    Use when observations are naturally paired (e.g., before/after).

    Args:
        before: Observations before treatment
        after: Observations after treatment
        alpha: Significance level

    Returns:
        TTestResult with test statistic and significance
    """
    _require_scipy()

    arr1 = np.asarray(before, dtype=np.float64)
    arr2 = np.asarray(after, dtype=np.float64)

    if len(arr1) != len(arr2):
        raise ValueError("Paired samples must have equal length")

    t_stat, p_value = stats.ttest_rel(arr1, arr2)

    return TTestResult(
        t_statistic=float(t_stat),
        p_value=float(p_value),
        significant=p_value < alpha,
        confidence_level=1 - alpha,
        degrees_freedom=float(len(arr1) - 1),
        mean_diff=float(np.mean(arr1 - arr2)),
    )


def mann_whitney_u(
    group1: list[float] | np.ndarray,
    group2: list[float] | np.ndarray,
    alpha: float = 0.05,
    alternative: Literal["two-sided", "less", "greater"] = "two-sided",
) -> MannWhitneyResult:
    """Mann-Whitney U test (non-parametric).

    Tests whether the distributions of two groups differ. Does not assume
    normality. Preferred for ordinal data or when normality is violated.

    Args:
        group1: First group data
        group2: Second group data
        alpha: Significance level
        alternative: Type of alternative hypothesis

    Returns:
        MannWhitneyResult with U statistic, p-value, and effect size r

    Example:
        >>> iqx_baseline = [0.8, 0.9, 1.0, 0.85, 0.95]
        >>> iqx_wrong_hint = [1.5, 1.8, 2.0, 1.6, 1.7]
        >>> result = mann_whitney_u(iqx_baseline, iqx_wrong_hint)
        >>> print(f"U = {result.u_statistic}, p = {result.p_value:.4f}")
    """
    _require_scipy()

    arr1 = np.asarray(group1, dtype=np.float64)
    arr2 = np.asarray(group2, dtype=np.float64)

    u_stat, p_value = stats.mannwhitneyu(arr1, arr2, alternative=alternative)

    # Compute effect size r = Z / sqrt(N)
    n1, n2 = len(arr1), len(arr2)
    n_total = n1 + n2

    # Convert U to Z using normal approximation
    mean_u = n1 * n2 / 2
    std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    z = (u_stat - mean_u) / std_u if std_u > 0 else 0
    effect_r = abs(z) / np.sqrt(n_total)

    return MannWhitneyResult(
        u_statistic=float(u_stat),
        p_value=float(p_value),
        significant=p_value < alpha,
        confidence_level=1 - alpha,
        effect_size_r=float(effect_r),
    )


def effect_size_cohens_d(
    group1: list[float] | np.ndarray,
    group2: list[float] | np.ndarray,
) -> EffectSizeResult:
    """Compute Cohen's d effect size.

    Measures standardized difference between two group means.
    Uses pooled standard deviation.

    Interpretation (Cohen, 1988):
        - |d| < 0.2: negligible
        - 0.2 <= |d| < 0.5: small
        - 0.5 <= |d| < 0.8: medium
        - |d| >= 0.8: large

    Args:
        group1: First group data
        group2: Second group data

    Returns:
        EffectSizeResult with d value and interpretation
    """
    arr1 = np.asarray(group1, dtype=np.float64)
    arr2 = np.asarray(group2, dtype=np.float64)

    n1, n2 = len(arr1), len(arr2)
    var1 = np.var(arr1, ddof=1)
    var2 = np.var(arr2, ddof=1)

    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        d = 0.0
    else:
        d = (np.mean(arr1) - np.mean(arr2)) / pooled_std

    # Interpret effect size
    abs_d = abs(d)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSizeResult(effect_size=float(d), interpretation=interpretation)


def effect_size_glass_delta(
    treatment: list[float] | np.ndarray,
    control: list[float] | np.ndarray,
) -> EffectSizeResult:
    """Compute Glass's delta effect size.

    Uses control group's standard deviation as denominator.
    Preferred when treatment affects variance.

    Args:
        treatment: Treatment group data
        control: Control group data

    Returns:
        EffectSizeResult with delta value and interpretation
    """
    arr_treatment = np.asarray(treatment, dtype=np.float64)
    arr_control = np.asarray(control, dtype=np.float64)

    control_std = np.std(arr_control, ddof=1)

    if control_std == 0:
        delta = 0.0
    else:
        delta = (np.mean(arr_treatment) - np.mean(arr_control)) / control_std

    # Use same interpretation thresholds as Cohen's d
    abs_d = abs(delta)
    if abs_d < 0.2:
        interpretation = "negligible"
    elif abs_d < 0.5:
        interpretation = "small"
    elif abs_d < 0.8:
        interpretation = "medium"
    else:
        interpretation = "large"

    return EffectSizeResult(effect_size=float(delta), interpretation=interpretation)


def shapiro_normality(
    data: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> NormalityResult:
    """Shapiro-Wilk test for normality.

    Tests whether data comes from a normal distribution.
    Sample size should be 3 <= n <= 5000 for reliable results.

    Args:
        data: Sample data
        alpha: Significance level

    Returns:
        NormalityResult with test statistic and normality assessment

    Note:
        is_normal=True means we FAIL to reject normality (p > alpha),
        not that the data is definitely normal.
    """
    _require_scipy()

    arr = np.asarray(data, dtype=np.float64)

    if len(arr) < 3:
        return NormalityResult(statistic=0.0, p_value=1.0, is_normal=True)

    if len(arr) > 5000:
        # Shapiro-Wilk is not recommended for n > 5000
        # Use first 5000 samples
        arr = arr[:5000]

    stat, p_value = stats.shapiro(arr)

    return NormalityResult(
        statistic=float(stat),
        p_value=float(p_value),
        is_normal=p_value > alpha,
    )


def levene_variance(
    *groups: list[float] | np.ndarray,
    alpha: float = 0.05,
) -> TTestResult:
    """Levene's test for equality of variances.

    Tests whether multiple groups have equal variances.
    Less sensitive to non-normality than Bartlett's test.

    Args:
        *groups: Two or more groups of data
        alpha: Significance level

    Returns:
        TTestResult-like object with test results

    Note:
        significant=True means variances are NOT equal (reject H0).
    """
    _require_scipy()

    arrays = [np.asarray(g, dtype=np.float64) for g in groups]

    stat, p_value = stats.levene(*arrays)

    return TTestResult(
        t_statistic=float(stat),  # Actually F-statistic
        p_value=float(p_value),
        significant=p_value < alpha,  # Variances are NOT equal
        confidence_level=1 - alpha,
        degrees_freedom=float(len(groups) - 1),
        mean_diff=0.0,  # Not applicable
    )


def format_p_value(p: float) -> str:
    """Format p-value for display.

    Args:
        p: P-value

    Returns:
        Formatted string (e.g., "p < 0.001", "p = 0.023")
    """
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def format_ci(result: CIResult) -> str:
    """Format confidence interval for display.

    Args:
        result: CIResult object

    Returns:
        Formatted string (e.g., "1.23 [1.10, 1.36]")
    """
    return f"{result.mean:.3f} [{result.ci_low:.3f}, {result.ci_high:.3f}]"
