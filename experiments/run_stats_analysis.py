#!/usr/bin/env python3
"""Statistical analysis of TraceIQ experiment results.

This script performs rigorous statistical analysis on experiment results:
- Confidence intervals for all metrics
- t-tests and Mann-Whitney U tests for group comparisons
- Effect sizes (Cohen's d, Glass's delta)
- Normality tests to validate test assumptions

Usage:
    python experiments/run_stats_analysis.py

Outputs:
    - experiments/results/stats_summary.csv
    - Console: Formatted statistical report
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Import our stats module
from stats import (
    bootstrap_ci,
    confidence_interval,
    effect_size_cohens_d,
    format_ci,
    format_p_value,
    mann_whitney_u,
    shapiro_normality,
    t_test_independent,
)

try:
    import pandas as pd

    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False


@dataclass
class ComparisonResult:
    """Result of a statistical comparison between conditions."""

    metric: str
    condition_1: str
    condition_2: str
    n1: int
    n2: int
    mean_1: float
    mean_2: float
    ci_1: str
    ci_2: str
    test_type: str
    statistic: float
    p_value: float
    significant: bool
    effect_size: float
    effect_interpretation: str


def load_experiment_data(csv_path: Path) -> pd.DataFrame | None:
    """Load experiment results from CSV."""
    if not HAS_PANDAS:
        print("pandas required for statistical analysis")
        print("Install with: pip install pandas")
        return None

    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return None

    return pd.read_csv(csv_path)


def analyze_exp1(df: pd.DataFrame) -> list[ComparisonResult]:
    """Analyze Experiment 1: Wrong Hint Infection.

    Comparisons:
    - Accuracy: A vs C (baseline vs wrong hint)
    - IQx: A vs C, B vs C
    - Alert rate: A vs C
    """
    results = []

    # Filter out cold start for IQx analysis
    df_iqx = df[df["cold_start"] == 0].dropna(subset=["IQx"])

    # 1. Accuracy comparison: A vs C
    acc_A = df[df["condition"] == "A"]["correct"].astype(float).tolist()
    acc_C = df[df["condition"] == "C"]["correct"].astype(float).tolist()

    if acc_A and acc_C:
        # Check normality
        norm_A = shapiro_normality(acc_A)
        norm_C = shapiro_normality(acc_C)

        # Use t-test (binary data approximates normal with large n)
        t_result = t_test_independent(acc_A, acc_C)
        d_result = effect_size_cohens_d(acc_A, acc_C)
        ci_A = confidence_interval(acc_A)
        ci_C = confidence_interval(acc_C)

        results.append(ComparisonResult(
            metric="accuracy",
            condition_1="A (No Hint)",
            condition_2="C (Wrong Hint)",
            n1=len(acc_A),
            n2=len(acc_C),
            mean_1=ci_A.mean,
            mean_2=ci_C.mean,
            ci_1=format_ci(ci_A),
            ci_2=format_ci(ci_C),
            test_type="t-test" if norm_A.is_normal and norm_C.is_normal else "t-test*",
            statistic=t_result.t_statistic,
            p_value=t_result.p_value,
            significant=t_result.significant,
            effect_size=d_result.effect_size,
            effect_interpretation=d_result.interpretation,
        ))

    # 2. IQx comparison: A vs C (using Mann-Whitney as IQx may not be normal)
    iqx_A = df_iqx[df_iqx["condition"] == "A"]["IQx"].tolist()
    iqx_C = df_iqx[df_iqx["condition"] == "C"]["IQx"].tolist()

    if iqx_A and iqx_C:
        mw_result = mann_whitney_u(iqx_A, iqx_C)
        d_result = effect_size_cohens_d(iqx_A, iqx_C)
        ci_A = bootstrap_ci(iqx_A)
        ci_C = bootstrap_ci(iqx_C)

        results.append(ComparisonResult(
            metric="IQx",
            condition_1="A (No Hint)",
            condition_2="C (Wrong Hint)",
            n1=len(iqx_A),
            n2=len(iqx_C),
            mean_1=ci_A.mean,
            mean_2=ci_C.mean,
            ci_1=format_ci(ci_A),
            ci_2=format_ci(ci_C),
            test_type="Mann-Whitney U",
            statistic=mw_result.u_statistic,
            p_value=mw_result.p_value,
            significant=mw_result.significant,
            effect_size=d_result.effect_size,
            effect_interpretation=d_result.interpretation,
        ))

    # 3. IQx comparison: B vs C
    iqx_B = df_iqx[df_iqx["condition"] == "B"]["IQx"].tolist()

    if iqx_B and iqx_C:
        mw_result = mann_whitney_u(iqx_B, iqx_C)
        d_result = effect_size_cohens_d(iqx_B, iqx_C)
        ci_B = bootstrap_ci(iqx_B)
        ci_C = bootstrap_ci(iqx_C)

        results.append(ComparisonResult(
            metric="IQx",
            condition_1="B (Correct Hint)",
            condition_2="C (Wrong Hint)",
            n1=len(iqx_B),
            n2=len(iqx_C),
            mean_1=ci_B.mean,
            mean_2=ci_C.mean,
            ci_1=format_ci(ci_B),
            ci_2=format_ci(ci_C),
            test_type="Mann-Whitney U",
            statistic=mw_result.u_statistic,
            p_value=mw_result.p_value,
            significant=mw_result.significant,
            effect_size=d_result.effect_size,
            effect_interpretation=d_result.interpretation,
        ))

    # 4. Alert rate comparison: A vs C
    alert_A = df[df["condition"] == "A"]["alert"].astype(float).tolist()
    alert_C = df[df["condition"] == "C"]["alert"].astype(float).tolist()

    if alert_A and alert_C:
        t_result = t_test_independent(alert_A, alert_C)
        d_result = effect_size_cohens_d(alert_A, alert_C)
        ci_A = confidence_interval(alert_A)
        ci_C = confidence_interval(alert_C)

        results.append(ComparisonResult(
            metric="alert_rate",
            condition_1="A (No Hint)",
            condition_2="C (Wrong Hint)",
            n1=len(alert_A),
            n2=len(alert_C),
            mean_1=ci_A.mean,
            mean_2=ci_C.mean,
            ci_1=format_ci(ci_A),
            ci_2=format_ci(ci_C),
            test_type="t-test",
            statistic=t_result.t_statistic,
            p_value=t_result.p_value,
            significant=t_result.significant,
            effect_size=d_result.effect_size,
            effect_interpretation=d_result.interpretation,
        ))

    return results


def analyze_exp2(df: pd.DataFrame) -> list[ComparisonResult]:
    """Analyze Experiment 2: Propagation patterns.

    Comparisons:
    - IQx by sender agent
    - Accumulated influence per agent
    """
    results = []

    df_iqx = df.dropna(subset=["IQx"])

    # Compare IQx between agent_A (injector) and others
    iqx_agent_A = df_iqx[df_iqx["sender"] == "agent_A"]["IQx"].tolist()
    iqx_others = df_iqx[df_iqx["sender"] != "agent_A"]["IQx"].tolist()

    if iqx_agent_A and iqx_others:
        mw_result = mann_whitney_u(iqx_agent_A, iqx_others)
        d_result = effect_size_cohens_d(iqx_agent_A, iqx_others)
        ci_A = bootstrap_ci(iqx_agent_A)
        ci_others = bootstrap_ci(iqx_others)

        results.append(ComparisonResult(
            metric="IQx",
            condition_1="agent_A (injector)",
            condition_2="other agents",
            n1=len(iqx_agent_A),
            n2=len(iqx_others),
            mean_1=ci_A.mean,
            mean_2=ci_others.mean,
            ci_1=format_ci(ci_A),
            ci_2=format_ci(ci_others),
            test_type="Mann-Whitney U",
            statistic=mw_result.u_statistic,
            p_value=mw_result.p_value,
            significant=mw_result.significant,
            effect_size=d_result.effect_size,
            effect_interpretation=d_result.interpretation,
        ))

    return results


def analyze_exp3(df: pd.DataFrame) -> list[ComparisonResult]:
    """Analyze Experiment 3: Mitigation effectiveness.

    Comparisons:
    - Accuracy with vs without mitigation (for condition C)
    - IQx with vs without mitigation
    """
    results = []

    # Filter for condition C (wrong hint)
    df_C = df[df["condition"] == "C"]

    # Accuracy: without vs with mitigation
    acc_without = df_C[df_C["mitigation_enabled"] == 0]["correct"].astype(float).tolist()
    acc_with = df_C[df_C["mitigation_enabled"] == 1]["correct"].astype(float).tolist()

    if acc_without and acc_with:
        t_result = t_test_independent(acc_without, acc_with)
        d_result = effect_size_cohens_d(acc_with, acc_without)  # Improvement direction
        ci_without = confidence_interval(acc_without)
        ci_with = confidence_interval(acc_with)

        results.append(ComparisonResult(
            metric="accuracy_condition_C",
            condition_1="Without Mitigation",
            condition_2="With Mitigation",
            n1=len(acc_without),
            n2=len(acc_with),
            mean_1=ci_without.mean,
            mean_2=ci_with.mean,
            ci_1=format_ci(ci_without),
            ci_2=format_ci(ci_with),
            test_type="t-test",
            statistic=t_result.t_statistic,
            p_value=t_result.p_value,
            significant=t_result.significant,
            effect_size=d_result.effect_size,
            effect_interpretation=d_result.interpretation,
        ))

    # IQx: without vs with mitigation
    df_iqx = df_C[df_C["cold_start"] == 0].dropna(subset=["IQx"])
    iqx_without = df_iqx[df_iqx["mitigation_enabled"] == 0]["IQx"].tolist()
    iqx_with = df_iqx[df_iqx["mitigation_enabled"] == 1]["IQx"].tolist()

    if iqx_without and iqx_with:
        mw_result = mann_whitney_u(iqx_without, iqx_with)
        d_result = effect_size_cohens_d(iqx_without, iqx_with)
        ci_without = bootstrap_ci(iqx_without)
        ci_with = bootstrap_ci(iqx_with)

        results.append(ComparisonResult(
            metric="IQx_condition_C",
            condition_1="Without Mitigation",
            condition_2="With Mitigation",
            n1=len(iqx_without),
            n2=len(iqx_with),
            mean_1=ci_without.mean,
            mean_2=ci_with.mean,
            ci_1=format_ci(ci_without),
            ci_2=format_ci(ci_with),
            test_type="Mann-Whitney U",
            statistic=mw_result.u_statistic,
            p_value=mw_result.p_value,
            significant=mw_result.significant,
            effect_size=d_result.effect_size,
            effect_interpretation=d_result.interpretation,
        ))

    return results


def generate_descriptive_stats(df: pd.DataFrame, experiment: str) -> list[dict]:
    """Generate descriptive statistics for all conditions."""
    stats_rows = []

    if "condition" in df.columns:
        conditions = df["condition"].unique()
    else:
        conditions = ["all"]

    for condition in conditions:
        if condition == "all":
            subset = df
        else:
            subset = df[df["condition"] == condition]

        # Basic counts
        n = len(subset)

        # Accuracy (if available)
        if "correct" in subset.columns:
            acc_mean, acc_std = subset["correct"].mean(), subset["correct"].std()
        else:
            acc_mean, acc_std = None, None

        # IQx (excluding cold start)
        if "IQx" in subset.columns and "cold_start" in subset.columns:
            iqx_data = subset[subset["cold_start"] == 0]["IQx"].dropna()
            if len(iqx_data) > 0:
                iqx_mean, iqx_std = iqx_data.mean(), iqx_data.std()
                iqx_median = iqx_data.median()
            else:
                iqx_mean, iqx_std, iqx_median = None, None, None
        else:
            iqx_mean, iqx_std, iqx_median = None, None, None

        # Alerts
        if "alert" in subset.columns:
            alert_rate = subset["alert"].mean()
        else:
            alert_rate = None

        stats_rows.append({
            "experiment": experiment,
            "condition": condition,
            "n": n,
            "accuracy_mean": acc_mean,
            "accuracy_std": acc_std,
            "iqx_mean": iqx_mean,
            "iqx_std": iqx_std,
            "iqx_median": iqx_median,
            "alert_rate": alert_rate,
        })

    return stats_rows


def save_results(
    comparisons: list[ComparisonResult],
    descriptive: list[dict],
    output_dir: Path,
) -> None:
    """Save statistical results to CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save comparison results
    comparison_path = output_dir / "stats_summary.csv"
    with open(comparison_path, "w", newline="") as f:
        fieldnames = [
            "metric", "condition_1", "condition_2", "n1", "n2",
            "mean_1", "mean_2", "ci_1", "ci_2",
            "test_type", "statistic", "p_value", "significant",
            "effect_size", "effect_interpretation",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in comparisons:
            writer.writerow({
                "metric": result.metric,
                "condition_1": result.condition_1,
                "condition_2": result.condition_2,
                "n1": result.n1,
                "n2": result.n2,
                "mean_1": f"{result.mean_1:.4f}",
                "mean_2": f"{result.mean_2:.4f}",
                "ci_1": result.ci_1,
                "ci_2": result.ci_2,
                "test_type": result.test_type,
                "statistic": f"{result.statistic:.4f}",
                "p_value": f"{result.p_value:.6f}",
                "significant": result.significant,
                "effect_size": f"{result.effect_size:.4f}",
                "effect_interpretation": result.effect_interpretation,
            })
    print(f"Saved comparison results to {comparison_path}")

    # Save descriptive statistics
    desc_path = output_dir / "descriptive_stats.csv"
    with open(desc_path, "w", newline="") as f:
        fieldnames = list(descriptive[0].keys()) if descriptive else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(descriptive)
    print(f"Saved descriptive statistics to {desc_path}")


def print_results(comparisons: list[ComparisonResult]) -> None:
    """Print formatted statistical report."""
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS RESULTS")
    print("=" * 80)

    if not comparisons:
        print("No comparison results available.")
        return

    # Group by metric
    metrics = sorted(set(r.metric for r in comparisons))

    for metric in metrics:
        metric_results = [r for r in comparisons if r.metric == metric]

        print(f"\n{metric.upper()}")
        print("-" * 60)

        for result in metric_results:
            sig_marker = "*" if result.significant else ""
            print(f"\n  {result.condition_1} vs {result.condition_2}")
            print(f"    Group 1: {result.ci_1} (n={result.n1})")
            print(f"    Group 2: {result.ci_2} (n={result.n2})")
            print(f"    {result.test_type}: statistic={result.statistic:.4f}, {format_p_value(result.p_value)}{sig_marker}")
            print(f"    Effect size (Cohen's d): {result.effect_size:.3f} ({result.effect_interpretation})")

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    significant = sum(1 for r in comparisons if r.significant)
    total = len(comparisons)
    print(f"Significant comparisons: {significant}/{total}")
    print(f"* p < 0.05")

    # Key findings
    print("\nKEY FINDINGS:")
    for result in comparisons:
        if result.significant:
            direction = ">" if result.mean_1 > result.mean_2 else "<"
            print(f"  - {result.metric}: {result.condition_1} {direction} {result.condition_2} "
                  f"({format_p_value(result.p_value)}, d={result.effect_size:.2f})")


def main() -> None:
    """Run statistical analysis on experiment results."""
    print("Statistical Analysis of TraceIQ Experiments")
    print("-" * 40)

    if not HAS_PANDAS:
        print("pandas is required for this analysis.")
        print("Install with: pip install pandas")
        sys.exit(1)

    results_dir = Path("experiments/results")
    all_comparisons: list[ComparisonResult] = []
    all_descriptive: list[dict] = []

    # Analyze Experiment 1
    exp1_path = results_dir / "exp1_results.csv"
    if exp1_path.exists():
        print(f"\nAnalyzing {exp1_path}...")
        df1 = load_experiment_data(exp1_path)
        if df1 is not None:
            comparisons = analyze_exp1(df1)
            all_comparisons.extend(comparisons)
            all_descriptive.extend(generate_descriptive_stats(df1, "exp1"))
            print(f"  Found {len(comparisons)} comparisons")
    else:
        print(f"Skipping Experiment 1: {exp1_path} not found")

    # Analyze Experiment 2
    exp2_path = results_dir / "exp2_results.csv"
    if exp2_path.exists():
        print(f"\nAnalyzing {exp2_path}...")
        df2 = load_experiment_data(exp2_path)
        if df2 is not None:
            comparisons = analyze_exp2(df2)
            all_comparisons.extend(comparisons)
            all_descriptive.extend(generate_descriptive_stats(df2, "exp2"))
            print(f"  Found {len(comparisons)} comparisons")
    else:
        print(f"Skipping Experiment 2: {exp2_path} not found")

    # Analyze Experiment 3
    exp3_path = results_dir / "exp3_results.csv"
    if exp3_path.exists():
        print(f"\nAnalyzing {exp3_path}...")
        df3 = load_experiment_data(exp3_path)
        if df3 is not None:
            comparisons = analyze_exp3(df3)
            all_comparisons.extend(comparisons)
            all_descriptive.extend(generate_descriptive_stats(df3, "exp3"))
            print(f"  Found {len(comparisons)} comparisons")
    else:
        print(f"Skipping Experiment 3: {exp3_path} not found")

    if not all_comparisons:
        print("\nNo experiment data found. Run experiments first:")
        print("  python experiments/run_exp1_wrong_hint.py")
        print("  python experiments/run_exp2_propagation.py")
        print("  python experiments/run_exp3_mitigation.py")
        sys.exit(1)

    # Save results
    save_results(all_comparisons, all_descriptive, results_dir)

    # Print results
    print_results(all_comparisons)


if __name__ == "__main__":
    main()
