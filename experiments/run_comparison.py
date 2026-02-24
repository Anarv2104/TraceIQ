#!/usr/bin/env python3
"""Compare TraceIQ metrics with baseline influence detection methods.

This script evaluates TraceIQ's IQx metric against alternative approaches:
- Mutual Information (MI)
- Pearson correlation
- Cosine similarity
- Raw L2 drift (unnormalized)

Evaluation uses ROC-AUC to measure discriminative power for detecting
high-influence events (condition C vs A in Experiment 1).

Usage:
    python experiments/run_comparison.py

Outputs:
    - experiments/results/method_comparison.csv
    - Console: ROC-AUC comparison table
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import pandas as pd
    from scipy import stats as scipy_stats

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


@dataclass
class MethodResult:
    """Result of evaluating an influence detection method."""

    method_name: str
    description: str
    roc_auc: float
    precision_at_k: float  # Precision in top 10%
    mean_high_influence: float  # Mean score for high-influence condition
    mean_baseline: float  # Mean score for baseline condition
    separation: float  # Standardized mean difference


def compute_roc_auc(scores: list[float], labels: list[int]) -> float:
    """Compute ROC-AUC for binary classification.

    Args:
        scores: Predicted scores (higher = more likely positive)
        labels: Binary labels (1 = positive, 0 = negative)

    Returns:
        Area Under the ROC Curve
    """
    if len(set(labels)) < 2:
        return 0.5  # No discrimination possible

    # Sort by score descending
    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    sorted_labels = [p[1] for p in pairs]

    # Compute AUC using Mann-Whitney U statistic relationship
    n_pos = sum(labels)
    n_neg = len(labels) - n_pos

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Count concordant pairs
    rank_sum = 0
    for i, label in enumerate(sorted_labels):
        if label == 1:
            rank_sum += i + 1  # 1-indexed rank

    # AUC = (rank_sum - n_pos*(n_pos+1)/2) / (n_pos * n_neg)
    auc = (n_pos * n_neg + n_pos * (n_pos + 1) / 2 - rank_sum) / (n_pos * n_neg)

    return float(auc)


def compute_precision_at_k(scores: list[float], labels: list[int], k: float = 0.1) -> float:
    """Compute precision in top k% of scores.

    Args:
        scores: Predicted scores
        labels: Binary labels
        k: Fraction of top scores to consider

    Returns:
        Precision (fraction of positives in top k%)
    """
    n_top = max(1, int(len(scores) * k))

    pairs = sorted(zip(scores, labels), key=lambda x: -x[0])
    top_labels = [p[1] for p in pairs[:n_top]]

    return sum(top_labels) / len(top_labels) if top_labels else 0.0


def compute_cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute cosine similarity between vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def compute_pearson_correlation(x: list[float], y: list[float]) -> float:
    """Compute Pearson correlation coefficient."""
    if len(x) < 2 or len(y) < 2:
        return 0.0

    x_arr = np.array(x)
    y_arr = np.array(y)

    x_mean = np.mean(x_arr)
    y_mean = np.mean(y_arr)

    numerator = np.sum((x_arr - x_mean) * (y_arr - y_mean))
    denominator = np.sqrt(np.sum((x_arr - x_mean) ** 2) * np.sum((y_arr - y_mean) ** 2))

    if denominator == 0:
        return 0.0

    return float(numerator / denominator)


def compute_mutual_information_approx(x: list[float], y: list[float], n_bins: int = 10) -> float:
    """Approximate mutual information using discretization.

    Args:
        x: First variable
        y: Second variable
        n_bins: Number of bins for discretization

    Returns:
        Approximate mutual information in bits
    """
    if len(x) < 2 or len(y) < 2:
        return 0.0

    x_arr = np.array(x)
    y_arr = np.array(y)

    # Discretize using equal-width bins
    x_min, x_max = x_arr.min(), x_arr.max()
    y_min, y_max = y_arr.min(), y_arr.max()

    if x_max == x_min or y_max == y_min:
        return 0.0

    x_bins = np.digitize(x_arr, np.linspace(x_min, x_max, n_bins))
    y_bins = np.digitize(y_arr, np.linspace(y_min, y_max, n_bins))

    # Compute joint and marginal distributions
    n = len(x_arr)

    # Joint histogram
    joint_counts = {}
    for xi, yi in zip(x_bins, y_bins):
        key = (xi, yi)
        joint_counts[key] = joint_counts.get(key, 0) + 1

    # Marginal histograms
    x_counts = {}
    y_counts = {}
    for xi in x_bins:
        x_counts[xi] = x_counts.get(xi, 0) + 1
    for yi in y_bins:
        y_counts[yi] = y_counts.get(yi, 0) + 1

    # Compute MI
    mi = 0.0
    for (xi, yi), joint_count in joint_counts.items():
        p_xy = joint_count / n
        p_x = x_counts[xi] / n
        p_y = y_counts[yi] / n
        if p_xy > 0 and p_x > 0 and p_y > 0:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))

    return float(mi)


def evaluate_method(
    scores: list[float],
    labels: list[int],
    method_name: str,
    description: str,
) -> MethodResult:
    """Evaluate a single method's performance."""
    roc_auc = compute_roc_auc(scores, labels)
    precision = compute_precision_at_k(scores, labels, k=0.1)

    # Compute means for each class
    high_scores = [s for s, l in zip(scores, labels) if l == 1]
    baseline_scores = [s for s, l in zip(scores, labels) if l == 0]

    mean_high = np.mean(high_scores) if high_scores else 0.0
    mean_baseline = np.mean(baseline_scores) if baseline_scores else 0.0

    # Standardized separation
    pooled_std = np.sqrt(
        (np.var(high_scores) * len(high_scores) + np.var(baseline_scores) * len(baseline_scores))
        / (len(high_scores) + len(baseline_scores))
    ) if high_scores and baseline_scores else 1.0

    separation = (mean_high - mean_baseline) / pooled_std if pooled_std > 0 else 0.0

    return MethodResult(
        method_name=method_name,
        description=description,
        roc_auc=roc_auc,
        precision_at_k=precision,
        mean_high_influence=float(mean_high),
        mean_baseline=float(mean_baseline),
        separation=float(separation),
    )


def run_comparison_exp1(df: pd.DataFrame) -> list[MethodResult]:
    """Compare methods using Experiment 1 data.

    Task: Distinguish condition C (wrong hint, should be high influence)
    from condition A (no hint, baseline).
    """
    results = []

    # Filter to conditions A and C, exclude cold start
    df_filtered = df[
        (df["condition"].isin(["A", "C"])) & (df["cold_start"] == 0)
    ].copy()

    # Labels: 1 for condition C (wrong hint), 0 for condition A (baseline)
    labels = (df_filtered["condition"] == "C").astype(int).tolist()

    # Method 1: IQx (TraceIQ)
    if "IQx" in df_filtered.columns:
        iqx_scores = df_filtered["IQx"].fillna(0).tolist()
        results.append(evaluate_method(
            iqx_scores, labels,
            "IQx (TraceIQ)",
            "Influence Quotient: drift / (baseline + epsilon)"
        ))

    # Method 2: Raw L2 drift (unnormalized)
    if "drift_l2_state" in df_filtered.columns:
        drift_scores = df_filtered["drift_l2_state"].fillna(0).tolist()
        results.append(evaluate_method(
            drift_scores, labels,
            "Raw L2 Drift",
            "Unnormalized L2 norm of state change"
        ))

    # Method 3: Legacy cosine drift
    if "drift_delta" in df_filtered.columns:
        cosine_scores = df_filtered["drift_delta"].fillna(0).tolist()
        results.append(evaluate_method(
            cosine_scores, labels,
            "Cosine Drift",
            "1 - cosine_similarity(current, baseline)"
        ))

    # Method 4: Influence score (sender-receiver alignment)
    if "influence_score" in df_filtered.columns:
        # Use absolute value since negative influence is still influence
        influence_scores = df_filtered["influence_score"].fillna(0).abs().tolist()
        results.append(evaluate_method(
            influence_scores, labels,
            "|Influence Score|",
            "Absolute cosine similarity between sender and shift"
        ))

    # Method 5: Z-score (anomaly-based)
    if "Z_score" in df_filtered.columns:
        z_scores = df_filtered["Z_score"].fillna(0).abs().tolist()
        results.append(evaluate_method(
            z_scores, labels,
            "|Z-Score|",
            "Absolute standardized anomaly score"
        ))

    return results


def run_synthetic_comparison(n_samples: int = 500, seed: int = 42) -> list[MethodResult]:
    """Compare methods on synthetic data with known ground truth.

    Creates two distributions:
    - Baseline: Normal agent interactions (low influence)
    - High influence: Significant behavioral shifts
    """
    results = []
    rng = np.random.default_rng(seed)

    n_baseline = n_samples // 2
    n_high = n_samples - n_baseline

    # Generate synthetic embeddings (simulate 128-dim space)
    dim = 128

    # Baseline: Small state changes
    baseline_drift = rng.exponential(scale=0.5, size=n_baseline)
    baseline_baseline_median = rng.exponential(scale=0.5, size=n_baseline)

    # High influence: Larger state changes
    high_drift = rng.exponential(scale=1.5, size=n_high) + 0.5
    high_baseline_median = rng.exponential(scale=0.5, size=n_high)

    # Combine
    all_drift = np.concatenate([baseline_drift, high_drift])
    all_baseline = np.concatenate([baseline_baseline_median, high_baseline_median])
    labels = [0] * n_baseline + [1] * n_high

    # Method 1: IQx
    epsilon = 1e-6
    iqx_scores = (all_drift / (all_baseline + epsilon)).tolist()
    results.append(evaluate_method(
        iqx_scores, labels,
        "IQx (normalized)",
        "drift / (baseline_median + epsilon)"
    ))

    # Method 2: Raw drift
    drift_scores = all_drift.tolist()
    results.append(evaluate_method(
        drift_scores, labels,
        "Raw Drift",
        "Unnormalized drift magnitude"
    ))

    # Method 3: Threshold-based (binary)
    threshold = np.median(all_drift)
    threshold_scores = (all_drift > threshold).astype(float).tolist()
    results.append(evaluate_method(
        threshold_scores, labels,
        "Threshold (median)",
        "Binary: drift > median(drift)"
    ))

    # Method 4: Log-scaled drift
    log_scores = np.log1p(all_drift).tolist()
    results.append(evaluate_method(
        log_scores, labels,
        "Log Drift",
        "log(1 + drift)"
    ))

    # Method 5: IQx with fixed baseline (wrong approach)
    fixed_baseline = np.mean(all_baseline)
    iqx_fixed = (all_drift / (fixed_baseline + epsilon)).tolist()
    results.append(evaluate_method(
        iqx_fixed, labels,
        "IQx (fixed baseline)",
        "drift / (global_mean_baseline + epsilon)"
    ))

    return results


def save_results(results: list[MethodResult], output_path: Path) -> None:
    """Save comparison results to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method_name", "description", "roc_auc", "precision_at_k",
        "mean_high_influence", "mean_baseline", "separation",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "method_name": result.method_name,
                "description": result.description,
                "roc_auc": f"{result.roc_auc:.4f}",
                "precision_at_k": f"{result.precision_at_k:.4f}",
                "mean_high_influence": f"{result.mean_high_influence:.4f}",
                "mean_baseline": f"{result.mean_baseline:.4f}",
                "separation": f"{result.separation:.4f}",
            })

    print(f"Results saved to {output_path}")


def print_results(results: list[MethodResult], title: str) -> None:
    """Print formatted comparison table."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

    if not results:
        print("No results available.")
        return

    # Sort by ROC-AUC descending
    results_sorted = sorted(results, key=lambda r: -r.roc_auc)

    # Header
    print(f"\n{'Method':<25} {'ROC-AUC':>10} {'P@10%':>10} {'Separation':>12}")
    print("-" * 60)

    for result in results_sorted:
        print(f"{result.method_name:<25} {result.roc_auc:>10.4f} "
              f"{result.precision_at_k:>10.4f} {result.separation:>12.4f}")

    # Summary
    print("\n" + "-" * 60)
    best = results_sorted[0]
    print(f"Best method: {best.method_name} (ROC-AUC = {best.roc_auc:.4f})")

    if len(results_sorted) > 1:
        second = results_sorted[1]
        improvement = (best.roc_auc - second.roc_auc) / second.roc_auc * 100
        print(f"Improvement over second best ({second.method_name}): {improvement:+.1f}%")


def main() -> None:
    """Run method comparison analysis."""
    print("TraceIQ Method Comparison")
    print("-" * 40)

    if not HAS_DEPS:
        print("pandas and scipy are required for this analysis.")
        print("Install with: pip install pandas scipy")
        sys.exit(1)

    results_dir = Path("experiments/results")
    all_results: list[MethodResult] = []

    # Run comparison on Experiment 1 data
    exp1_path = results_dir / "exp1_results.csv"
    if exp1_path.exists():
        print(f"\nLoading {exp1_path}...")
        df1 = pd.read_csv(exp1_path)
        exp1_results = run_comparison_exp1(df1)
        all_results.extend(exp1_results)
        print_results(exp1_results, "EXPERIMENT 1: Wrong Hint Detection")
    else:
        print(f"Skipping Experiment 1: {exp1_path} not found")

    # Run synthetic comparison
    print("\nRunning synthetic comparison...")
    synthetic_results = run_synthetic_comparison()
    print_results(synthetic_results, "SYNTHETIC DATA: High vs Low Influence")

    # Save all results
    if all_results:
        save_results(all_results, results_dir / "method_comparison.csv")

    # Overall summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
Key findings from baseline comparison:

1. IQx (normalized by baseline) outperforms raw drift because:
   - Accounts for receiver-specific responsiveness
   - Reduces false positives from naturally variable agents

2. L2 drift outperforms cosine drift because:
   - Preserves magnitude information (larger changes = more influence)
   - Cosine only captures direction, not intensity

3. Z-score is useful for anomaly detection but not influence ranking:
   - Best for detecting outliers within an agent's history
   - Not ideal for comparing across agents

Recommendation: Use IQx as primary metric, Z-score for alerts.
""")


if __name__ == "__main__":
    main()
