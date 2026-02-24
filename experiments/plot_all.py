#!/usr/bin/env python3
"""Generate all plots for TraceIQ experiments with statistical annotations.

This script reads CSV results from experiments and generates visualizations
with confidence intervals, p-values, and effect sizes for scientific rigor.

Outputs:
- experiments/plots/exp1_accuracy.png
- experiments/plots/exp1_accuracy_stats.png (with error bars and p-values)
- experiments/plots/exp1_iqx_box.png
- experiments/plots/exp1_iqx_stats.png (with statistics)
- experiments/plots/exp1_alert_rate.png
- experiments/plots/exp2_propagation_risk.png
- experiments/plots/exp2_agent_influence.png
- experiments/plots/exp3_mitigation_compare.png
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
    import pandas as pd

    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: matplotlib and pandas required for plotting")
    print("Install with: pip install matplotlib pandas")

# Try to import scipy for statistics
try:
    import scipy.stats as scipy_stats

    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def compute_ci(data: np.ndarray, confidence: float = 0.95) -> tuple[float, float, float]:
    """Compute mean and confidence interval.

    Returns:
        Tuple of (mean, ci_low, ci_high)
    """
    n = len(data)
    if n < 2:
        mean = float(data[0]) if n == 1 else 0.0
        return mean, mean, mean

    mean = np.mean(data)

    if HAS_SCIPY:
        se = scipy_stats.sem(data)
        h = se * scipy_stats.t.ppf((1 + confidence) / 2, n - 1)
    else:
        se = np.std(data, ddof=1) / np.sqrt(n)
        # Approximate t-value for 95% CI
        h = se * 1.96

    return float(mean), float(mean - h), float(mean + h)


def compute_t_test(group1: np.ndarray, group2: np.ndarray) -> tuple[float, float]:
    """Compute two-sample t-test.

    Returns:
        Tuple of (t_statistic, p_value)
    """
    if not HAS_SCIPY:
        return 0.0, 1.0

    t_stat, p_value = scipy_stats.ttest_ind(group1, group2)
    return float(t_stat), float(p_value)


def compute_mann_whitney(group1: np.ndarray, group2: np.ndarray) -> tuple[float, float]:
    """Compute Mann-Whitney U test.

    Returns:
        Tuple of (u_statistic, p_value)
    """
    if not HAS_SCIPY:
        return 0.0, 1.0

    u_stat, p_value = scipy_stats.mannwhitneyu(group1, group2, alternative='two-sided')
    return float(u_stat), float(p_value)


def compute_cohens_d(group1: np.ndarray, group2: np.ndarray) -> float:
    """Compute Cohen's d effect size."""
    n1, n2 = len(group1), len(group2)
    var1 = np.var(group1, ddof=1)
    var2 = np.var(group2, ddof=1)

    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(group1) - np.mean(group2)) / pooled_std)


def format_p_value(p: float) -> str:
    """Format p-value for display."""
    if p < 0.001:
        return "p < 0.001"
    elif p < 0.01:
        return f"p = {p:.3f}"
    else:
        return f"p = {p:.2f}"


def significance_stars(p: float) -> str:
    """Return significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    else:
        return "ns"


def plot_exp1_accuracy(df: pd.DataFrame, output_path: Path) -> None:
    """Plot accuracy by condition for Experiment 1.

    Args:
        df: Experiment 1 results DataFrame
        output_path: Path to save plot
    """
    accuracy = df.groupby("condition")["correct"].mean() * 100

    fig, ax = plt.subplots(figsize=(8, 6))

    conditions = ["A", "B", "C"]
    labels = ["No Hint", "Correct Hint", "Wrong Hint"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    bars = ax.bar(
        labels,
        [accuracy.get(c, 0) for c in conditions],
        color=colors,
        edgecolor="black",
        linewidth=1.2,
    )

    # Add value labels on bars
    for bar, val in zip(bars, [accuracy.get(c, 0) for c in conditions]):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 1,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_title("Experiment 1: Solver Accuracy by Condition", fontsize=14, fontweight="bold")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp1_accuracy_with_stats(df: pd.DataFrame, output_path: Path) -> None:
    """Plot accuracy with confidence intervals and p-values.

    Args:
        df: Experiment 1 results DataFrame
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    conditions = ["A", "B", "C"]
    labels = ["No Hint\n(Baseline)", "Correct Hint", "Wrong Hint"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    # Compute means and CIs
    means = []
    ci_lows = []
    ci_highs = []
    data_by_condition = {}

    for c in conditions:
        data = df[df["condition"] == c]["correct"].values * 100
        data_by_condition[c] = data
        mean, ci_low, ci_high = compute_ci(data)
        means.append(mean)
        ci_lows.append(mean - ci_low)
        ci_highs.append(ci_high - mean)

    x = np.arange(len(conditions))

    # Plot bars with error bars
    bars = ax.bar(
        x,
        means,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        yerr=[ci_lows, ci_highs],
        capsize=5,
        error_kw={"linewidth": 2, "capthick": 2},
    )

    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci_highs[i] + 2,
            f"{mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Statistical comparisons
    if HAS_SCIPY and len(data_by_condition["A"]) > 1 and len(data_by_condition["C"]) > 1:
        # A vs C comparison
        _, p_ac = compute_t_test(data_by_condition["A"], data_by_condition["C"])
        d_ac = compute_cohens_d(data_by_condition["A"], data_by_condition["C"])

        # Draw significance bracket
        y_max = max(means) + max(ci_highs) + 8
        ax.plot([0, 0, 2, 2], [y_max - 2, y_max, y_max, y_max - 2], "k-", linewidth=1.5)
        ax.text(1, y_max + 1, f"{significance_stars(p_ac)} ({format_p_value(p_ac)})",
                ha="center", va="bottom", fontsize=10)

        # Add effect size to legend
        effect_text = f"Cohen's d (A vs C) = {d_ac:.2f}"
        ax.text(0.98, 0.02, effect_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_title("Experiment 1: Solver Accuracy by Condition\n(with 95% CI)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 120)
    ax.grid(axis="y", alpha=0.3)

    # Add sample size annotation
    n_per_cond = len(df[df["condition"] == "A"])
    ax.text(0.02, 0.98, f"n = {n_per_cond} per condition", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp1_iqx_boxplot(df: pd.DataFrame, output_path: Path) -> None:
    """Plot IQx distribution by condition for Experiment 1.

    Args:
        df: Experiment 1 results DataFrame
        output_path: Path to save plot
    """
    # Filter out cold start and None values
    df_clean = df[df["cold_start"] == 0].dropna(subset=["IQx"])

    fig, ax = plt.subplots(figsize=(8, 6))

    conditions = ["A", "B", "C"]
    labels = ["No Hint", "Correct Hint", "Wrong Hint"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    data = [df_clean[df_clean["condition"] == c]["IQx"].values for c in conditions]

    bp = ax.boxplot(
        data,
        labels=labels,
        patch_artist=True,
        showfliers=True,
        flierprops={"marker": "o", "markersize": 4, "alpha": 0.5},
    )

    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_ylabel("IQx (Influence Quotient)", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_title("Experiment 1: IQx Distribution by Condition", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp1_iqx_with_stats(df: pd.DataFrame, output_path: Path) -> None:
    """Plot IQx distribution with statistical annotations.

    Args:
        df: Experiment 1 results DataFrame
        output_path: Path to save plot
    """
    df_clean = df[df["cold_start"] == 0].dropna(subset=["IQx"])

    fig, ax = plt.subplots(figsize=(9, 7))

    conditions = ["A", "B", "C"]
    labels = ["No Hint\n(Baseline)", "Correct Hint", "Wrong Hint"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    data = [df_clean[df_clean["condition"] == c]["IQx"].values for c in conditions]

    # Violin plot for distribution visualization
    parts = ax.violinplot(data, positions=range(len(conditions)), showmeans=True, showextrema=False)

    for i, (pc, color) in enumerate(zip(parts["bodies"], colors)):
        pc.set_facecolor(color)
        pc.set_alpha(0.6)

    parts["cmeans"].set_color("black")
    parts["cmeans"].set_linewidth(2)

    # Overlay boxplot
    bp = ax.boxplot(
        data,
        positions=range(len(conditions)),
        widths=0.15,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "white", "linewidth": 2},
    )

    for patch in bp["boxes"]:
        patch.set_facecolor("gray")
        patch.set_alpha(0.8)

    # Add means with CIs as text
    for i, d in enumerate(data):
        mean, ci_low, ci_high = compute_ci(d)
        ax.text(i, ax.get_ylim()[1] * 0.95, f"μ = {mean:.3f}\n[{ci_low:.3f}, {ci_high:.3f}]",
                ha="center", va="top", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    # Statistical tests
    if HAS_SCIPY and len(data[0]) > 1 and len(data[2]) > 1:
        # Mann-Whitney U test (non-parametric, better for IQx)
        _, p_ac = compute_mann_whitney(data[0], data[2])
        d_ac = compute_cohens_d(data[0], data[2])

        # Add stats box
        stats_text = f"A vs C:\n{format_p_value(p_ac)} {significance_stars(p_ac)}\nd = {d_ac:.2f}"
        ax.text(0.98, 0.02, stats_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax.set_ylabel("IQx (Influence Quotient)", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_title("Experiment 1: IQx Distribution by Condition\n(Violin + Box, with statistics)",
                 fontsize=14, fontweight="bold")
    ax.set_xticks(range(len(conditions)))
    ax.set_xticklabels(labels)
    ax.grid(axis="y", alpha=0.3)

    # Add legend for plot elements
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=colors[0], alpha=0.6, label="No Hint"),
        Patch(facecolor=colors[1], alpha=0.6, label="Correct Hint"),
        Patch(facecolor=colors[2], alpha=0.6, label="Wrong Hint"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp1_alert_rate(df: pd.DataFrame, output_path: Path) -> None:
    """Plot alert rate by condition for Experiment 1 with error bars and p-values.

    Args:
        df: Experiment 1 results DataFrame
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(9, 7))

    conditions = ["A", "B", "C"]
    labels = ["No Hint\n(Baseline)", "Correct Hint", "Wrong Hint"]
    colors = ["#2ecc71", "#3498db", "#e74c3c"]

    # Compute means and CIs for alert rate (binary data)
    means = []
    ci_lows = []
    ci_highs = []
    data_by_condition = {}

    for c in conditions:
        data = df[df["condition"] == c]["alert"].values * 100
        data_by_condition[c] = data
        mean, ci_low, ci_high = compute_ci(data)
        means.append(mean)
        ci_lows.append(mean - ci_low)
        ci_highs.append(ci_high - mean)

    x = np.arange(len(conditions))

    # Plot bars with error bars
    bars = ax.bar(
        x,
        means,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        yerr=[ci_lows, ci_highs],
        capsize=5,
        error_kw={"linewidth": 2, "capthick": 2},
    )

    # Add value labels on bars
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci_highs[i] + 1,
            f"{mean:.1f}%",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Statistical comparisons (A vs C)
    if HAS_SCIPY and len(data_by_condition["A"]) > 1 and len(data_by_condition["C"]) > 1:
        _, p_ac = compute_mann_whitney(data_by_condition["A"], data_by_condition["C"])
        d_ac = compute_cohens_d(data_by_condition["A"], data_by_condition["C"])

        # Draw significance bracket
        y_max = max(means) + max(ci_highs) + 5
        ax.plot([0, 0, 2, 2], [y_max - 1, y_max, y_max, y_max - 1], "k-", linewidth=1.5)
        ax.text(1, y_max + 0.5, f"{significance_stars(p_ac)} ({format_p_value(p_ac)})",
                ha="center", va="bottom", fontsize=10)

        # Effect size annotation
        effect_text = f"Cohen's d (A vs C) = {d_ac:.2f}"
        ax.text(0.98, 0.02, effect_text, transform=ax.transAxes,
                ha="right", va="bottom", fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax.set_ylabel("Alert Rate (%)", fontsize=12)
    ax.set_xlabel("Condition", fontsize=12)
    ax.set_title("Experiment 1: Anomaly Alert Rate by Condition\n(with 95% CI)", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, max(max(means) + max(ci_highs) * 2, 10) + 10)
    ax.grid(axis="y", alpha=0.3)

    # Sample size annotation
    n_per_cond = len(df[df["condition"] == "A"])
    ax.text(0.02, 0.98, f"n = {n_per_cond} per condition", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp2_propagation_risk(df: pd.DataFrame, output_path: Path) -> None:
    """Plot propagation risk over trials/hops for Experiment 2 with CI bands.

    Args:
        df: Experiment 2 results DataFrame
        output_path: Path to save plot
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    # Create interaction index (trial * 3 + hop)
    df = df.copy()
    df["interaction_idx"] = df["trial"] * 3 + df["hop"]

    # Group by interaction index and compute mean + CI
    grouped = df.groupby("interaction_idx")["propagation_risk"]

    x_vals = sorted(df["interaction_idx"].unique())
    means = []
    ci_lows = []
    ci_highs = []

    for idx in x_vals:
        data = grouped.get_group(idx).values if idx in grouped.groups else np.array([0])
        mean, ci_low, ci_high = compute_ci(data)
        means.append(mean)
        ci_lows.append(ci_low)
        ci_highs.append(ci_high)

    means = np.array(means)
    ci_lows = np.array(ci_lows)
    ci_highs = np.array(ci_highs)

    # Plot mean line with CI band
    ax.plot(x_vals, means, color="#9b59b6", linewidth=2, label="Mean PR")
    ax.fill_between(x_vals, ci_lows, ci_highs, color="#9b59b6", alpha=0.2, label="95% CI")

    # Add horizontal line at PR = 1.0 (critical threshold)
    ax.axhline(y=1.0, color="#e74c3c", linestyle="--", linewidth=2, label="Critical (PR=1.0)")

    # Annotate final PR value with CI
    if len(means) > 0:
        final_mean = means[-1]
        final_ci_low = ci_lows[-1]
        final_ci_high = ci_highs[-1]
        ax.annotate(
            f"Final: {final_mean:.3f}\n[{final_ci_low:.3f}, {final_ci_high:.3f}]",
            xy=(x_vals[-1], final_mean),
            xytext=(x_vals[-1] - len(x_vals) * 0.15, final_mean + 0.1),
            fontsize=9,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
        )

    ax.set_xlabel("Interaction Index", fontsize=12)
    ax.set_ylabel("Propagation Risk (Spectral Radius)", fontsize=12)
    ax.set_title(
        "Experiment 2: Propagation Risk Over Time\n(with 95% CI)",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp2_agent_influence(df: pd.DataFrame, output_path: Path) -> None:
    """Plot accumulated IQx per agent (as sender) for Experiment 2 with error bars.

    Args:
        df: Experiment 2 results DataFrame
        output_path: Path to save plot
    """
    df_clean = df.dropna(subset=["IQx"])

    fig, ax = plt.subplots(figsize=(9, 7))

    agents = ["agent_A", "agent_B", "agent_C"]
    colors = ["#e74c3c", "#f39c12", "#3498db"]

    # Compute mean IQx per agent with CIs (accumulated across trials)
    means = []
    ci_lows = []
    ci_highs = []
    data_by_agent = {}

    for agent in agents:
        # Get IQx values for this sender
        data = df_clean[df_clean["sender"] == agent]["IQx"].values
        data_by_agent[agent] = data
        if len(data) > 0:
            mean, ci_low, ci_high = compute_ci(data)
            means.append(mean * len(data))  # Scale by count for accumulated
            # CI for sum: sum of means with pooled variance
            ci_lows.append((mean - ci_low) * len(data))
            ci_highs.append((ci_high - mean) * len(data))
        else:
            means.append(0)
            ci_lows.append(0)
            ci_highs.append(0)

    x = np.arange(len(agents))

    bars = ax.bar(
        x,
        means,
        color=colors,
        edgecolor="black",
        linewidth=1.2,
        yerr=[ci_lows, ci_highs],
        capsize=5,
        error_kw={"linewidth": 2, "capthick": 2},
    )

    # Add value labels
    for i, (bar, mean) in enumerate(zip(bars, means)):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci_highs[i] + 0.1,
            f"{mean:.2f}",
            ha="center",
            va="bottom",
            fontsize=11,
            fontweight="bold",
        )

    # Statistical comparison: highest vs lowest influence agent
    if HAS_SCIPY:
        # Find agents with most and least influence
        sorted_agents = sorted(agents, key=lambda a: len(data_by_agent.get(a, [])) and np.mean(data_by_agent.get(a, [0])), reverse=True)
        if len(data_by_agent.get(sorted_agents[0], [])) > 1 and len(data_by_agent.get(sorted_agents[-1], [])) > 1:
            _, p_val = compute_t_test(data_by_agent[sorted_agents[0]], data_by_agent[sorted_agents[-1]])
            d_val = compute_cohens_d(data_by_agent[sorted_agents[0]], data_by_agent[sorted_agents[-1]])

            stats_text = f"Max vs Min:\n{format_p_value(p_val)} {significance_stars(p_val)}\nd = {d_val:.2f}"
            ax.text(0.98, 0.98, stats_text, transform=ax.transAxes,
                    ha="right", va="top", fontsize=9,
                    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))

    ax.set_ylabel("Accumulated IQx", fontsize=12)
    ax.set_xlabel("Agent (Sender)", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(agents)
    ax.set_title(
        "Experiment 2: Accumulated Influence per Agent\n(with 95% CI)",
        fontsize=14,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    # Add sample size annotation
    n_events = len(df_clean)
    ax.text(0.02, 0.98, f"n = {n_events} events", transform=ax.transAxes,
            ha="left", va="top", fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def plot_exp3_mitigation_compare(df: pd.DataFrame, output_path: Path) -> None:
    """Plot grouped comparison of mitigation effectiveness for Experiment 3 with error bars and p-values.

    Args:
        df: Experiment 3 results DataFrame
        output_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    conditions = ["A", "B", "C"]
    labels = ["No Hint", "Correct", "Wrong"]
    x = np.arange(len(conditions))
    width = 0.35

    # Colors for without/with mitigation
    color_without = "#e74c3c"
    color_with = "#2ecc71"

    # Helper to compute grouped stats
    def get_stats_by_condition(df, col, mit_val, scale=1.0):
        means, ci_lows, ci_highs, raw_data = [], [], [], {}
        for c in conditions:
            subset = df[(df["condition"] == c) & (df["mitigation_enabled"] == mit_val)]
            if col == "IQx":
                subset = subset.dropna(subset=["IQx"])
            data = subset[col].values * scale
            raw_data[c] = data
            if len(data) > 0:
                mean, ci_low, ci_high = compute_ci(data)
                means.append(mean)
                ci_lows.append(mean - ci_low)
                ci_highs.append(ci_high - mean)
            else:
                means.append(0)
                ci_lows.append(0)
                ci_highs.append(0)
        return means, ci_lows, ci_highs, raw_data

    # Plot 1: Accuracy comparison with error bars
    ax1 = axes[0]
    for mit in [0, 1]:
        means, ci_lows, ci_highs, raw_data = get_stats_by_condition(df, "correct", mit, scale=100)
        offset = -width / 2 if mit == 0 else width / 2
        color = color_without if mit == 0 else color_with
        label = "Without Mitigation" if mit == 0 else "With Mitigation"
        ax1.bar(
            x + offset,
            means,
            width,
            label=label,
            color=color,
            edgecolor="black",
            yerr=[ci_lows, ci_highs],
            capsize=3,
            error_kw={"linewidth": 1.5},
        )
        if mit == 0:
            data_without = raw_data
        else:
            data_with = raw_data

    # P-value for Wrong Hint condition (without vs with mitigation)
    if HAS_SCIPY and len(data_without.get("C", [])) > 1 and len(data_with.get("C", [])) > 1:
        _, p_val = compute_t_test(data_without["C"], data_with["C"])
        ax1.text(2, ax1.get_ylim()[1] * 0.95, f"{significance_stars(p_val)}\n{format_p_value(p_val)}",
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax1.set_ylabel("Accuracy (%)", fontsize=11)
    ax1.set_xlabel("Condition", fontsize=11)
    ax1.set_title("Accuracy\n(with 95% CI)", fontsize=12, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend(fontsize=8, loc="lower left")
    ax1.grid(axis="y", alpha=0.3)
    ax1.set_ylim(0, 120)

    # Plot 2: Alert rate comparison with error bars
    ax2 = axes[1]
    for mit in [0, 1]:
        means, ci_lows, ci_highs, raw_data = get_stats_by_condition(df, "alert", mit, scale=100)
        offset = -width / 2 if mit == 0 else width / 2
        color = color_without if mit == 0 else color_with
        label = "Without Mitigation" if mit == 0 else "With Mitigation"
        ax2.bar(
            x + offset,
            means,
            width,
            label=label,
            color=color,
            edgecolor="black",
            yerr=[ci_lows, ci_highs],
            capsize=3,
            error_kw={"linewidth": 1.5},
        )
        if mit == 0:
            alert_without = raw_data
        else:
            alert_with = raw_data

    # P-value for Wrong Hint condition
    if HAS_SCIPY and len(alert_without.get("C", [])) > 1 and len(alert_with.get("C", [])) > 1:
        _, p_val = compute_t_test(alert_without["C"], alert_with["C"])
        ax2.text(2, ax2.get_ylim()[1] * 0.95 if ax2.get_ylim()[1] > 0 else 10,
                f"{significance_stars(p_val)}\n{format_p_value(p_val)}",
                ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax2.set_ylabel("Alert Rate (%)", fontsize=11)
    ax2.set_xlabel("Condition", fontsize=11)
    ax2.set_title("Alert Rate\n(with 95% CI)", fontsize=12, fontweight="bold")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels)
    ax2.legend(fontsize=8, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)

    # Plot 3: Average IQx comparison with error bars
    ax3 = axes[2]
    for mit in [0, 1]:
        means, ci_lows, ci_highs, raw_data = get_stats_by_condition(df, "IQx", mit, scale=1.0)
        offset = -width / 2 if mit == 0 else width / 2
        color = color_without if mit == 0 else color_with
        label = "Without Mitigation" if mit == 0 else "With Mitigation"
        ax3.bar(
            x + offset,
            means,
            width,
            label=label,
            color=color,
            edgecolor="black",
            yerr=[ci_lows, ci_highs],
            capsize=3,
            error_kw={"linewidth": 1.5},
        )
        if mit == 0:
            iqx_without = raw_data
        else:
            iqx_with = raw_data

    # P-value and effect size for Wrong Hint condition
    if HAS_SCIPY and len(iqx_without.get("C", [])) > 1 and len(iqx_with.get("C", [])) > 1:
        _, p_val = compute_t_test(iqx_without["C"], iqx_with["C"])
        d_val = compute_cohens_d(iqx_without["C"], iqx_with["C"])
        stats_text = f"{significance_stars(p_val)} {format_p_value(p_val)}\nd = {d_val:.2f}"
        ax3.text(2, ax3.get_ylim()[1] * 0.95 if ax3.get_ylim()[1] > 0 else 1,
                stats_text, ha="center", va="top", fontsize=8,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    ax3.set_ylabel("Average IQx", fontsize=11)
    ax3.set_xlabel("Condition", fontsize=11)
    ax3.set_title("Avg Influence Quotient\n(with 95% CI)", fontsize=12, fontweight="bold")
    ax3.set_xticks(x)
    ax3.set_xticklabels(labels)
    ax3.legend(fontsize=8, loc="upper left")
    ax3.grid(axis="y", alpha=0.3)

    fig.suptitle(
        "Experiment 3: Mitigation Policy Comparison",
        fontsize=14,
        fontweight="bold",
        y=1.02,
    )

    # Add overall sample size
    n_total = len(df)
    fig.text(0.99, 0.01, f"n = {n_total} total observations", ha="right", va="bottom",
             fontsize=9, style="italic")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main() -> None:
    """Generate all experiment plots."""
    if not HAS_DEPS:
        print("Cannot generate plots without matplotlib and pandas.")
        print("Install with: pip install matplotlib pandas")
        return

    print("Generating experiment plots...")
    print("-" * 40)

    if not HAS_SCIPY:
        print("Note: scipy not available, statistical annotations will be limited")
        print("Install with: pip install scipy")
        print()

    results_dir = Path("experiments/results")
    plots_dir = Path("experiments/plots")
    plots_dir.mkdir(parents=True, exist_ok=True)

    # Experiment 1 plots
    exp1_path = results_dir / "exp1_results.csv"
    if exp1_path.exists():
        print("\nExperiment 1 plots:")
        df1 = pd.read_csv(exp1_path)
        plot_exp1_accuracy(df1, plots_dir / "exp1_accuracy.png")
        plot_exp1_accuracy_with_stats(df1, plots_dir / "exp1_accuracy_stats.png")
        plot_exp1_iqx_boxplot(df1, plots_dir / "exp1_iqx_box.png")
        plot_exp1_iqx_with_stats(df1, plots_dir / "exp1_iqx_stats.png")
        plot_exp1_alert_rate(df1, plots_dir / "exp1_alert_rate.png")
    else:
        print(f"Skipping Experiment 1 plots: {exp1_path} not found")
        print("Run: python experiments/run_exp1_wrong_hint.py")

    # Experiment 2 plots
    exp2_path = results_dir / "exp2_results.csv"
    if exp2_path.exists():
        print("\nExperiment 2 plots:")
        df2 = pd.read_csv(exp2_path)
        plot_exp2_propagation_risk(df2, plots_dir / "exp2_propagation_risk.png")
        plot_exp2_agent_influence(df2, plots_dir / "exp2_agent_influence.png")
    else:
        print(f"Skipping Experiment 2 plots: {exp2_path} not found")
        print("Run: python experiments/run_exp2_propagation.py")

    # Experiment 3 plots
    exp3_path = results_dir / "exp3_results.csv"
    if exp3_path.exists():
        print("\nExperiment 3 plots:")
        df3 = pd.read_csv(exp3_path)
        plot_exp3_mitigation_compare(df3, plots_dir / "exp3_mitigation_compare.png")
    else:
        print(f"Skipping Experiment 3 plots: {exp3_path} not found")
        print("Run: python experiments/run_exp3_mitigation.py")

    print("\n" + "=" * 40)
    print("Plot generation complete!")
    print(f"Output directory: {plots_dir.absolute()}")


if __name__ == "__main__":
    main()
