"""Risk report generation for TraceIQ.

This module provides functions to generate summary reports from tracked events,
including risk analysis, policy effectiveness, and agent rankings.
"""

from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from traceiq.models import ScoreResult
    from traceiq.schema import TraceIQEvent


def generate_risk_report(
    events: list[TraceIQEvent],
    scores: list[ScoreResult],
    run_id: str,
    output_path: Path | str,
    format: Literal["markdown", "json"] = "markdown",
) -> None:
    """Generate a comprehensive risk report from tracked events.

    The report includes:
    - Top risky agents (by accumulated risk_score)
    - Top risky edges (by mean risk on edge)
    - Propagation risk timeline
    - Alert timeline
    - Policy effectiveness (blocked vs applied)

    Args:
        events: List of TraceIQEvent objects
        scores: List of ScoreResult objects
        run_id: Run identifier for the report
        output_path: Path to write the report
        format: Output format ("markdown" or "json")
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build report data
    report_data = _build_report_data(events, scores, run_id)

    if format == "json":
        _write_json_report(report_data, output_path)
    else:
        _write_markdown_report(report_data, output_path)


def _build_report_data(
    events: list[TraceIQEvent],
    scores: list[ScoreResult],
    run_id: str,
) -> dict[str, Any]:
    """Build report data from events and scores."""
    # Create score lookup
    score_map = {str(s.event_id): s for s in scores}

    # Initialize accumulators
    agent_risk_totals: dict[str, float] = defaultdict(float)
    agent_event_counts: dict[str, int] = defaultdict(int)
    edge_risks: dict[tuple[str, str], list[float]] = defaultdict(list)

    # Event type counts
    attempted_count = 0
    applied_count = 0
    blocked_count = 0

    # Alert tracking
    alerts: list[dict[str, Any]] = []
    alert_count = 0

    # Process events
    for event in events:
        score = score_map.get(event.event_id)

        # Count event types
        if event.event_type == "attempted":
            attempted_count += 1
        elif event.event_type == "applied":
            applied_count += 1
        elif event.event_type == "blocked":
            blocked_count += 1

        if score is None:
            continue

        # Accumulate risk by sender
        risk_score = getattr(score, "risk_score", None)
        if risk_score is not None:
            agent_risk_totals[event.sender_id] += risk_score
            agent_event_counts[event.sender_id] += 1
            edge_risks[(event.sender_id, event.receiver_id)].append(risk_score)

        # Track alerts
        if score.alert_flag:
            alert_count += 1
            alerts.append(
                {
                    "event_id": event.event_id,
                    "ts": event.ts,
                    "sender_id": event.sender_id,
                    "receiver_id": event.receiver_id,
                    "z_score": score.Z_score,
                    "iqx": score.IQx,
                    "risk_score": risk_score,
                }
            )

    # Compute top risky agents
    top_agents = sorted(
        [
            {
                "agent_id": agent_id,
                "total_risk": total,
                "event_count": agent_event_counts[agent_id],
                "avg_risk": total / agent_event_counts[agent_id]
                if agent_event_counts[agent_id] > 0
                else 0,
            }
            for agent_id, total in agent_risk_totals.items()
        ],
        key=lambda x: -x["total_risk"],
    )[:10]

    # Compute top risky edges
    top_edges = sorted(
        [
            {
                "sender": edge[0],
                "receiver": edge[1],
                "event_count": len(risks),
                "mean_risk": sum(risks) / len(risks) if risks else 0,
                "max_risk": max(risks) if risks else 0,
            }
            for edge, risks in edge_risks.items()
        ],
        key=lambda x: -x["mean_risk"],
    )[:10]

    # Policy effectiveness
    total_events = len(events)
    policy_effectiveness = {
        "total_events": total_events,
        "attempted_count": attempted_count,
        "applied_count": applied_count,
        "blocked_count": blocked_count,
        "block_rate": blocked_count / total_events if total_events > 0 else 0,
        "alert_count": alert_count,
        "alert_rate_applied": alert_count / applied_count if applied_count > 0 else 0,
    }

    return {
        "run_id": run_id,
        "generated_at": datetime.now().isoformat(),
        "summary": {
            "total_events": total_events,
            "unique_senders": len({e.sender_id for e in events}),
            "unique_receivers": len({e.receiver_id for e in events}),
            "alert_count": alert_count,
        },
        "top_risky_agents": top_agents,
        "top_risky_edges": top_edges,
        "policy_effectiveness": policy_effectiveness,
        "recent_alerts": alerts[-20:],  # Last 20 alerts
    }


def _write_markdown_report(data: dict[str, Any], output_path: Path) -> None:
    """Write report in Markdown format."""
    lines = [
        "# TraceIQ Risk Report",
        "",
        f"**Run ID:** {data['run_id']}",
        "",
        f"**Generated:** {data['generated_at']}",
        "",
        "---",
        "",
        "## Summary",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| Total Events | {data['summary']['total_events']} |",
        f"| Unique Senders | {data['summary']['unique_senders']} |",
        f"| Unique Receivers | {data['summary']['unique_receivers']} |",
        f"| Alert Count | {data['summary']['alert_count']} |",
        "",
        "---",
        "",
        "## Top Risky Agents",
        "",
    ]

    if data["top_risky_agents"]:
        lines.extend(
            [
                "| Rank | Agent | Total Risk | Events | Avg Risk |",
                "|------|-------|------------|--------|----------|",
            ]
        )
        for i, agent in enumerate(data["top_risky_agents"], 1):
            lines.append(
                f"| {i} | {agent['agent_id']} | {agent['total_risk']:.4f} | "
                f"{agent['event_count']} | {agent['avg_risk']:.4f} |"
            )
    else:
        lines.append("*No risky agents found*")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Top Risky Edges",
            "",
        ]
    )

    if data["top_risky_edges"]:
        lines.extend(
            [
                "| Sender | Receiver | Events | Mean Risk | Max Risk |",
                "|--------|----------|--------|-----------|----------|",
            ]
        )
        for edge in data["top_risky_edges"]:
            lines.append(
                f"| {edge['sender']} | {edge['receiver']} | {edge['event_count']} | "
                f"{edge['mean_risk']:.4f} | {edge['max_risk']:.4f} |"
            )
    else:
        lines.append("*No risky edges found*")

    lines.extend(
        [
            "",
            "---",
            "",
            "## Policy Effectiveness",
            "",
            "| Metric | Value |",
            "|--------|-------|",
            f"| Total Events | {data['policy_effectiveness']['total_events']} |",
            f"| Attempted | {data['policy_effectiveness']['attempted_count']} |",
            f"| Applied | {data['policy_effectiveness']['applied_count']} |",
            f"| Blocked | {data['policy_effectiveness']['blocked_count']} |",
            f"| Block Rate | {data['policy_effectiveness']['block_rate']:.2%} |",
            f"| Alert Count | {data['policy_effectiveness']['alert_count']} |",
            f"| Alert Rate (Applied) | {data['policy_effectiveness']['alert_rate_applied']:.2%} |",
            "",
            "---",
            "",
            "## Recent Alerts",
            "",
        ]
    )

    if data["recent_alerts"]:
        lines.extend(
            [
                "| Event ID | Sender | Receiver | Z-Score | IQx | Risk |",
                "|----------|--------|----------|---------|-----|------|",
            ]
        )
        for alert in data["recent_alerts"]:
            z = f"{alert['z_score']:.2f}" if alert["z_score"] else "-"
            iqx = f"{alert['iqx']:.4f}" if alert["iqx"] else "-"
            risk = f"{alert['risk_score']:.4f}" if alert["risk_score"] else "-"
            lines.append(
                f"| {str(alert['event_id'])[:8]}... | {alert['sender_id']} | "
                f"{alert['receiver_id']} | {z} | {iqx} | {risk} |"
            )
    else:
        lines.append("*No recent alerts*")

    lines.append("")

    output_path.write_text("\n".join(lines))


def _write_json_report(data: dict[str, Any], output_path: Path) -> None:
    """Write report in JSON format."""
    output_path.write_text(json.dumps(data, indent=2, default=str))


def generate_quick_summary(
    events: list[TraceIQEvent],
    scores: list[ScoreResult],
) -> dict[str, Any]:
    """Generate a quick summary without writing to file.

    Args:
        events: List of events
        scores: List of scores

    Returns:
        Summary dict with key metrics
    """
    score_map = {str(s.event_id): s for s in scores}

    total = len(events)
    blocked = sum(1 for e in events if e.event_type == "blocked")
    applied = sum(1 for e in events if e.event_type == "applied")
    alerts = sum(1 for s in scores if s.alert_flag)

    # Get risk scores
    risk_scores = [
        getattr(score_map.get(e.event_id), "risk_score", None) for e in events
    ]
    valid_risks = [r for r in risk_scores if r is not None]

    return {
        "total_events": total,
        "applied_count": applied,
        "blocked_count": blocked,
        "alert_count": alerts,
        "block_rate": blocked / total if total > 0 else 0,
        "alert_rate": alerts / applied if applied > 0 else 0,
        "avg_risk": sum(valid_risks) / len(valid_risks) if valid_risks else None,
        "max_risk": max(valid_risks) if valid_risks else None,
    }


def plot_risk_calibration_curve(
    risk_scores: list[float],
    outcomes: list[bool],
    output_path: Path | str,
    n_bins: int = 10,
) -> None:
    """Plot calibration curve showing observed failure rate per risk bin.

    This proves metric validity by showing:
    - Monotonic relationship between risk_score and failure rate
    - Well-calibrated risk scores track actual outcomes

    The calibration curve is a critical validation plot that should be
    part of any production deployment report.

    Args:
        risk_scores: List of risk scores in [0, 1]
        outcomes: List of boolean outcomes (True = failure/bad outcome)
        output_path: Path to save the plot
        n_bins: Number of bins for risk score grouping (default: 10)

    Raises:
        ImportError: If matplotlib is not installed
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as err:
        raise ImportError(
            "matplotlib and numpy required for plotting. "
            "Install with: pip install 'traceiq[plot]'"
        ) from err

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Filter to valid pairs
    valid_pairs = [
        (r, o) for r, o in zip(risk_scores, outcomes, strict=False) if r is not None
    ]

    if len(valid_pairs) < n_bins:
        # Not enough data for meaningful calibration
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.text(
            0.5,
            0.5,
            f"Insufficient data ({len(valid_pairs)} samples)\n"
            f"Need at least {n_bins} for calibration curve",
            ha="center",
            va="center",
            fontsize=12,
        )
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Predicted Risk Score")
        ax.set_ylabel("Observed Failure Rate")
        ax.set_title("Risk Calibration Curve")
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
        return

    risk_arr = np.array([r for r, _ in valid_pairs])
    outcome_arr = np.array([o for _, o in valid_pairs], dtype=float)

    # Bin risk scores
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    observed_rates = []
    bin_counts = []

    for i in range(n_bins):
        mask = (risk_arr >= bins[i]) & (risk_arr < bins[i + 1])
        count = mask.sum()
        bin_counts.append(count)
        if count > 0:
            rate = outcome_arr[mask].mean()
        else:
            rate = np.nan
        observed_rates.append(rate)

    # Create plot
    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot observed rates
    valid_mask = ~np.isnan(observed_rates)
    ax.plot(
        np.array(bin_centers)[valid_mask],
        np.array(observed_rates)[valid_mask],
        "o-",
        color="blue",
        label="Observed",
        markersize=8,
    )

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")

    # Labels and title
    ax.set_xlabel("Predicted Risk Score", fontsize=12)
    ax.set_ylabel("Observed Failure Rate", fontsize=12)
    ax.set_title("Risk Calibration Curve", fontsize=14)
    ax.legend(loc="upper left")
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, alpha=0.3)

    # Add sample counts as annotations
    for _i, (x, y, count) in enumerate(
        zip(bin_centers, observed_rates, bin_counts, strict=False)
    ):
        if not np.isnan(y) and count > 0:
            ax.annotate(
                f"n={count}",
                xy=(x, y),
                xytext=(0, 10),
                textcoords="offset points",
                ha="center",
                fontsize=8,
                alpha=0.7,
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
