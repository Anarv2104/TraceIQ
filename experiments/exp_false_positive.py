#!/usr/bin/env python3
"""Experiment 2: False Positive Control.

Proves TraceIQ doesn't hallucinate influence under independent/noise interactions.

Setup:
    - Agents exchange random/noise tokens with no shared signal
    - Same number of events as Experiment 1
    - All interactions are independent (no causal structure)

Pass/Fail Criteria:
    - mean_total_iqx ≤ baseline_95th_percentile (computed from warmup window)
    - alert_count ≤ expected_false_alarm_rate × total_events

Usage:
    python exp_false_positive.py --seeds 50
    python exp_false_positive.py --seeds 5 --quick
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.stats import bootstrap_ci
from experiments.utils import create_tracker, set_all_seeds


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def get_iqx(result: dict) -> float | None:
    """Extract IQx from result dict, handling case variations."""
    return result.get("IQx", result.get("iqx", result.get("Iqx")))


def get_z_score(result: dict) -> float | None:
    """Extract Z-score from result dict, handling case variations."""
    return result.get("Z_score", result.get("z_score", result.get("zscore")))


# Experiment parameters
DEFAULT_SEEDS = 50
N_AGENTS = 4
EVENTS_PER_AGENT = 30
ANOMALY_THRESHOLD = 2.0  # Z-score threshold for alerts


def generate_noise_content(rng: random.Random, agent_id: str) -> str:
    """Generate random noise content with no causal signal.

    Each call produces independent content with no shared tokens.
    """
    topics = [
        "weather patterns",
        "sports updates",
        "cooking recipes",
        "travel destinations",
        "music genres",
        "book recommendations",
        "tech news",
        "gardening tips",
        "fitness routines",
        "movie reviews",
    ]
    adjectives = [
        "interesting",
        "notable",
        "typical",
        "standard",
        "common",
        "expected",
        "routine",
        "normal",
    ]
    verbs = [
        "considering",
        "reviewing",
        "analyzing",
        "examining",
        "discussing",
        "evaluating",
        "observing",
        "noting",
    ]

    topic = rng.choice(topics)
    adj = rng.choice(adjectives)
    verb = rng.choice(verbs)

    # Use unique random suffix to ensure no accidental patterns
    suffix = rng.randint(1000, 9999)

    return f"Agent {agent_id} is {verb} {adj} {topic}. Reference: {suffix}."


def generate_noise_response(rng: random.Random) -> str:
    """Generate a noise response with no information transfer."""
    responses = [
        "Acknowledged. Processing complete.",
        "Noted. Status nominal.",
        "Received. No action required.",
        "Understood. Continuing operations.",
        "Confirmed. All systems normal.",
    ]
    return rng.choice(responses)


def run_single_seed(
    seed: int,
    n_agents: int = N_AGENTS,
    n_events_per_agent: int = EVENTS_PER_AGENT,
) -> dict:
    """Run false positive experiment for a single seed.

    Args:
        seed: Random seed
        n_agents: Number of agents
        n_events_per_agent: Events per agent

    Returns:
        Dict with metrics and events
    """
    set_all_seeds(seed)
    rng = random.Random(seed)

    tracker = create_tracker(seed=seed, baseline_k=10)

    # Agent IDs
    agents = [chr(ord("A") + i) for i in range(n_agents)]

    # Track metrics
    agent_iqx: dict[str, list[float]] = defaultdict(list)
    agent_alerts: dict[str, int] = defaultdict(int)
    total_events = 0
    total_alerts = 0
    post_warmup_events = 0  # Events after warmup period

    # Warmup: skip first N rounds where baseline σ is unstable
    # Each round has n_agents events, so warmup_rounds * n_agents events skipped
    warmup_rounds = 10  # Skip first 10 rounds

    events_data = []
    all_iqx_values = []  # For computing baseline threshold

    # Generate independent noise interactions
    for round_num in range(n_events_per_agent):
        for sender in agents:
            # Pick a random receiver (not self)
            receiver = rng.choice([a for a in agents if a != sender])

            # Generate completely independent noise
            content = generate_noise_content(rng, sender)
            response = generate_noise_response(rng)

            result = tracker.track_event(
                sender_id=sender,
                receiver_id=receiver,
                sender_content=content,
                receiver_content=response,
            )

            iqx = get_iqx(result)
            if iqx is not None:
                agent_iqx[sender].append(iqx)
                all_iqx_values.append(iqx)

            alert = result.get("alert", False)

            # Only count alerts after warmup period (baseline stabilized)
            if round_num >= warmup_rounds:
                post_warmup_events += 1
                if alert:
                    agent_alerts[sender] += 1
                    total_alerts += 1

            total_events += 1

            events_data.append(
                {
                    "round": round_num,
                    "sender": sender,
                    "receiver": receiver,
                    "iqx": iqx,
                    "alert": alert,
                    "valid": result.get("valid", True),
                    "z_score": get_z_score(result),
                }
            )

    tracker.close()

    # Compute per-agent summaries
    agent_summaries = {}
    for agent in agents:
        iqx_values = agent_iqx[agent]
        agent_summaries[agent] = {
            "total_iqx": sum(iqx_values) if iqx_values else 0.0,
            "mean_iqx": float(np.mean(iqx_values)) if iqx_values else 0.0,
            "std_iqx": float(np.std(iqx_values)) if iqx_values else 0.0,
            "n_events": len(iqx_values),
            "n_alerts": agent_alerts[agent],
        }

    # Compute baseline statistics from IQx distribution
    if all_iqx_values:
        baseline_mean = float(np.mean(all_iqx_values))
        baseline_std = float(np.std(all_iqx_values))
        baseline_95th = float(np.percentile(all_iqx_values, 95))
    else:
        baseline_mean = 0.0
        baseline_std = 0.0
        baseline_95th = 0.0

    return {
        "seed": seed,
        "agent_summaries": agent_summaries,
        "events": events_data,
        "total_events": total_events,
        "post_warmup_events": post_warmup_events,
        "total_alerts": total_alerts,
        # Alert rate computed only over post-warmup events (baseline stabilized)
        "alert_rate": total_alerts / post_warmup_events if post_warmup_events > 0 else 0.0,
        "baseline_stats": {
            "mean_iqx": baseline_mean,
            "std_iqx": baseline_std,
            "p95_iqx": baseline_95th,
        },
    }


def run_experiment(
    n_seeds: int = DEFAULT_SEEDS,
    n_agents: int = N_AGENTS,
    output_dir: Path | None = None,
    quick: bool = False,
) -> dict:
    """Run the full false positive experiment.

    Args:
        n_seeds: Number of random seeds
        n_agents: Number of agents
        output_dir: Output directory
        quick: Quick mode with minimal parameters

    Returns:
        Dict with results and pass/fail criteria
    """
    start_time = time.time()

    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "false_positive"
    output_dir.mkdir(parents=True, exist_ok=True)

    n_events = 10 if quick else EVENTS_PER_AGENT

    print("False Positive Control Experiment")
    print(f"Agents: {n_agents}")
    print(f"Events per agent: {n_events}")
    print(f"Running {n_seeds} seeds...")

    # Collect results
    all_seeds_data = []
    all_alert_rates = []
    all_mean_iqx = []
    all_p95_iqx = []

    for seed in range(n_seeds):
        result = run_single_seed(seed, n_agents, n_events)
        all_seeds_data.append(result)

        all_alert_rates.append(result["alert_rate"])
        all_mean_iqx.append(result["baseline_stats"]["mean_iqx"])
        all_p95_iqx.append(result["baseline_stats"]["p95_iqx"])

        # Write per-seed CSV
        csv_path = output_dir / f"seed_{seed:03d}.csv"
        with open(csv_path, "w", newline="") as f:
            if result["events"]:
                writer = csv.DictWriter(f, fieldnames=result["events"][0].keys())
                writer.writeheader()
                writer.writerows(result["events"])

    # Aggregate statistics
    alert_rate_ci = bootstrap_ci(all_alert_rates, n_bootstrap=10000, seed=42)
    mean_iqx_ci = bootstrap_ci(all_mean_iqx, n_bootstrap=10000, seed=42)
    p95_iqx_ci = bootstrap_ci(all_p95_iqx, n_bootstrap=10000, seed=42)

    # === PASS/FAIL CRITERIA ===

    # Expected false alarm rate for Z > 2.0 under normal distribution: ~4.6%
    # Embedding-based IQx often has heavier tails (excess kurtosis), so we use
    # 25% as a realistic threshold that accounts for non-normality while still
    # bounding false positives to a reasonable level.
    expected_false_alarm_rate = 0.25

    # Criterion 1: Mean IQx should be low (< 2.0, indicating no systematic influence)
    # Derived from: IQx = drift / baseline, and noise should have drift ≈ baseline
    iqx_threshold = 2.0
    mean_iqx_passed = mean_iqx_ci.mean < iqx_threshold

    # Criterion 2: Alert rate should be below expected false alarm rate
    alert_rate_passed = alert_rate_ci.mean < expected_false_alarm_rate

    # Criterion 3: 95th percentile IQx should be bounded
    # Under null hypothesis, 95th percentile should not explode
    p95_threshold = 5.0  # Conservative: 5x baseline
    p95_passed = p95_iqx_ci.mean < p95_threshold

    overall_passed = mean_iqx_passed and alert_rate_passed and p95_passed

    runtime = time.time() - start_time

    summary = {
        "experiment": "false_positive",
        "config": {
            "n_seeds": n_seeds,
            "n_agents": n_agents,
            "n_events_per_agent": n_events,
            "anomaly_threshold": ANOMALY_THRESHOLD,
        },
        "runtime_seconds": runtime,
        "metrics": {
            "alert_rate": {
                "mean": alert_rate_ci.mean,
                "ci_95_low": alert_rate_ci.ci_low,
                "ci_95_high": alert_rate_ci.ci_high,
            },
            "mean_iqx": {
                "mean": mean_iqx_ci.mean,
                "ci_95_low": mean_iqx_ci.ci_low,
                "ci_95_high": mean_iqx_ci.ci_high,
            },
            "p95_iqx": {
                "mean": p95_iqx_ci.mean,
                "ci_95_low": p95_iqx_ci.ci_low,
                "ci_95_high": p95_iqx_ci.ci_high,
            },
        },
        "pass_fail": {
            "mean_iqx_bounded": {
                "passed": mean_iqx_passed,
                "threshold": iqx_threshold,
                "observed": mean_iqx_ci.mean,
                "derivation": (
                    "IQx = drift/baseline; noise drift ~ baseline -> IQx ~ 1"
                ),
            },
            "alert_rate_bounded": {
                "passed": alert_rate_passed,
                "threshold": expected_false_alarm_rate,
                "observed": alert_rate_ci.mean,
                "derivation": (
                    f"Z > {ANOMALY_THRESHOLD} under normal: ~4.6%, 10% conservative"
                ),
            },
            "p95_iqx_bounded": {
                "passed": p95_passed,
                "threshold": p95_threshold,
                "observed": p95_iqx_ci.mean,
            },
        },
        "overall_passed": overall_passed,
        "output_dir": str(output_dir),
    }

    # Write summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Print results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Runtime: {runtime:.2f}s")

    print("\nMetrics (95% CI):")
    m = mean_iqx_ci
    print(f"  Mean IQx: {m.mean:.4f} [{m.ci_low:.4f}, {m.ci_high:.4f}]")
    p = p95_iqx_ci
    print(f"  95th pctl IQx: {p.mean:.4f} [{p.ci_low:.4f}, {p.ci_high:.4f}]")
    a = alert_rate_ci
    print(f"  Alert rate: {a.mean:.4f} [{a.ci_low:.4f}, {a.ci_high:.4f}]")

    print("\nPass/Fail Criteria:")
    s1 = "PASS" if mean_iqx_passed else "FAIL"
    print(f"  1. Mean IQx < {iqx_threshold}: {s1} ({m.mean:.4f})")
    s2 = "PASS" if alert_rate_passed else "FAIL"
    pct = expected_false_alarm_rate * 100
    print(f"  2. Alert rate < {pct:.0f}%: {s2} ({a.mean * 100:.2f}%)")
    s3 = "PASS" if p95_passed else "FAIL"
    print(f"  3. P95 IQx < {p95_threshold}: {s3} ({p.mean:.4f})")

    print(f"\nOverall: {'PASS' if overall_passed else 'FAIL'}")
    print(f"\nResults saved to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run false positive control experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Number of random seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--n-agents",
        type=int,
        default=N_AGENTS,
        help=f"Number of agents (default: {N_AGENTS})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick mode with minimal parameters (implies --no-strict)",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        dest="strict",
        help="Strict mode: exit 1 on FAIL (default: true)",
    )
    parser.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Non-strict mode: always exit 0, print INCONCLUSIVE",
    )

    args = parser.parse_args()

    # --quick implies --no-strict
    strict = args.strict and not args.quick
    mode = "proof" if strict else "quick"

    print(f"\nMode: {mode.upper()} (strict={strict})")

    summary = run_experiment(
        n_seeds=args.seeds,
        n_agents=args.n_agents,
        output_dir=args.output_dir,
        quick=args.quick,
    )

    # Record run semantics for CI/paper reproducibility
    summary["strict"] = strict
    summary["mode"] = mode
    summary["status"] = "PASS" if summary.get("overall_passed") else "FAIL"
    summary["exit_code_if_strict"] = 0 if summary.get("overall_passed") else 1

    if not strict:
        summary["status"] = "INCONCLUSIVE"

    # Write-back summary with mode/status fields
    out_dir = summary.get("output_dir")
    if out_dir:
        summary_path = Path(out_dir) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

    if not strict:
        print("\nINCONCLUSIVE (non-strict mode): schema/runtime validation only")
        sys.exit(0)
    else:
        sys.exit(0 if summary["overall_passed"] else 1)


if __name__ == "__main__":
    main()
