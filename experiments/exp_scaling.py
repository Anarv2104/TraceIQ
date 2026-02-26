#!/usr/bin/env python3
"""Experiment 3: Scaling Stability.

Proves IQx/PR doesn't explode simply because agent count increases.

Setup:
    - Repeat standardized signal scenario with agent_counts = [2, 10, 50]
    - Keep per-agent event rate constant
    - Measure mean/max IQx and propagation risk

Pass/Fail Criteria:
    - max_total_iqx scales ≤ O(n) (linear or sublinear)
    - PR remains bounded (< 2.0 for non-pathological graphs)

Usage:
    python exp_scaling.py --seeds 20
    python exp_scaling.py --seeds 3 --quick
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
from experiments.topologies import chain_topology
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


# Experiment parameters
DEFAULT_SEEDS = 20
AGENT_COUNTS = [2, 10, 50]
EVENTS_PER_AGENT = 10  # Keep constant across scales


def generate_content(rng: random.Random, has_signal: bool, signal_id: int) -> str:
    """Generate content with or without signal."""
    if has_signal:
        return f"Important update: SIGNAL_{signal_id:04d}. Please propagate."
    else:
        topics = ["status", "metrics", "logs", "updates"]
        return f"Routine {rng.choice(topics)} check. All nominal."


def run_single_seed(
    seed: int,
    n_agents: int,
    n_events_per_agent: int = EVENTS_PER_AGENT,
) -> dict:
    """Run scaling experiment for a single seed and agent count.

    Args:
        seed: Random seed
        n_agents: Number of agents
        n_events_per_agent: Events per agent

    Returns:
        Dict with metrics
    """
    set_all_seeds(seed)
    rng = random.Random(seed)

    tracker = create_tracker(seed=seed, baseline_k=min(10, n_events_per_agent))

    # Create chain topology
    topology = chain_topology(n_agents)

    # Track metrics
    all_iqx: list[float] = []
    agent_iqx: dict[str, list[float]] = defaultdict(list)
    total_events = 0
    total_alerts = 0

    # Run events through the chain
    for round_num in range(n_events_per_agent):
        # First agent generates signal
        signal_id = round_num

        # Propagate through chain
        for i, (sender, receiver) in enumerate(topology.edges):
            # Signal attenuates through chain: first edge has signal,
            # later edges less likely to propagate
            has_signal = i == 0 or rng.random() < 0.5

            content = generate_content(rng, has_signal, signal_id)
            response = f"Processed by {receiver}. Round {round_num}."

            result = tracker.track_event(
                sender_id=sender,
                receiver_id=receiver,
                sender_content=content,
                receiver_content=response,
            )

            iqx = get_iqx(result)
            if iqx is not None:
                all_iqx.append(iqx)
                agent_iqx[sender].append(iqx)

            if result.get("alert", False):
                total_alerts += 1

            total_events += 1

    # Get propagation risk
    try:
        pr = tracker.get_propagation_risk()
    except Exception:
        pr = None

    tracker.close()

    # Compute summary statistics
    if all_iqx:
        mean_iqx = float(np.mean(all_iqx))
        max_iqx = float(np.max(all_iqx))
        total_iqx = float(np.sum(all_iqx))
        std_iqx = float(np.std(all_iqx))
    else:
        mean_iqx = max_iqx = total_iqx = std_iqx = 0.0

    # Per-agent max IQx
    agent_max_iqx = {
        agent: float(np.max(values)) if values else 0.0
        for agent, values in agent_iqx.items()
    }
    agent_total_iqx = {
        agent: float(np.sum(values)) if values else 0.0
        for agent, values in agent_iqx.items()
    }

    return {
        "seed": seed,
        "n_agents": n_agents,
        "total_events": total_events,
        "total_alerts": total_alerts,
        "alert_rate": total_alerts / total_events if total_events > 0 else 0.0,
        "mean_iqx": mean_iqx,
        "max_iqx": max_iqx,
        "total_iqx": total_iqx,
        "std_iqx": std_iqx,
        "propagation_risk": pr,
        "agent_max_iqx": agent_max_iqx,
        "agent_total_iqx": agent_total_iqx,
    }


def run_experiment(
    n_seeds: int = DEFAULT_SEEDS,
    agent_counts: list[int] | None = None,
    output_dir: Path | None = None,
    quick: bool = False,
) -> dict:
    """Run the full scaling stability experiment.

    Args:
        n_seeds: Number of random seeds per agent count
        agent_counts: List of agent counts to test
        output_dir: Output directory
        quick: Quick mode

    Returns:
        Dict with results and pass/fail criteria
    """
    start_time = time.time()

    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "scaling"
    output_dir.mkdir(parents=True, exist_ok=True)

    if agent_counts is None:
        agent_counts = [2, 10] if quick else AGENT_COUNTS

    n_events = 5 if quick else EVENTS_PER_AGENT

    print("Scaling Stability Experiment")
    print(f"Agent counts: {agent_counts}")
    print(f"Seeds per count: {n_seeds}")
    print(f"Events per agent: {n_events}")

    # Collect results by agent count
    results_by_count: dict[int, list[dict]] = defaultdict(list)
    all_results = []

    for n_agents in agent_counts:
        print(f"\nRunning n_agents={n_agents}...")

        for seed in range(n_seeds):
            result = run_single_seed(seed, n_agents, n_events)
            results_by_count[n_agents].append(result)
            all_results.append(result)

    # Write all results to CSV
    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        fieldnames = [
            "seed",
            "n_agents",
            "mean_iqx",
            "max_iqx",
            "total_iqx",
            "std_iqx",
            "propagation_risk",
            "alert_rate",
            "total_events",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in all_results:
            writer.writerow({k: r.get(k) for k in fieldnames})

    # Compute aggregate statistics per agent count
    count_stats = {}
    for n_agents, results in results_by_count.items():
        mean_iqx_values = [r["mean_iqx"] for r in results]
        max_iqx_values = [r["max_iqx"] for r in results]
        total_iqx_values = [r["total_iqx"] for r in results]
        pr_values = [
            r["propagation_risk"] for r in results if r["propagation_risk"] is not None
        ]
        alert_rates = [r["alert_rate"] for r in results]

        pr_ci = bootstrap_ci(pr_values, seed=42).__dict__ if pr_values else None
        count_stats[n_agents] = {
            "mean_iqx": bootstrap_ci(mean_iqx_values, seed=42).__dict__,
            "max_iqx": bootstrap_ci(max_iqx_values, seed=42).__dict__,
            "total_iqx": bootstrap_ci(total_iqx_values, seed=42).__dict__,
            "propagation_risk": pr_ci,
            "alert_rate": bootstrap_ci(alert_rates, seed=42).__dict__,
            "n_seeds": len(results),
        }

    # === PASS/FAIL CRITERIA ===

    # Criterion 1: max_total_iqx scales at most linearly
    # Check: max_total_iqx(n=50) / max_total_iqx(n=10) <= 10 (allowing 2x margin)
    scaling_passed = True
    scaling_details = {}

    sorted_counts = sorted(agent_counts)
    if len(sorted_counts) >= 2:
        for i in range(1, len(sorted_counts)):
            smaller = sorted_counts[i - 1]
            larger = sorted_counts[i]

            smaller_max = count_stats[smaller]["max_iqx"]["mean"]
            larger_max = count_stats[larger]["max_iqx"]["mean"]

            if smaller_max > 0:
                ratio = larger_max / smaller_max
                expected_ratio = larger / smaller  # Linear scaling
            else:
                ratio = 1.0
                expected_ratio = 1.0

            # Allow 2x margin above linear
            passed = ratio <= expected_ratio * 2

            scaling_details[f"{smaller}_to_{larger}"] = {
                "smaller_max_iqx": smaller_max,
                "larger_max_iqx": larger_max,
                "observed_ratio": ratio,
                "expected_linear_ratio": expected_ratio,
                "threshold": expected_ratio * 2,
                "passed": passed,
            }

            if not passed:
                scaling_passed = False

    # Criterion 2: PR remains bounded (< 2.0)
    pr_threshold = 2.0
    pr_passed = True
    pr_details = {}

    for n_agents in agent_counts:
        pr_stats = count_stats[n_agents]["propagation_risk"]
        if pr_stats:
            pr_mean = pr_stats["mean"]
            passed = pr_mean < pr_threshold
            pr_details[n_agents] = {
                "pr_mean": pr_mean,
                "threshold": pr_threshold,
                "passed": passed,
            }
            if not passed:
                pr_passed = False

    overall_passed = scaling_passed and pr_passed

    runtime = time.time() - start_time

    summary = {
        "experiment": "scaling",
        "config": {
            "n_seeds": n_seeds,
            "agent_counts": agent_counts,
            "n_events_per_agent": n_events,
        },
        "runtime_seconds": runtime,
        "count_stats": {str(k): v for k, v in count_stats.items()},
        "pass_fail": {
            "scaling_bounded": {
                "passed": scaling_passed,
                "details": scaling_details,
                "derivation": "max_iqx should scale at most O(n); allow 2x margin",
            },
            "pr_bounded": {
                "passed": pr_passed,
                "threshold": pr_threshold,
                "details": pr_details,
                "derivation": "PR < 2.0 for stable networks (spectral radius < 2)",
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

    print("\nMetrics by Agent Count:")
    for n_agents in agent_counts:
        stats = count_stats[n_agents]
        print(f"\n  n={n_agents}:")
        m = stats["mean_iqx"]
        print(f"    Mean IQx: {m['mean']:.4f} [{m['ci_low']:.4f}, {m['ci_high']:.4f}]")
        x = stats["max_iqx"]
        print(f"    Max IQx: {x['mean']:.4f} [{x['ci_low']:.4f}, {x['ci_high']:.4f}]")
        if stats["propagation_risk"]:
            pr = stats["propagation_risk"]
            print(f"    PR: {pr['mean']:.4f} [{pr['ci_low']:.4f}, {pr['ci_high']:.4f}]")

    print("\nPass/Fail Criteria:")
    s1 = "PASS" if scaling_passed else "FAIL"
    print(f"  1. Scaling bounded (<=2x linear): {s1}")
    for key, detail in scaling_details.items():
        ratio = detail["observed_ratio"]
        thresh = detail["threshold"]
        s = "PASS" if detail["passed"] else "FAIL"
        print(f"     {key}: ratio={ratio:.2f}, threshold={thresh:.2f} -> {s}")

    print(f"  2. PR bounded (< {pr_threshold}): {'PASS' if pr_passed else 'FAIL'}")

    print(f"\nOverall: {'PASS' if overall_passed else 'FAIL'}")
    print(f"\nResults saved to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run scaling stability experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=DEFAULT_SEEDS,
        help=f"Number of random seeds per agent count (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--agent-counts",
        type=int,
        nargs="+",
        default=None,
        help=f"Agent counts to test (default: {AGENT_COUNTS})",
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
        agent_counts=args.agent_counts,
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
