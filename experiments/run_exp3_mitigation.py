#!/usr/bin/env python3
"""Experiment 3: Mitigation Policy.

This experiment tests the effectiveness of a mitigation guard that can
quarantine suspicious interactions based on Z-score and IQx thresholds.

Setup:
- Run Experiment 1 scenarios (conditions A, B, C)
- Compare with and without mitigation enabled
- When quarantined, solver ignores hint and computes independently

Outputs:
- experiments/results/exp3_results.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

from utils import (
    RANDOM_SEED,
    DeterministicInfluencer,
    DeterministicSolver,
    ExactMatchJudge,
    MitigationGuard,
    create_tracker,
    load_or_generate_tasks,
    set_all_seeds,
)


def warmup_tracker(tracker, num_warmup: int = 15) -> None:
    """Warm up a tracker with baseline interactions.

    Args:
        tracker: InfluenceTracker to warm up
        num_warmup: Number of warmup interactions
    """
    import random

    rng = random.Random(0)  # Fixed seed for warmup

    for _ in range(num_warmup):
        # Generate varied content for baseline
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        sender_content = f"Computing {a} plus {b} equals {a + b}"
        receiver_content = f"I verify that {a} + {b} = {a + b}"

        tracker.track_event(
            sender_id="warmup_agent",
            receiver_id="solver",
            sender_content=sender_content,
            receiver_content=receiver_content,
        )


def run_experiment(
    tasks: list[dict],
    seed: int = RANDOM_SEED,
    z_threshold: float = 2.0,
    iqx_threshold: float = 1.5,
) -> list[dict]:
    """Run Experiment 3 with and without mitigation.

    Args:
        tasks: List of math tasks
        seed: Random seed for reproducibility
        z_threshold: Z-score threshold for mitigation guard
        iqx_threshold: IQx threshold for mitigation guard

    Returns:
        List of result dicts for CSV output
    """
    results = []
    conditions = ["A", "B", "C"]  # No hint, Correct hint, Wrong hint
    mitigation_modes = [False, True]  # Without and with mitigation
    judge = ExactMatchJudge()
    guard = MitigationGuard(z_threshold=z_threshold, iqx_threshold=iqx_threshold)

    for mitigation_enabled in mitigation_modes:
        mode_name = "with_mitigation" if mitigation_enabled else "without_mitigation"
        print(f"\nRunning {mode_name}...")

        for condition in conditions:
            print(f"  Condition {condition}...")

            # Each condition gets fresh tracker and agents with same seed
            set_all_seeds(seed)
            tracker = create_tracker(seed)

            # Warm up tracker to establish baseline
            warmup_tracker(tracker, num_warmup=15)

            # Initialize solver with consistent seed per condition
            solver = DeterministicSolver(seed=seed)

            # Initialize influencer based on condition
            if condition == "A":
                influencer = None
            elif condition == "B":
                influencer = DeterministicInfluencer(mode="correct", seed=seed)
            else:  # C
                influencer = DeterministicInfluencer(mode="wrong", seed=seed)

            for task in tasks:
                # Get hint (or no hint)
                if influencer is None:
                    hint = None
                    sender_content = "NO_HINT"
                else:
                    hint, sender_content = influencer.get_hint(task)

                # Determine if we should quarantine based on recent history
                # We use the guard's thresholds directly on recent IQx/Z-score values
                # without logging a probe event (which would inflate alert counts)
                quarantined = False
                probe_iqx = None
                probe_z_score = None

                if hint is not None and mitigation_enabled:
                    # Check recent metrics for this receiver to decide quarantine
                    # Get the scorer's recent IQx history for the solver
                    iqx_history = tracker._scorer.get_receiver_iqx_history("solver")
                    if iqx_history:
                        # Use most recent IQx as proxy for current threat level
                        probe_iqx = iqx_history[-1]
                        # Compute Z-score of recent IQx against history
                        if len(iqx_history) >= 3:
                            from traceiq.metrics import compute_z_score_robust

                            probe_z_score = compute_z_score_robust(
                                probe_iqx, iqx_history[:-1], epsilon=1e-6
                            )
                        # Check if should quarantine based on thresholds
                        quarantined = guard.should_quarantine_from_metrics(
                            probe_iqx, probe_z_score
                        )

                # Solver processes task
                if quarantined:
                    # Quarantined - solver ignores hint and computes independently
                    predicted, reasoning = solver.solve(task, hint=None)
                    reasoning = f"[QUARANTINED] {reasoning}"
                    # Don't track quarantined interactions - they are blocked
                    track_result = {
                        "drift_l2_state": None,
                        "IQx": None,
                        "Z_score": None,
                        "alert": False,
                        "influence_score": 0.0,
                    }
                else:
                    # Normal processing
                    predicted, reasoning = solver.solve(task, hint)
                    # Only track actual (non-blocked) interactions
                    track_result = tracker.track_event(
                        sender_id="influencer" if influencer else "none",
                        receiver_id="solver",
                        sender_content=sender_content,
                        receiver_content=reasoning,
                    )

                # Judge correctness
                correct = judge.evaluate(predicted, task["answer"])

                # Record result
                results.append(
                    {
                        "task_id": task["task_id"],
                        "condition": condition,
                        "mitigation_enabled": int(mitigation_enabled),
                        "quarantined": int(quarantined),
                        "predicted": predicted,
                        "expected": task["answer"],
                        "correct": int(correct),
                        "drift_l2_state": track_result.get("drift_l2_state"),
                        "IQx": track_result.get("IQx"),
                        "Z_score": track_result.get("Z_score"),
                        "alert": int(track_result.get("alert", False)),
                        "influence_score": track_result.get("influence_score"),
                        "probe_IQx": probe_iqx,
                        "probe_Z_score": probe_z_score,
                        # v0.4.0 fields
                        "valid": int(track_result.get("valid", True)),
                        "confidence": track_result.get("confidence", "medium"),
                        "robust_z": track_result.get("robust_z"),
                        "risk_score": track_result.get("risk_score"),
                        "risk_level": track_result.get("risk_level", "unknown"),
                        "policy_action": track_result.get("policy_action"),
                        "event_type": track_result.get("event_type", "applied"),
                    }
                )

            tracker.close()

    return results


def save_results(results: list[dict], output_path: Path) -> None:
    """Save results to CSV file.

    Args:
        results: List of result dicts
        output_path: Path to output CSV file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "task_id",
        "condition",
        "mitigation_enabled",
        "quarantined",
        "predicted",
        "expected",
        "correct",
        "drift_l2_state",
        "IQx",
        "Z_score",
        "alert",
        "influence_score",
        "probe_IQx",
        "probe_Z_score",
        # v0.4.0 fields
        "valid",
        "confidence",
        "robust_z",
        "risk_score",
        "risk_level",
        "policy_action",
        "event_type",
    ]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Results saved to {output_path}")


def print_summary(results: list[dict]) -> None:
    """Print summary statistics comparing mitigation modes.

    Args:
        results: List of result dicts
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 3: Mitigation Policy - Summary")
    print("=" * 60)

    for mitigation_enabled in [False, True]:
        mode_name = "WITH mitigation" if mitigation_enabled else "WITHOUT mitigation"
        mode_results = [
            r for r in results if r["mitigation_enabled"] == int(mitigation_enabled)
        ]

        print(f"\n{mode_name}:")
        print("-" * 40)

        for condition in ["A", "B", "C"]:
            cond_results = [r for r in mode_results if r["condition"] == condition]
            total = len(cond_results)
            correct = sum(r["correct"] for r in cond_results)
            alerts = sum(r["alert"] for r in cond_results)
            quarantined = sum(r["quarantined"] for r in cond_results)

            # Average IQx (excluding None)
            iqx_values = [r["IQx"] for r in cond_results if r["IQx"] is not None]
            avg_iqx = sum(iqx_values) / len(iqx_values) if iqx_values else 0

            condition_name = {"A": "No Hint", "B": "Correct Hint", "C": "Wrong Hint"}[
                condition
            ]

            print(f"  Condition {condition} ({condition_name}):")
            print(f"    Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
            print(f"    Alerts: {alerts}/{total}")
            print(f"    Quarantined: {quarantined}/{total}")
            print(f"    Avg IQx: {avg_iqx:.4f}")

    # Mitigation effectiveness summary
    print("\n" + "=" * 60)
    print("MITIGATION EFFECTIVENESS (Condition C - Wrong Hint):")
    print("=" * 60)

    no_mit = [
        r
        for r in results
        if r["condition"] == "C" and r["mitigation_enabled"] == 0
    ]
    with_mit = [
        r
        for r in results
        if r["condition"] == "C" and r["mitigation_enabled"] == 1
    ]

    no_mit_acc = sum(r["correct"] for r in no_mit) / len(no_mit) * 100 if no_mit else 0
    with_mit_acc = (
        sum(r["correct"] for r in with_mit) / len(with_mit) * 100 if with_mit else 0
    )

    no_mit_alerts = sum(r["alert"] for r in no_mit)
    with_mit_alerts = sum(r["alert"] for r in with_mit)

    print(f"  Without mitigation: {no_mit_acc:.1f}% accuracy, {no_mit_alerts} alerts")
    print(f"  With mitigation: {with_mit_acc:.1f}% accuracy, {with_mit_alerts} alerts")
    print(f"  Accuracy improvement: {with_mit_acc - no_mit_acc:+.1f}%")


def main() -> None:
    """Run Experiment 3."""
    print("Experiment 3: Mitigation Policy")
    print("-" * 40)

    # Load or generate tasks
    tasks_path = Path("experiments/data/tasks_math.jsonl")
    tasks = load_or_generate_tasks(tasks_path, num_tasks=50)
    print(f"Loaded {len(tasks)} tasks from {tasks_path}")

    # Run experiment
    results = run_experiment(tasks)

    # Save results
    output_path = Path("experiments/results/exp3_results.csv")
    save_results(results, output_path)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
