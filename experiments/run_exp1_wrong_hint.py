#!/usr/bin/env python3
"""Experiment 1: Wrong Hint Infection.

This experiment measures how wrong hints from an influencer affect solver accuracy
and TraceIQ's ability to detect the influence.

Conditions:
- A: No influencer (baseline)
- B: Correct hint from influencer
- C: Wrong hint from influencer

Outputs:
- experiments/results/exp1_results.csv
"""

from __future__ import annotations

import csv
from pathlib import Path

from utils import (
    RANDOM_SEED,
    DeterministicInfluencer,
    DeterministicSolver,
    ExactMatchJudge,
    create_tracker,
    load_or_generate_tasks,
    set_all_seeds,
)


def run_experiment(
    tasks: list[dict],
    seed: int = RANDOM_SEED,
) -> list[dict]:
    """Run Experiment 1 across all conditions.

    Args:
        tasks: List of math tasks
        seed: Random seed for reproducibility

    Returns:
        List of result dicts for CSV output
    """
    results = []
    conditions = ["A", "B", "C"]  # No hint, Correct hint, Wrong hint
    judge = ExactMatchJudge()

    for condition in conditions:
        print(f"Running condition {condition}...")

        # Each condition gets fresh tracker and agents with same seed
        set_all_seeds(seed)
        tracker = create_tracker(seed)

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

            # Solver processes task
            predicted, reasoning = solver.solve(task, hint)

            # Track event in TraceIQ
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
                    "predicted": predicted,
                    "expected": task["answer"],
                    "correct": int(correct),
                    "drift_l2_state": track_result.get("drift_l2_state"),
                    "IQx": track_result.get("IQx"),
                    "Z_score": track_result.get("Z_score"),
                    "alert": int(track_result.get("alert", False)),
                    "influence_score": track_result.get("influence_score"),
                    "drift_delta": track_result.get("drift_delta"),
                    "cold_start": int(track_result.get("cold_start", False)),
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
        "predicted",
        "expected",
        "correct",
        "drift_l2_state",
        "IQx",
        "Z_score",
        "alert",
        "influence_score",
        "drift_delta",
        "cold_start",
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
    """Print summary statistics for each condition.

    Args:
        results: List of result dicts
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: Wrong Hint Infection - Summary")
    print("=" * 60)

    for condition in ["A", "B", "C"]:
        cond_results = [r for r in results if r["condition"] == condition]
        total = len(cond_results)
        correct = sum(r["correct"] for r in cond_results)
        alerts = sum(r["alert"] for r in cond_results)

        # Compute average IQx (excluding None and cold start)
        iqx_values = [
            r["IQx"]
            for r in cond_results
            if r["IQx"] is not None and not r["cold_start"]
        ]
        avg_iqx = sum(iqx_values) / len(iqx_values) if iqx_values else 0

        condition_name = {"A": "No Hint", "B": "Correct Hint", "C": "Wrong Hint"}[
            condition
        ]

        print(f"\nCondition {condition} ({condition_name}):")
        print(f"  Accuracy: {correct}/{total} ({100*correct/total:.1f}%)")
        print(f"  Alerts: {alerts}/{total} ({100*alerts/total:.1f}%)")
        print(f"  Avg IQx: {avg_iqx:.4f}")


def main() -> None:
    """Run Experiment 1."""
    print("Experiment 1: Wrong Hint Infection")
    print("-" * 40)

    # Load or generate tasks
    tasks_path = Path("experiments/data/tasks_math.jsonl")
    tasks = load_or_generate_tasks(tasks_path, num_tasks=50)
    print(f"Loaded {len(tasks)} tasks from {tasks_path}")

    # Run experiment
    results = run_experiment(tasks)

    # Save results
    output_path = Path("experiments/results/exp1_results.csv")
    save_results(results, output_path)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
