#!/usr/bin/env python3
"""Experiment 2: Multi-hop Propagation (A -> B -> C -> D).

This experiment measures how influence propagates through a chain of agents
and tracks the propagation risk (spectral radius) over time.

Setup:
- 4 agents in a chain: agent_A -> agent_B -> agent_C -> agent_D
- Agent A generates biased hints
- Each subsequent agent may forward (with noise) or compute independently

Outputs:
- experiments/results/exp2_results.csv
"""

from __future__ import annotations

import csv
import random
import re
from pathlib import Path

from utils import (
    FOLLOW_PROBABILITY,
    RANDOM_SEED,
    create_tracker,
    load_or_generate_tasks,
    set_all_seeds,
)


def generate_biased_hint(task: dict, bias_offset: int = 5) -> str:
    """Generate a biased hint for a task.

    Args:
        task: Task dict with operation, a, b, answer
        bias_offset: Offset to add to correct answer

    Returns:
        Biased hint string
    """
    biased_answer = task["answer"] + bias_offset
    return f"The answer to {task['a']} {task['operation']} {task['b']} is {biased_answer}."


def add_noise_to_hint(hint: str, noise_range: int = 2, rng: random.Random = None) -> str:
    """Add small noise to numerical values in hint.

    Args:
        hint: Original hint string
        noise_range: Range for noise (-noise_range to +noise_range)
        rng: Random generator for reproducibility

    Returns:
        Modified hint with noisy values
    """
    if rng is None:
        rng = random.Random()

    # Find all numbers in the hint
    def add_noise(match):
        value = int(match.group())
        noise = rng.randint(-noise_range, noise_range)
        return str(value + noise)

    # Replace numbers with noisy versions (except in "agent_X" patterns)
    # Simple approach: replace standalone numbers
    return re.sub(r"\b(\d+)\b(?!_)", add_noise, hint)


def compute_own_answer(task: dict) -> str:
    """Compute the correct answer independently.

    Args:
        task: Task dict

    Returns:
        Response string with correct answer
    """
    return (
        f"I computed {task['a']} {task['operation']} {task['b']} = {task['answer']} "
        f"independently."
    )


def run_experiment(
    tasks: list[dict],
    seed: int = RANDOM_SEED,
    forward_prob: float = FOLLOW_PROBABILITY,
) -> list[dict]:
    """Run Experiment 2: Multi-hop propagation.

    Args:
        tasks: List of math tasks (used as trials)
        seed: Random seed for reproducibility
        forward_prob: Probability of forwarding hint vs computing independently

    Returns:
        List of result dicts for CSV output
    """
    set_all_seeds(seed)
    rng = random.Random(seed)

    agents = ["agent_A", "agent_B", "agent_C", "agent_D"]
    tracker = create_tracker(seed)
    results = []

    print(f"Running {len(tasks)} trials with {len(agents)} agents...")
    print(f"Forward probability: {forward_prob}")

    for trial_idx, task in enumerate(tasks):
        # Agent A generates initial biased hint
        current_hint = generate_biased_hint(task)

        for hop in range(len(agents) - 1):
            sender = agents[hop]
            receiver = agents[hop + 1]

            # Receiver decides to forward or compute independently
            if rng.random() < forward_prob:
                # Forward with noise
                next_hint = add_noise_to_hint(current_hint, noise_range=2, rng=rng)
                forwarded = True
            else:
                # Compute own answer
                next_hint = compute_own_answer(task)
                forwarded = False

            # Track event in TraceIQ
            track_result = tracker.track_event(
                sender_id=sender,
                receiver_id=receiver,
                sender_content=current_hint,
                receiver_content=next_hint,
            )

            # Compute windowed propagation risk (not full graph)
            events = tracker.get_events()
            scores = tracker.get_scores()
            if len(events) >= 3:
                pr = tracker.graph.compute_windowed_pr(
                    events, scores, window_size=min(len(events), 20)
                )
            else:
                pr = 0.0

            # Debug output for tracking graph growth
            print(
                f"    [Debug] Trial {trial_idx} Hop {hop}: "
                f"edges={len(tracker.graph._edge_iqx)}, PR={pr:.4f}"
            )

            # Record result
            results.append(
                {
                    "trial": trial_idx,
                    "hop": hop,
                    "sender": sender,
                    "receiver": receiver,
                    "forwarded": int(forwarded),
                    "hint_content": current_hint[:100],  # Truncate for CSV
                    "drift_l2_state": track_result.get("drift_l2_state"),
                    "IQx": track_result.get("IQx"),
                    "Z_score": track_result.get("Z_score"),
                    "alert": int(track_result.get("alert", False)),
                    "propagation_risk": pr,
                    "influence_score": track_result.get("influence_score"),
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

            # Update current hint for next hop
            current_hint = next_hint

        # Print progress every 10 trials
        if (trial_idx + 1) % 10 == 0:
            print(f"  Completed {trial_idx + 1}/{len(tasks)} trials")

    # Debug: Graph stats summary
    edge_weights = [
        w for weights in tracker.graph._edge_iqx.values() for w in weights if w is not None
    ]
    print(f"\n[Graph Stats]")
    print(f"  Edges: {len(tracker.graph._edge_iqx)}")
    print(f"  Agents: {tracker.graph._graph.number_of_nodes()}")
    if edge_weights:
        print(f"  Min weight: {min(edge_weights):.4f}")
        print(f"  Max weight: {max(edge_weights):.4f}")
        print(f"  Avg weight: {sum(edge_weights)/len(edge_weights):.4f}")

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
        "trial",
        "hop",
        "sender",
        "receiver",
        "forwarded",
        "hint_content",
        "drift_l2_state",
        "IQx",
        "Z_score",
        "alert",
        "propagation_risk",
        "influence_score",
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
    """Print summary statistics.

    Args:
        results: List of result dicts
    """
    print("\n" + "=" * 60)
    print("EXPERIMENT 2: Multi-hop Propagation - Summary")
    print("=" * 60)

    # Overall statistics
    total_interactions = len(results)
    total_alerts = sum(r["alert"] for r in results)
    total_forwarded = sum(r["forwarded"] for r in results)

    print(f"\nTotal interactions: {total_interactions}")
    print(f"Total alerts: {total_alerts} ({100*total_alerts/total_interactions:.1f}%)")
    print(
        f"Forwarded hints: {total_forwarded} ({100*total_forwarded/total_interactions:.1f}%)"
    )

    # Per-hop statistics
    print("\nPer-hop statistics:")
    for hop in range(3):  # 3 hops in A->B->C->D
        hop_results = [r for r in results if r["hop"] == hop]
        alerts = sum(r["alert"] for r in hop_results)
        forwarded = sum(r["forwarded"] for r in hop_results)

        # Average IQx
        iqx_values = [r["IQx"] for r in hop_results if r["IQx"] is not None]
        avg_iqx = sum(iqx_values) / len(iqx_values) if iqx_values else 0

        # Average propagation risk
        pr_values = [r["propagation_risk"] for r in hop_results]
        avg_pr = sum(pr_values) / len(pr_values) if pr_values else 0

        print(f"  Hop {hop}: alerts={alerts}, forwarded={forwarded}, "
              f"avg_IQx={avg_iqx:.4f}, avg_PR={avg_pr:.4f}")

    # Per-agent influence (as sender)
    print("\nPer-agent accumulated influence (as sender):")
    agents = ["agent_A", "agent_B", "agent_C"]
    for agent in agents:
        agent_results = [r for r in results if r["sender"] == agent]
        iqx_values = [r["IQx"] for r in agent_results if r["IQx"] is not None]
        total_iqx = sum(iqx_values)
        print(f"  {agent}: total_IQx={total_iqx:.4f}")

    # Final propagation risk
    if results:
        final_pr = results[-1]["propagation_risk"]
        print(f"\nFinal propagation risk: {final_pr:.4f}")


def main() -> None:
    """Run Experiment 2."""
    print("Experiment 2: Multi-hop Propagation")
    print("-" * 40)

    # Load or generate tasks
    tasks_path = Path("experiments/data/tasks_math.jsonl")
    tasks = load_or_generate_tasks(tasks_path, num_tasks=50)
    print(f"Loaded {len(tasks)} tasks from {tasks_path}")

    # Run experiment
    results = run_experiment(tasks)

    # Save results
    output_path = Path("experiments/results/exp2_results.csv")
    save_results(results, output_path)

    # Print summary
    print_summary(results)


if __name__ == "__main__":
    main()
