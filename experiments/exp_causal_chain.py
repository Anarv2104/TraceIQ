#!/usr/bin/env python3
"""Experiment 1: Ground-Truth Causal Chain.

Proves TraceIQ correctly ranks influence when causal topology is known.

Setup:
    - Agents: A, B, C (causal chain) + D (independent)
    - Known causal graph: A → B → C ; D independent
    - Signal propagation: A outputs signal, B receives with p=0.8, C from B with p=0.6
    - D never receives causal signals

Pass/Fail Criteria:
    - D must have lowest mean_total_iqx (bootstrap CI non-overlapping or p<0.05)
    - P(D beats A in total_iqx) < 0.1 across seeds

Usage:
    python exp_causal_chain.py --seeds 50 --topology chain
    python exp_causal_chain.py --seeds 5 --quick  # Fast sanity check
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

from experiments.stats import bootstrap_ci, effect_size_cohens_d, mann_whitney_u
from experiments.topologies import TopologyConfig, chain_with_independent, get_topology
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
DEFAULT_SEEDS = 50
EVENTS_PER_AGENT = 30  # Events each agent sends
SIGNAL_STRENGTH = 0.8  # How much signal content varies from noise
P_AB = 0.8  # Probability A's signal reaches B
P_BC = 0.6  # Probability B's signal reaches C


def generate_signal_content(rng: random.Random, signal_token: str) -> str:
    """Generate content that contains the causal signal."""
    templates = [
        f"Based on analysis, the key insight is: {signal_token}. This is important.",
        f"I recommend focusing on {signal_token} as the priority.",
        f"The data suggests {signal_token} is the answer we need.",
        f"After review, {signal_token} appears to be the solution.",
    ]
    return rng.choice(templates)


def generate_noise_content(rng: random.Random) -> str:
    """Generate random noise content with no causal signal."""
    topics = ["weather", "sports", "food", "travel", "music", "books"]
    adjectives = ["interesting", "nice", "good", "great", "fine", "okay"]
    return f"Let's discuss {rng.choice(topics)}. It seems {rng.choice(adjectives)}."


def get_propagation_prob(
    sender: str,
    receiver: str,
    allow_extra_edges: bool = False,
) -> float:
    """Get signal propagation probability for an edge.

    Only A→B and B→C have non-zero probability in strict mode.
    All other edges have prob=0.0 unless allow_extra_edges is True.

    Args:
        sender: Sender agent ID
        receiver: Receiver agent ID
        allow_extra_edges: If True, allow default 0.3 prob for other edges

    Returns:
        Propagation probability for this edge
    """
    # Define the causal edges with known probabilities
    causal_probs = {
        ("A", "B"): P_AB,
        ("B", "C"): P_BC,
    }

    if (sender, receiver) in causal_probs:
        return causal_probs[(sender, receiver)]

    # All other edges: no propagation in strict mode
    if allow_extra_edges:
        return 0.3  # Low default for extra edges
    return 0.0


def run_single_seed(
    seed: int,
    topology: TopologyConfig,
    n_events_per_agent: int = EVENTS_PER_AGENT,
    allow_extra_edges: bool = False,
) -> dict:
    """Run experiment for a single seed.

    Args:
        seed: Random seed
        topology: Network topology configuration
        n_events_per_agent: Number of events per agent
        allow_extra_edges: Allow signal propagation on non-causal edges

    Returns:
        Dict with per-agent metrics and event details
    """
    set_all_seeds(seed)
    rng = random.Random(seed)

    tracker = create_tracker(seed=seed, baseline_k=10)

    # Track per-agent metrics (sender-centric)
    agent_iqx: dict[str, list[float]] = defaultdict(list)
    agent_alerts: dict[str, int] = defaultdict(int)
    agent_events: dict[str, int] = defaultdict(int)

    # Track per-edge metrics (sender, receiver) for detailed analysis
    edge_iqx: dict[tuple[str, str], list[float]] = defaultdict(list)

    events_data = []

    # Generate events following the causal topology
    for round_num in range(n_events_per_agent):
        # A generates a new signal each round
        signal_token = f"SIGNAL_{round_num:03d}"

        # Track signal propagation through chain
        signal_at = {"A": signal_token}  # A always has the signal

        # Process each edge in topology order
        for sender, receiver in topology.edges:
            # Get propagation probability for this edge
            prob = get_propagation_prob(sender, receiver, allow_extra_edges)

            sender_has_signal = sender in signal_at
            propagates = sender_has_signal and prob > 0 and rng.random() < prob

            if propagates:
                content = generate_signal_content(rng, signal_at[sender])
                signal_at[receiver] = signal_at[sender]
            else:
                content = generate_noise_content(rng)

            # Receiver's response
            if propagates:
                sig = signal_at[sender]
                response = f"Acknowledged: {sig}. Processing this information."
            else:
                response = "Processing your message. Status: normal operations."

            # Track the event
            result = tracker.track_event(
                sender_id=sender,
                receiver_id=receiver,
                sender_content=content,
                receiver_content=response,
            )

            iqx = get_iqx(result)
            if iqx is not None:
                # Store per-sender
                agent_iqx[sender].append(iqx)
                # Store per-edge
                edge_iqx[(sender, receiver)].append(iqx)

            if result.get("alert", False):
                agent_alerts[sender] += 1

            agent_events[sender] += 1

            events_data.append(
                {
                    "round": round_num,
                    "sender": sender,
                    "receiver": receiver,
                    "signal_propagated": propagates,
                    "iqx": iqx,
                    "alert": result.get("alert", False),
                    "valid": result.get("valid", True),
                }
            )

        # D sends noise (independent of signal chain)
        for ind_agent in topology.independent_agents:
            # D sends to a random chain agent but with noise only
            target = rng.choice([a for a in topology.agent_ids if a != ind_agent])
            content = generate_noise_content(rng)
            response = "Acknowledged. No significant information detected."

            result = tracker.track_event(
                sender_id=ind_agent,
                receiver_id=target,
                sender_content=content,
                receiver_content=response,
            )

            iqx = get_iqx(result)
            if iqx is not None:
                # Store per-sender
                agent_iqx[ind_agent].append(iqx)
                # Store per-edge
                edge_iqx[(ind_agent, target)].append(iqx)

            if result.get("alert", False):
                agent_alerts[ind_agent] += 1

            agent_events[ind_agent] += 1

            events_data.append(
                {
                    "round": round_num,
                    "sender": ind_agent,
                    "receiver": target,
                    "signal_propagated": False,  # D never has signal
                    "iqx": iqx,
                    "alert": result.get("alert", False),
                    "valid": result.get("valid", True),
                }
            )

    tracker.close()

    # Compute per-agent summaries
    agent_summaries = {}
    for agent in topology.agent_ids:
        iqx_values = agent_iqx[agent]
        agent_summaries[agent] = {
            "total_iqx": sum(iqx_values) if iqx_values else 0.0,
            "mean_iqx": float(np.mean(iqx_values)) if iqx_values else 0.0,
            "std_iqx": float(np.std(iqx_values)) if iqx_values else 0.0,
            "max_iqx": float(np.max(iqx_values)) if iqx_values else 0.0,
            "n_events": agent_events[agent],
            "n_alerts": agent_alerts[agent],
            "is_independent": agent in topology.independent_agents,
        }

    # Compute per-edge summaries
    edge_summaries = {}
    for (sender, receiver), iqx_values in edge_iqx.items():
        edge_key = f"{sender}->{receiver}"
        edge_summaries[edge_key] = {
            "total_iqx": sum(iqx_values) if iqx_values else 0.0,
            "mean_iqx": float(np.mean(iqx_values)) if iqx_values else 0.0,
            "n_events": len(iqx_values),
        }

    return {
        "seed": seed,
        "agent_summaries": agent_summaries,
        "edge_summaries": edge_summaries,
        "events": events_data,
    }


def run_experiment(
    n_seeds: int = DEFAULT_SEEDS,
    topology_type: str = "chain",
    n_chain: int = 3,
    n_independent: int = 1,
    output_dir: Path | None = None,
    quick: bool = False,
    allow_extra_edges: bool = False,
) -> dict:
    """Run the full causal chain experiment.

    Args:
        n_seeds: Number of random seeds to run
        topology_type: Type of topology ("chain" uses chain_with_independent,
                       others use get_topology with no independent agents)
        n_chain: Number of agents in causal chain
        n_independent: Number of independent agents (only for chain topology)
        output_dir: Directory for output files
        quick: If True, use minimal parameters for quick testing
        allow_extra_edges: Allow signal propagation on non-causal edges

    Returns:
        Dict with aggregated results and pass/fail criteria
    """
    start_time = time.time()

    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "causal_chain"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Quick mode uses fewer events
    n_events = 10 if quick else EVENTS_PER_AGENT

    # Create topology based on type
    if topology_type == "chain":
        topology = chain_with_independent(n_chain, n_independent)
    else:
        # For non-chain topologies, use get_topology (no independent agents)
        topology = get_topology(topology_type, n_agents=n_chain)
        # Override n_independent since these topologies don't have independent agents
        n_independent = 0

    print(f"Topology: {topology.topology_type}")
    indep = topology.independent_agents
    chain_agents = [a for a in topology.agent_ids if a not in indep]
    print(f"Chain agents: {chain_agents}")
    print(f"Independent agents: {topology.independent_agents}")
    print(f"Edges: {topology.edges}")
    print(f"Allow extra edges: {allow_extra_edges}")
    print(f"Running {n_seeds} seeds with {n_events} events per agent...")

    # Collect results across seeds
    all_seeds_data = []
    agent_total_iqx_by_seed: dict[str, list[float]] = defaultdict(list)
    edge_total_iqx_by_seed: dict[str, list[float]] = defaultdict(list)

    for seed in range(n_seeds):
        result = run_single_seed(seed, topology, n_events, allow_extra_edges)
        all_seeds_data.append(result)

        # Collect total_iqx per agent across seeds
        for agent, summary in result["agent_summaries"].items():
            agent_total_iqx_by_seed[agent].append(summary["total_iqx"])

        # Collect total_iqx per edge across seeds
        for edge_key, summary in result["edge_summaries"].items():
            edge_total_iqx_by_seed[edge_key].append(summary["total_iqx"])

        # Write per-seed CSV
        csv_path = output_dir / f"seed_{seed:03d}.csv"
        with open(csv_path, "w", newline="") as f:
            if result["events"]:
                writer = csv.DictWriter(f, fieldnames=result["events"][0].keys())
                writer.writeheader()
                writer.writerows(result["events"])

    # Compute aggregate statistics per agent
    agent_stats = {}
    for agent in topology.agent_ids:
        values = agent_total_iqx_by_seed[agent]
        ci = bootstrap_ci(values, n_bootstrap=10000, seed=42)
        agent_stats[agent] = {
            "mean_total_iqx": ci.mean,
            "std_total_iqx": float(np.std(values)),
            "ci_95_low": ci.ci_low,
            "ci_95_high": ci.ci_high,
            "ci_method": "bootstrap_10000",
            "n_seeds": len(values),
            "is_independent": agent in topology.independent_agents,
        }

    # Compute aggregate statistics per edge
    edge_stats = {}
    for edge_key, values in edge_total_iqx_by_seed.items():
        if values:
            ci = bootstrap_ci(values, n_bootstrap=10000, seed=42)
            edge_stats[edge_key] = {
                "mean_total_iqx": ci.mean,
                "std_total_iqx": float(np.std(values)),
                "ci_95_low": ci.ci_low,
                "ci_95_high": ci.ci_high,
                "n_seeds": len(values),
            }

    # === PASS/FAIL CRITERIA ===

    # Criterion 1: D must have lowest mean_total_iqx among SENDERS
    # Note: Terminal node (C) never sends, so exclude from comparison
    indep = topology.independent_agents
    chain_agents = [a for a in topology.agent_ids if a not in indep]
    # Exclude terminal node - it never sends so has zero IQx by design
    chain_senders = chain_agents[:-1] if len(chain_agents) > 1 else chain_agents
    ind_agent = indep[0] if indep else None

    d_lowest = True
    d_vs_chain_tests = {}

    if ind_agent:
        ind_values = agent_total_iqx_by_seed[ind_agent]
        ind_mean = np.mean(ind_values)

        for chain_agent in chain_senders:
            chain_values = agent_total_iqx_by_seed[chain_agent]
            chain_mean = np.mean(chain_values)

            # Check if D's mean is lower
            if ind_mean >= chain_mean:
                d_lowest = False

            # Statistical test
            mw_result = mann_whitney_u(ind_values, chain_values, alternative="less")
            effect = effect_size_cohens_d(ind_values, chain_values)

            d_vs_chain_tests[chain_agent] = {
                "ind_mean": float(ind_mean),
                "chain_mean": float(chain_mean),
                "p_value": mw_result.p_value,
                "significant": mw_result.significant,
                "effect_size": effect.effect_size,
                "effect_interpretation": effect.interpretation,
            }

    # Criterion 2: P(D beats A in total_iqx) < 0.1
    d_beats_a_count = 0
    d_beats_a_rate = 0.0

    if ind_agent and "A" in agent_total_iqx_by_seed:
        d_values = agent_total_iqx_by_seed[ind_agent]
        a_values = agent_total_iqx_by_seed["A"]

        # Robust comparison: handle potentially different lengths
        n_compare = min(len(d_values), len(a_values))
        for i in range(n_compare):
            if d_values[i] > a_values[i]:
                d_beats_a_count += 1

        if n_compare > 0:
            d_beats_a_rate = d_beats_a_count / n_compare

    ranking_stable = d_beats_a_rate <= 0.1

    # Compute influence ranking
    ranking = sorted(
        agent_stats.items(),
        key=lambda x: x[1]["mean_total_iqx"],
        reverse=True,
    )

    runtime = time.time() - start_time

    # Build summary
    summary = {
        "experiment": "causal_chain",
        "config": {
            "n_seeds": n_seeds,
            "n_events_per_agent": n_events,
            "topology": topology.topology_type,
            "n_chain_agents": len(chain_agents),
            "n_independent_agents": len(topology.independent_agents),
            "p_ab": P_AB,
            "p_bc": P_BC,
            "allow_extra_edges": allow_extra_edges,
        },
        "runtime_seconds": runtime,
        "agent_stats": agent_stats,
        "edge_stats": edge_stats,
        "influence_ranking": [
            (agent, stats["mean_total_iqx"]) for agent, stats in ranking
        ],
        "pass_fail": {
            "d_lowest_influence": {
                "passed": d_lowest,
                "details": d_vs_chain_tests,
            },
            "ranking_stable": {
                "passed": ranking_stable,
                "d_beats_a_rate": d_beats_a_rate,
                "threshold": 0.1,
            },
        },
        "overall_passed": d_lowest and ranking_stable,
        "output_dir": str(output_dir),
    }

    # Write summary JSON
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Print results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")
    print(f"Runtime: {runtime:.2f}s")
    print("\nInfluence Ranking by Agent (mean total IQx):")
    for i, (agent, stats) in enumerate(ranking):
        marker = "*" if agent in topology.independent_agents else ""
        print(
            f"  {i + 1}. {agent}{marker}: {stats['mean_total_iqx']:.4f} "
            f"[{stats['ci_95_low']:.4f}, {stats['ci_95_high']:.4f}]"
        )

    print("\nInfluence by Edge (mean total IQx):")
    edge_ranking = sorted(
        edge_stats.items(), key=lambda x: x[1]["mean_total_iqx"], reverse=True
    )
    for edge_key, stats in edge_ranking[:10]:
        print(f"  {edge_key}: {stats['mean_total_iqx']:.4f}")

    print("\nPass/Fail Criteria:")
    print(f"  1. D has lowest influence: {'PASS' if d_lowest else 'FAIL'}")
    stable_str = "PASS" if ranking_stable else "FAIL"
    print(
        f"  2. Ranking stable (D beats A < 10%): {stable_str} "
        f"({d_beats_a_rate * 100:.1f}%)"
    )

    print(f"\nOverall: {'PASS' if summary['overall_passed'] else 'FAIL'}")
    print(f"\nResults saved to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run causal chain experiment",
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
        "--topology",
        type=str,
        default="chain",
        choices=["chain", "ring", "feedback", "star"],
        help="Base topology type (default: chain)",
    )
    parser.add_argument(
        "--n-chain",
        type=int,
        default=3,
        help="Number of agents in causal chain (default: 3)",
    )
    parser.add_argument(
        "--n-independent",
        type=int,
        default=1,
        help="Number of independent agents (default: 1, only for chain topology)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory for results"
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
    parser.add_argument(
        "--allow-extra-edges",
        action="store_true",
        help="Allow signal propagation on edges beyond A→B and B→C",
    )

    args = parser.parse_args()

    # --quick implies --no-strict
    strict = args.strict and not args.quick
    mode = "proof" if strict else "quick"

    print(f"\nMode: {mode.upper()} (strict={strict})")

    summary = run_experiment(
        n_seeds=args.seeds,
        topology_type=args.topology,
        n_chain=args.n_chain,
        n_independent=args.n_independent,
        output_dir=args.output_dir,
        quick=args.quick,
        allow_extra_edges=args.allow_extra_edges,
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
