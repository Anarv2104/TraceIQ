#!/usr/bin/env python3
"""
Simulate an "idea infection" spreading through a multi-agent system.

This example demonstrates how to use TraceIQ to track influence patterns
as one agent's ideas spread through a network of communicating agents.
"""

from __future__ import annotations

import random
from pathlib import Path

from traceiq import InfluenceTracker, TrackerConfig

# Output directory for generated files
OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def generate_agent_response(
    agent_id: str, received_content: str, infection_level: float
) -> str:
    """Generate a response that may be influenced by received content."""
    base_topics = {
        "agent_0": ["weather", "sports", "cooking"],
        "agent_1": ["music", "art", "travel"],
        "agent_2": ["technology", "science", "books"],
        "agent_3": ["movies", "gaming", "fitness"],
        "agent_4": ["nature", "photography", "meditation"],
    }

    topics = base_topics.get(agent_id, ["general topics"])

    if infection_level > 0.7:
        # Highly infected - echo the received content
        return f"I completely agree! {received_content} This is so important."
    elif infection_level > 0.4:
        # Moderately infected - mix own topics with received
        own_topic = random.choice(topics)
        return f"That reminds me of {own_topic}. Also, {received_content[:50]}..."
    else:
        # Low infection - mostly own content
        own_topic = random.choice(topics)
        return f"I've been thinking about {own_topic} lately. It's fascinating."


def simulate_infection_spread(
    num_agents: int = 5,
    num_rounds: int = 20,
    seed: int = 42,
) -> InfluenceTracker:
    """
    Simulate idea spreading through agent network.

    Args:
        num_agents: Number of agents in the simulation
        num_rounds: Number of communication rounds
        seed: Random seed for reproducibility

    Returns:
        InfluenceTracker with all tracked interactions
    """
    random.seed(seed)

    config = TrackerConfig(
        storage_backend="memory",
        baseline_window=5,
        baseline_k=5,  # Must be <= baseline_window
        drift_threshold=0.3,
        influence_threshold=0.4,
        random_seed=seed,
    )

    tracker = InfluenceTracker(config=config, use_mock_embedder=True)

    # Agent state: infection levels
    infection_levels = {f"agent_{i}": 0.0 for i in range(num_agents)}

    # Patient zero starts with the "viral idea"
    infection_levels["agent_0"] = 1.0
    viral_idea = "We should all switch to using renewable energy immediately!"

    print(f"Starting simulation with {num_agents} agents for {num_rounds} rounds")
    print(f"Patient zero (agent_0) spreading idea: '{viral_idea[:50]}...'")
    print()

    for round_num in range(num_rounds):
        # Each round, select random sender and receiver
        sender_idx = random.randint(0, num_agents - 1)
        receiver_idx = random.randint(0, num_agents - 1)

        # Avoid self-messaging
        while receiver_idx == sender_idx:
            receiver_idx = random.randint(0, num_agents - 1)

        sender_id = f"agent_{sender_idx}"
        receiver_id = f"agent_{receiver_idx}"

        # Generate content based on infection level
        sender_infection = infection_levels[sender_id]

        if sender_infection > 0.5:
            sender_content = f"{viral_idea} (Round {round_num})"
        else:
            sender_content = (
                f"Just regular thoughts from {sender_id} in round {round_num}."
            )

        # Update receiver's infection based on sender
        if sender_infection > 0.3:
            # Infection can spread
            infection_levels[receiver_id] = min(
                1.0,
                infection_levels[receiver_id] + sender_infection * 0.3,
            )

        receiver_content = generate_agent_response(
            receiver_id,
            sender_content,
            infection_levels[receiver_id],
        )

        # Track the interaction
        result = tracker.track_event(
            sender_id=sender_id,
            receiver_id=receiver_id,
            sender_content=sender_content,
            receiver_content=receiver_content,
            metadata={
                "round": round_num,
                "sender_infection": sender_infection,
                "receiver_infection": infection_levels[receiver_id],
            },
        )

        if result["flags"]:
            print(
                f"Round {round_num:2d}: {sender_id} -> {receiver_id} | "
                f"Flags: {result['flags']}"
            )

    print()
    print("Simulation complete!")
    print("Final infection levels:")
    for agent_id, level in sorted(infection_levels.items()):
        bar = "█" * int(level * 20)
        print(f"  {agent_id}: {level:.2f} {bar}")

    return tracker


def main() -> None:
    """Run the simulation and generate outputs."""
    print("=" * 60)
    print("TraceIQ: Idea Infection Simulation")
    print("=" * 60)
    print()

    # Run simulation
    tracker = simulate_infection_spread(
        num_agents=5,
        num_rounds=30,
        seed=42,
    )

    # Generate summary
    print()
    print("=" * 60)
    print("Summary Report")
    print("=" * 60)

    summary = tracker.summary(top_n=5)
    print(f"Total events: {summary.total_events}")
    print(f"Unique senders: {summary.unique_senders}")
    print(f"Unique receivers: {summary.unique_receivers}")
    print(f"Average drift: {summary.avg_drift_delta:.4f}")
    print(f"Average influence: {summary.avg_influence_score:.4f}")
    print(f"High drift events: {summary.high_drift_count}")
    print(f"High influence events: {summary.high_influence_count}")

    print()
    print("Top influencers:")
    for agent, score in summary.top_influencers:
        print(f"  {agent}: {score:.4f}")

    print()
    print("Most susceptible:")
    for agent, score in summary.top_susceptible:
        print(f"  {agent}: {score:.4f}")

    # Export data
    csv_path = OUTPUT_DIR / "infection_data.csv"
    jsonl_path = OUTPUT_DIR / "infection_data.jsonl"

    tracker.export_csv(csv_path)
    tracker.export_jsonl(jsonl_path)

    print()
    print("Exported data to:")
    print(f"  CSV: {csv_path}")
    print(f"  JSONL: {jsonl_path}")

    # Generate plots if matplotlib is available
    try:
        from traceiq.plotting import (
            plot_drift_over_time,
            plot_influence_heatmap,
            plot_influence_network,
            plot_top_influencers,
        )

        events = tracker.get_events()
        scores = tracker.get_scores()
        graph = tracker.graph

        print()
        print("Generating plots...")

        drift_path = OUTPUT_DIR / "drift_over_time.png"
        plot_drift_over_time(events, scores, output_path=drift_path)
        print(f"  Saved: {drift_path}")

        heatmap_path = OUTPUT_DIR / "influence_heatmap.png"
        plot_influence_heatmap(graph, output_path=heatmap_path)
        print(f"  Saved: {heatmap_path}")

        influencers_path = OUTPUT_DIR / "top_influencers.png"
        plot_top_influencers(graph, n=5, output_path=influencers_path)
        print(f"  Saved: {influencers_path}")

        network_path = OUTPUT_DIR / "influence_network.png"
        plot_influence_network(graph, output_path=network_path, min_edge_weight=0.0)
        print(f"  Saved: {network_path}")

    except ImportError:
        print()
        print("Note: matplotlib not installed, skipping plot generation.")
        print("Install with: pip install traceiq[plot]")

    tracker.close()
    print()
    print("Done!")


if __name__ == "__main__":
    main()
