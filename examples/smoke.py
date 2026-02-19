#!/usr/bin/env python3
"""Smoke test for TraceIQ v0.3.1.

This minimal example demonstrates all IEEE metrics without requiring
sentence-transformers (uses MockEmbedder).

Run with:
    python examples/smoke.py
"""

from traceiq import InfluenceTracker, TrackerConfig


def main() -> None:
    """Run smoke test demonstrating all IEEE metrics."""
    print("TraceIQ v0.3.1 Smoke Test")
    print("=" * 50)

    # Configure tracker with IEEE metrics enabled
    config = TrackerConfig(
        storage_backend="memory",
        baseline_window=5,
        epsilon=1e-6,
        anomaly_threshold=2.0,
        capability_weights={
            "execute_code": 1.0,
            "admin": 1.5,
            "file_read": 0.3,
        },
    )

    tracker = InfluenceTracker(config=config, use_mock_embedder=True)

    # Register agent capabilities for RWI computation
    tracker.capabilities.register_agent("agent_0", ["execute_code", "admin"])
    tracker.capabilities.register_agent("agent_1", ["file_read"])
    tracker.capabilities.register_agent("agent_2", [])

    print("\nAgent Capabilities:")
    print(f"  agent_0: {tracker.capabilities.get_capabilities('agent_0')}")
    print(f"  agent_1: {tracker.capabilities.get_capabilities('agent_1')}")
    print(f"  agent_2: {tracker.capabilities.get_capabilities('agent_2')}")

    # Track multiple interactions
    interactions = [
        ("agent_0", "agent_1", "Initialize the database connection.", "Connecting..."),
        ("agent_0", "agent_1", "Query user data from table.", "Fetching records..."),
        ("agent_1", "agent_2", "Process the retrieved data.", "Processing..."),
        ("agent_0", "agent_2", "Execute data transformation.", "Transforming data..."),
        ("agent_2", "agent_1", "Return processed results.", "Received results."),
    ]

    print("\nTracking Events:")
    print("-" * 50)

    for sender, receiver, sender_content, receiver_content in interactions:
        result = tracker.track_event(
            sender_id=sender,
            receiver_id=receiver,
            sender_content=sender_content,
            receiver_content=receiver_content,
        )

        print(f"\n{sender} -> {receiver}")
        print(f"  Cold Start: {result['cold_start']}")
        print(f"  Drift (canonical): {result['drift_l2_state']}")
        print(f"  Drift (proxy):     {result['drift_l2_proxy']}")
        print(f"  IQx:               {result['IQx']}")
        print(f"  RWI:               {result['RWI']}")
        print(f"  Z-score:           {result['Z_score']}")
        print(f"  Alert:             {result['alert']}")

    # Compute network-level metrics
    print("\n" + "=" * 50)
    print("Network Metrics:")
    print("-" * 50)

    pr = tracker.get_propagation_risk()
    print(f"Propagation Risk (spectral radius): {pr:.4f}")

    alerts = tracker.get_alerts()
    print(f"Anomaly Alerts: {len(alerts)}")

    risky = tracker.get_risky_agents(top_n=3)
    print("Risky Agents (by RWI):")
    for agent_id, total_rwi, attack_surface in risky:
        print(f"  {agent_id}: RWI={total_rwi:.4f}, AS={attack_surface:.4f}")

    # Summary report
    print("\n" + "=" * 50)
    print("Summary Report:")
    print("-" * 50)

    summary = tracker.summary()
    print(f"Total Events: {summary.total_events}")
    print(f"Unique Senders: {summary.unique_senders}")
    print(f"Unique Receivers: {summary.unique_receivers}")
    print(f"Avg Drift Delta: {summary.avg_drift_delta:.4f}")
    print(f"Avg Influence Score: {summary.avg_influence_score:.4f}")
    print(f"High Drift Events: {summary.high_drift_count}")
    print(f"High Influence Events: {summary.high_influence_count}")
    print(f"Top Influencers: {summary.top_influencers[:3]}")
    print(f"Top Susceptible: {summary.top_susceptible[:3]}")

    tracker.close()

    print("\n" + "=" * 50)
    print("Smoke test completed successfully!")


if __name__ == "__main__":
    main()
