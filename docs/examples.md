# Examples

Real-world usage patterns for TraceIQ.

## Example 1: Tracking a Multi-Agent Debate

Track how agents influence each other during a policy debate.

```python
from traceiq import InfluenceTracker, TrackerConfig

config = TrackerConfig(
    storage_backend="memory",
    baseline_window=5,
    baseline_k=5,  # Must be <= baseline_window
    drift_threshold=0.25,
    influence_threshold=0.3,
)

tracker = InfluenceTracker(config=config, use_mock_embedder=False)

# Simulate debate interactions
debate = [
    {
        "sender_id": "policy_expert",
        "receiver_id": "economist",
        "sender_content": "Carbon taxes are essential for reducing emissions.",
        "receiver_content": "Taxes could work, but we need to consider economic impacts.",
    },
    {
        "sender_id": "economist",
        "receiver_id": "policy_expert",
        "sender_content": "Economic analysis shows carbon pricing is cost-effective.",
        "receiver_content": "Good point. Let's design policies that minimize economic disruption.",
    },
    {
        "sender_id": "policy_expert",
        "receiver_id": "scientist",
        "sender_content": "We're considering carbon pricing with economic safeguards.",
        "receiver_content": "That sounds reasonable. The science supports urgent action.",
    },
]

results = tracker.bulk_track(debate)

# Analyze results
for r in results:
    print(f"{r['sender_id']} -> {r['receiver_id']}")
    print(f"  Influence: {r['influence_score']:+.3f}")
    print(f"  Drift: {r['drift_delta']:.3f}")
    if r['flags']:
        print(f"  Flags: {r['flags']}")
```

## Example 2: Detecting Idea Propagation

Track how a specific idea spreads through a network.

```python
from traceiq import InfluenceTracker

tracker = InfluenceTracker(use_mock_embedder=False)

# Phase 1: Establish baselines
baseline_interactions = [
    {"sender_id": "moderator", "receiver_id": "agent_1",
     "sender_content": "What do you think about technology?",
     "receiver_content": "Technology has pros and cons."},
    {"sender_id": "moderator", "receiver_id": "agent_2",
     "sender_content": "Share your views on innovation.",
     "receiver_content": "Innovation drives progress."},
    {"sender_id": "moderator", "receiver_id": "agent_3",
     "sender_content": "What's your perspective?",
     "receiver_content": "I prefer balanced approaches."},
]
tracker.bulk_track(baseline_interactions)

# Phase 2: Introduce viral idea
viral_idea = "AI will revolutionize everything within 5 years!"

propagation = [
    {"sender_id": "influencer", "receiver_id": "agent_1",
     "sender_content": viral_idea,
     "receiver_content": "AI is advancing rapidly. Revolution is possible."},
    {"sender_id": "agent_1", "receiver_id": "agent_2",
     "sender_content": "AI might revolutionize things soon.",
     "receiver_content": "You're right, AI progress is accelerating."},
    {"sender_id": "agent_2", "receiver_id": "agent_3",
     "sender_content": "AI revolution is coming faster than expected.",
     "receiver_content": "I'm starting to believe AI will change everything."},
]

results = tracker.bulk_track(propagation)

# Check propagation
summary = tracker.summary()
print(f"Top influencer: {summary.top_influencers[0]}")
print(f"Most influenced: {summary.top_susceptible[0]}")
print(f"Influence chains: {summary.influence_chains}")
```

## Example 3: Real-Time Monitoring

Monitor agent interactions in real-time.

```python
from traceiq import InfluenceTracker
import time

tracker = InfluenceTracker(use_mock_embedder=True)

def on_interaction(sender, receiver, sender_msg, receiver_msg):
    """Callback for each interaction."""
    result = tracker.track_event(
        sender_id=sender,
        receiver_id=receiver,
        sender_content=sender_msg,
        receiver_content=receiver_msg,
    )

    # Alert on high influence
    if "high_influence" in result["flags"]:
        print(f"ALERT: High influence detected!")
        print(f"  {sender} strongly influenced {receiver}")
        print(f"  Score: {result['influence_score']:.3f}")

    # Alert on high drift
    if "high_drift" in result["flags"]:
        print(f"WARNING: High drift for {receiver}")
        print(f"  Drift: {result['drift_delta']:.3f}")

    return result

# Simulate real-time interactions
interactions = [
    ("bot_a", "bot_b", "Buy crypto now!", "Crypto seems interesting."),
    ("bot_a", "bot_c", "Crypto is the future!", "Maybe I should invest."),
    ("bot_b", "bot_c", "I'm buying crypto!", "Everyone's talking about crypto!"),
]

for sender, receiver, s_msg, r_msg in interactions:
    on_interaction(sender, receiver, s_msg, r_msg)
    time.sleep(0.1)  # Simulate delay

print("\nFinal Summary:")
summary = tracker.summary()
print(f"Potential manipulation source: {summary.top_influencers[0][0]}")
```

## Example 4: Persistent Storage with SQLite

Store data persistently for long-running analysis.

```python
from traceiq import InfluenceTracker, TrackerConfig
from pathlib import Path

db_path = Path("agent_monitoring.db")

config = TrackerConfig(
    storage_backend="sqlite",
    storage_path=str(db_path),
)

# Session 1: Track some interactions
with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
    tracker.track_event(
        sender_id="agent_1",
        receiver_id="agent_2",
        sender_content="First message",
        receiver_content="First response",
    )
    print(f"Session 1: {len(tracker.get_events())} events")

# Session 2: Data persists
with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
    tracker.track_event(
        sender_id="agent_2",
        receiver_id="agent_3",
        sender_content="Second message",
        receiver_content="Second response",
    )
    print(f"Session 2: {len(tracker.get_events())} events")  # Shows 2 events
```

## Example 5: Generating Reports

Generate comprehensive reports with visualizations.

```python
from traceiq import InfluenceTracker
from traceiq.plotting import (
    plot_drift_over_time,
    plot_influence_heatmap,
    plot_top_influencers,
    plot_influence_network,
)
from pathlib import Path

tracker = InfluenceTracker(use_mock_embedder=True)

# Track many interactions...
interactions = [...]  # Your interaction data
tracker.bulk_track(interactions)

# Create output directory
output_dir = Path("report")
output_dir.mkdir(exist_ok=True)

# Export data
tracker.export_csv(output_dir / "data.csv")
tracker.export_jsonl(output_dir / "data.jsonl")

# Generate visualizations
events = tracker.get_events()
scores = tracker.get_scores()
graph = tracker.graph

plot_drift_over_time(events, scores, output_path=output_dir / "drift.png")
plot_influence_heatmap(graph, output_path=output_dir / "heatmap.png")
plot_top_influencers(graph, n=10, output_path=output_dir / "influencers.png")
plot_influence_network(graph, output_path=output_dir / "network.png")

# Generate summary
summary = tracker.summary()
with open(output_dir / "summary.txt", "w") as f:
    f.write(f"Total Events: {summary.total_events}\n")
    f.write(f"Unique Senders: {summary.unique_senders}\n")
    f.write(f"Unique Receivers: {summary.unique_receivers}\n")
    f.write(f"Avg Drift: {summary.avg_drift_delta:.4f}\n")
    f.write(f"Avg Influence: {summary.avg_influence_score:.4f}\n")
    f.write(f"\nTop Influencers:\n")
    for agent, score in summary.top_influencers:
        f.write(f"  {agent}: {score:.4f}\n")

print(f"Report generated in {output_dir}/")
```

## Example 6: Custom Thresholds for Different Domains

Adjust thresholds based on your domain.

```python
from traceiq import InfluenceTracker, TrackerConfig

# Sensitive domain: lower thresholds
sensitive_config = TrackerConfig(
    drift_threshold=0.15,      # Flag smaller changes
    influence_threshold=0.25,  # Flag weaker influence
    baseline_window=20,        # Longer baseline for stability
)

# General domain: default thresholds
general_config = TrackerConfig(
    drift_threshold=0.3,
    influence_threshold=0.5,
    baseline_window=10,
    baseline_k=10,
)

# Rapidly changing domain: higher thresholds
dynamic_config = TrackerConfig(
    drift_threshold=0.5,       # Only flag major changes
    influence_threshold=0.7,   # Only flag strong influence
    baseline_window=5,         # Short window for adaptation
    baseline_k=5,
)
```

## Running the Examples

The `examples/` directory contains runnable scripts:

```bash
# Infection simulation
python examples/simulate_infection.py

# Real agent test
python examples/test_real_agents.py
```
