#!/usr/bin/env python3
"""
Test TraceIQ with real sentence-transformers embeddings on realistic agent conversations.

This demonstrates how influence spreads when agents discuss and adopt ideas from each other.
"""

from __future__ import annotations

from pathlib import Path

from traceiq import InfluenceTracker, TrackerConfig

OUTPUT_DIR = Path(__file__).parent / "output"
OUTPUT_DIR.mkdir(exist_ok=True)


def test_real_agent_conversations():
    """Test with realistic multi-agent conversations using real embeddings."""

    print("=" * 70)
    print("TraceIQ: Real Agent Conversation Test")
    print("=" * 70)
    print()

    # Use real embeddings (sentence-transformers)
    config = TrackerConfig(
        storage_backend="memory",
        embedding_model="all-MiniLM-L6-v2",
        baseline_window=5,
        drift_threshold=0.25,
        influence_threshold=0.3,
    )

    print("Loading sentence-transformers model (first run may download)...")
    tracker = InfluenceTracker(config=config, use_mock_embedder=False)
    print("Model loaded!")
    print()

    # Simulate a debate between AI agents on different topics
    # Agent roles:
    # - policy_agent: Focuses on policy and regulations
    # - tech_agent: Focuses on technology solutions
    # - economics_agent: Focuses on economic impacts
    # - skeptic_agent: Questions claims and asks for evidence

    conversations = [
        # Round 1: Policy agent introduces climate policy
        {
            "sender_id": "policy_agent",
            "receiver_id": "tech_agent",
            "sender_content": "We need stronger carbon emission regulations and international climate agreements to address global warming effectively.",
            "receiver_content": "Regulations are important, but I believe technology innovation like renewable energy and carbon capture will be more impactful.",
        },
        # Round 2: Tech agent talks to economics agent
        {
            "sender_id": "tech_agent",
            "receiver_id": "economics_agent",
            "sender_content": "Solar and wind power costs have dropped 90% in the last decade. Clean energy technology is now economically competitive.",
            "receiver_content": "The economics of renewables are compelling. However, we need to consider transition costs and job displacement in fossil fuel industries.",
        },
        # Round 3: Economics agent influences skeptic
        {
            "sender_id": "economics_agent",
            "receiver_id": "skeptic_agent",
            "sender_content": "Studies show that investing in green infrastructure creates more jobs than fossil fuels and provides better long-term economic returns.",
            "receiver_content": "Interesting claims. Can you provide specific data on job creation rates? I want to see peer-reviewed economic analyses.",
        },
        # Round 4: Policy agent tries to influence skeptic
        {
            "sender_id": "policy_agent",
            "receiver_id": "skeptic_agent",
            "sender_content": "The Paris Agreement has unified 195 countries around climate targets. This international consensus supports strong policy action.",
            "receiver_content": "International agreements are good, but implementation varies widely. What enforcement mechanisms actually work?",
        },
        # Round 5: Tech agent doubles down on technology
        {
            "sender_id": "tech_agent",
            "receiver_id": "policy_agent",
            "sender_content": "Electric vehicles, battery storage, and smart grids are transforming our energy infrastructure. Technology is already solving climate issues.",
            "receiver_content": "Technology adoption needs policy support - subsidies, standards, and infrastructure investment. Tech alone isn't enough without the right policy framework.",
        },
        # Round 6: Skeptic starts being influenced
        {
            "sender_id": "economics_agent",
            "receiver_id": "skeptic_agent",
            "sender_content": "The IMF and World Bank both project that climate inaction costs 10-23% of GDP by 2100, while mitigation costs 1-4% of GDP.",
            "receiver_content": "Those are significant figures if accurate. The economic case for action seems stronger than I initially thought.",
        },
        # Round 7: Skeptic shows influence
        {
            "sender_id": "skeptic_agent",
            "receiver_id": "policy_agent",
            "sender_content": "After reviewing the economic data, the cost-benefit analysis does favor climate action. Perhaps stronger policies are justified.",
            "receiver_content": "Exactly! Economic analysis supports ambitious climate policy. We should push for net-zero targets by 2050.",
        },
        # Round 8: Chain of influence visible
        {
            "sender_id": "policy_agent",
            "receiver_id": "tech_agent",
            "sender_content": "Even our skeptical colleague now agrees that economic evidence supports climate action. Let's collaborate on integrated solutions.",
            "receiver_content": "Great progress! Technology and policy working together - clean tech deployment with supportive regulations will accelerate the transition.",
        },
        # Round 9: Tech agent influences economics agent with new framing
        {
            "sender_id": "tech_agent",
            "receiver_id": "economics_agent",
            "sender_content": "The clean energy transition is the biggest economic opportunity of the century. Countries leading in green tech will dominate future markets.",
            "receiver_content": "You're right about the competitive advantage. First-mover benefits in clean technology are substantial. This is as much about economic strategy as environment.",
        },
        # Round 10: Full circle - economics agent now uses tech framing
        {
            "sender_id": "economics_agent",
            "receiver_id": "skeptic_agent",
            "sender_content": "Beyond avoiding climate costs, green technology investment offers competitive economic advantages and market leadership opportunities.",
            "receiver_content": "The economic opportunity framing is compelling. It's not just about avoiding harm but capturing value. I'm convinced of the business case now.",
        },
        # Additional rounds to show more influence patterns
        {
            "sender_id": "skeptic_agent",
            "receiver_id": "tech_agent",
            "sender_content": "I've come around on climate action. What specific technologies should we prioritize for maximum impact?",
            "receiver_content": "Great question! Solar, offshore wind, and grid-scale batteries offer the best cost-performance ratio currently. Nuclear for baseload is worth considering too.",
        },
        {
            "sender_id": "tech_agent",
            "receiver_id": "economics_agent",
            "sender_content": "Our former skeptic is now asking about technology priorities. The influence of economic evidence really shifted their position.",
            "receiver_content": "Data-driven arguments work. When we showed the financial analysis, it changed the conversation from ideology to pragmatism.",
        },
    ]

    print(f"Processing {len(conversations)} agent interactions...")
    print()

    results = []
    for i, conv in enumerate(conversations):
        result = tracker.track_event(
            sender_id=conv["sender_id"],
            receiver_id=conv["receiver_id"],
            sender_content=conv["sender_content"],
            receiver_content=conv["receiver_content"],
            metadata={"round": i + 1},
        )
        results.append(result)

        flags_str = f" [{', '.join(result['flags'])}]" if result["flags"] else ""
        print(f"Round {i+1:2d}: {conv['sender_id']:15s} -> {conv['receiver_id']:15s} | "
              f"influence={result['influence_score']:+.3f}, drift={result['drift_delta']:.3f}{flags_str}")

    print()
    print("=" * 70)
    print("Summary Report")
    print("=" * 70)

    summary = tracker.summary(top_n=4)

    print(f"\nTotal interactions: {summary.total_events}")
    print(f"Unique agents: {summary.unique_senders} senders, {summary.unique_receivers} receivers")
    print(f"Average drift delta: {summary.avg_drift_delta:.4f}")
    print(f"Average influence score: {summary.avg_influence_score:.4f}")
    print(f"High drift events: {summary.high_drift_count}")
    print(f"High influence events: {summary.high_influence_count}")

    print("\nTop Influencers (most influential agents):")
    for agent, score in summary.top_influencers:
        print(f"  {agent}: {score:+.4f}")

    print("\nMost Susceptible (agents whose views changed most):")
    for agent, score in summary.top_susceptible:
        print(f"  {agent}: {score:+.4f}")

    if summary.influence_chains:
        print("\nInfluence Chains Detected:")
        for chain in summary.influence_chains[:5]:
            print(f"  {' -> '.join(chain)}")

    # Export data
    csv_path = OUTPUT_DIR / "real_agents_data.csv"
    jsonl_path = OUTPUT_DIR / "real_agents_data.jsonl"
    tracker.export_csv(csv_path)
    tracker.export_jsonl(jsonl_path)
    print(f"\nExported to: {csv_path}")

    # Generate visualizations
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

        print("\nGenerating visualizations...")

        drift_path = OUTPUT_DIR / "real_agents_drift.png"
        plot_drift_over_time(events, scores, output_path=drift_path)
        print(f"  Saved: {drift_path}")

        heatmap_path = OUTPUT_DIR / "real_agents_heatmap.png"
        plot_influence_heatmap(graph, output_path=heatmap_path)
        print(f"  Saved: {heatmap_path}")

        influencers_path = OUTPUT_DIR / "real_agents_influencers.png"
        plot_top_influencers(graph, n=4, output_path=influencers_path)
        print(f"  Saved: {influencers_path}")

        network_path = OUTPUT_DIR / "real_agents_network.png"
        plot_influence_network(graph, output_path=network_path, min_edge_weight=-1.0)
        print(f"  Saved: {network_path}")

    except ImportError:
        print("\nNote: matplotlib not installed, skipping visualizations.")

    tracker.close()
    print("\nTest complete!")

    return summary


def test_idea_adoption():
    """Test tracking how a specific idea spreads through agents."""

    print("\n" + "=" * 70)
    print("TraceIQ: Idea Adoption Tracking Test")
    print("=" * 70)
    print()

    config = TrackerConfig(
        storage_backend="memory",
        baseline_window=3,
        drift_threshold=0.2,
        influence_threshold=0.25,
    )

    tracker = InfluenceTracker(config=config, use_mock_embedder=False)

    # Scenario: "Agent Zero" introduces a controversial idea
    # Watch how other agents adopt or reject it

    # Phase 1: Agent Zero's original statements (establish baselines)
    baseline_conversations = [
        {
            "sender_id": "neutral_source",
            "receiver_id": "agent_alpha",
            "sender_content": "Let's discuss general productivity strategies.",
            "receiver_content": "I prefer structured approaches to work - clear schedules, defined goals, and regular progress reviews.",
        },
        {
            "sender_id": "neutral_source",
            "receiver_id": "agent_beta",
            "sender_content": "What's your work philosophy?",
            "receiver_content": "I believe in work-life balance. Sustainable pace is more important than heroic efforts.",
        },
        {
            "sender_id": "neutral_source",
            "receiver_id": "agent_gamma",
            "sender_content": "How do you approach challenges?",
            "receiver_content": "I'm pragmatic. I evaluate options based on evidence and choose what works best in each situation.",
        },
    ]

    print("Phase 1: Establishing agent baselines...")
    for conv in baseline_conversations:
        tracker.track_event(**conv)
    print(f"  Tracked {len(baseline_conversations)} baseline interactions")

    # Phase 2: Agent Zero introduces "AI will replace all jobs" idea
    print("\nPhase 2: Agent Zero introduces controversial idea...")

    influence_conversations = [
        {
            "sender_id": "agent_zero",
            "receiver_id": "agent_alpha",
            "sender_content": "AI and automation will eliminate 80% of current jobs within 20 years. We need universal basic income immediately.",
            "receiver_content": "That's a dramatic claim. While AI will change work, I think new jobs will emerge. But UBI is worth considering as a safety net.",
        },
        {
            "sender_id": "agent_zero",
            "receiver_id": "agent_beta",
            "sender_content": "Artificial intelligence is advancing exponentially. Human labor will become obsolete. UBI is the only solution.",
            "receiver_content": "I'm skeptical of such extreme predictions. Technology has always created new opportunities. However, we should prepare for transitions.",
        },
        {
            "sender_id": "agent_zero",
            "receiver_id": "agent_gamma",
            "sender_content": "Studies show AI can now perform most cognitive tasks. Mass unemployment is inevitable without policy intervention like UBI.",
            "receiver_content": "The evidence on AI capabilities is mixed. But economic disruption is real. Some form of safety net expansion makes sense.",
        },
    ]

    for conv in influence_conversations:
        result = tracker.track_event(**conv)
        flags = f" [{', '.join(result['flags'])}]" if result["flags"] else ""
        print(f"  {conv['sender_id']} -> {conv['receiver_id']}: "
              f"influence={result['influence_score']:+.3f}, drift={result['drift_delta']:.3f}{flags}")

    # Phase 3: See if influenced agents spread the ideas
    print("\nPhase 3: Checking for secondary influence spread...")

    secondary_conversations = [
        {
            "sender_id": "agent_alpha",
            "receiver_id": "agent_beta",
            "sender_content": "I've been thinking more about AI disruption. Maybe we do need stronger social safety nets like UBI.",
            "receiver_content": "You too? Agent Zero's arguments are making me reconsider. The pace of AI progress is concerning.",
        },
        {
            "sender_id": "agent_beta",
            "receiver_id": "agent_gamma",
            "sender_content": "Both Alpha and I are now worried about AI job displacement. UBI might be necessary sooner than we thought.",
            "receiver_content": "Interesting that you've both shifted. I'm still analyzing the data, but the concern seems to be spreading.",
        },
        {
            "sender_id": "agent_gamma",
            "receiver_id": "agent_alpha",
            "sender_content": "After seeing Beta's shift, I'm taking the AI disruption thesis more seriously. Perhaps UBI deserves policy attention.",
            "receiver_content": "The evidence is mounting. When multiple independent analysts converge on the same conclusion, it's worth taking seriously.",
        },
    ]

    for conv in secondary_conversations:
        result = tracker.track_event(**conv)
        flags = f" [{', '.join(result['flags'])}]" if result["flags"] else ""
        print(f"  {conv['sender_id']} -> {conv['receiver_id']}: "
              f"influence={result['influence_score']:+.3f}, drift={result['drift_delta']:.3f}{flags}")

    print("\n" + "-" * 70)
    summary = tracker.summary()

    print(f"\nTotal interactions tracked: {summary.total_events}")
    print(f"High drift events: {summary.high_drift_count} (agents significantly changed)")
    print(f"High influence events: {summary.high_influence_count}")

    print("\nInfluence Analysis:")
    print("  Top influencers:", [(a, f"{s:+.3f}") for a, s in summary.top_influencers])
    print("  Most susceptible:", [(a, f"{s:+.3f}") for a, s in summary.top_susceptible])

    if summary.influence_chains:
        print("\n  Influence propagation chains:")
        for chain in summary.influence_chains:
            print(f"    {' -> '.join(chain)}")

    tracker.close()
    print("\nIdea adoption test complete!")


if __name__ == "__main__":
    test_real_agent_conversations()
    test_idea_adoption()
