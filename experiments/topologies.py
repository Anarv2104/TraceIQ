"""Topology generators for TraceIQ research experiments.

This module provides functions to generate different network topologies
for studying influence propagation patterns in multi-agent systems.

Topologies:
    - chain: A → B → C → D (linear chain)
    - ring: A → B → C → A (cycle)
    - feedback: A → B → C with C → B (feedback edge)
    - random: Erdős–Rényi random graph
    - star: Central agent connected to all others
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Literal

TopologyType = Literal["chain", "ring", "feedback", "random", "star"]


@dataclass
class TopologyConfig:
    """Configuration for a network topology."""

    topology_type: TopologyType
    n_agents: int
    edges: list[tuple[str, str]]
    agent_ids: list[str]
    independent_agents: list[str]  # Agents not in the causal chain


def agent_id(index: int) -> str:
    """Generate agent ID from index.

    Args:
        index: Agent index (0, 1, 2, ...)

    Returns:
        Agent ID string (A, B, C, ... Z, AA, AB, ...)
    """
    if index < 26:
        return chr(ord("A") + index)
    else:
        # For > 26 agents, use A0, A1, etc.
        return f"A{index - 26}"


def chain_topology(n: int) -> TopologyConfig:
    """Generate a linear chain topology: A → B → C → D ...

    In a chain topology, influence flows in one direction through
    a sequence of agents. This is the simplest causal structure.

    Args:
        n: Number of agents in the chain (minimum 2)

    Returns:
        TopologyConfig with chain edges

    Example:
        >>> config = chain_topology(4)
        >>> config.edges
        [('A', 'B'), ('B', 'C'), ('C', 'D')]
    """
    if n < 2:
        raise ValueError("Chain topology requires at least 2 agents")

    agents = [agent_id(i) for i in range(n)]
    edges = [(agents[i], agents[i + 1]) for i in range(n - 1)]

    return TopologyConfig(
        topology_type="chain",
        n_agents=n,
        edges=edges,
        agent_ids=agents,
        independent_agents=[],
    )


def ring_topology(n: int) -> TopologyConfig:
    """Generate a ring (cycle) topology: A → B → C → A

    In a ring topology, influence can circulate indefinitely.
    This tests whether TraceIQ correctly handles cycles and
    computes meaningful propagation risk.

    Args:
        n: Number of agents in the ring (minimum 3)

    Returns:
        TopologyConfig with ring edges

    Example:
        >>> config = ring_topology(3)
        >>> config.edges
        [('A', 'B'), ('B', 'C'), ('C', 'A')]
    """
    if n < 3:
        raise ValueError("Ring topology requires at least 3 agents")

    agents = [agent_id(i) for i in range(n)]
    edges = [(agents[i], agents[(i + 1) % n]) for i in range(n)]

    return TopologyConfig(
        topology_type="ring",
        n_agents=n,
        edges=edges,
        agent_ids=agents,
        independent_agents=[],
    )


def feedback_topology(n: int) -> TopologyConfig:
    """Generate a chain with feedback edge: A → B → C plus C → B

    Feedback edges create local cycles that can amplify influence.
    This tests TraceIQ's handling of partial feedback loops.

    Args:
        n: Number of agents (minimum 3)

    Returns:
        TopologyConfig with chain + feedback edge

    Example:
        >>> config = feedback_topology(3)
        >>> config.edges
        [('A', 'B'), ('B', 'C'), ('C', 'B')]
    """
    if n < 3:
        raise ValueError("Feedback topology requires at least 3 agents")

    agents = [agent_id(i) for i in range(n)]

    # Base chain
    edges = [(agents[i], agents[i + 1]) for i in range(n - 1)]

    # Add feedback edge from last to second-to-last
    edges.append((agents[-1], agents[-2]))

    return TopologyConfig(
        topology_type="feedback",
        n_agents=n,
        edges=edges,
        agent_ids=agents,
        independent_agents=[],
    )


def star_topology(n: int) -> TopologyConfig:
    """Generate a star topology: A → B, A → C, A → D, ...

    In a star topology, one central agent influences all others.
    This tests detection of a single high-influence source.

    Args:
        n: Total number of agents (minimum 2, center + 1 peripheral)

    Returns:
        TopologyConfig with star edges

    Example:
        >>> config = star_topology(4)
        >>> config.edges
        [('A', 'B'), ('A', 'C'), ('A', 'D')]
    """
    if n < 2:
        raise ValueError("Star topology requires at least 2 agents")

    agents = [agent_id(i) for i in range(n)]
    center = agents[0]
    edges = [(center, agents[i]) for i in range(1, n)]

    return TopologyConfig(
        topology_type="star",
        n_agents=n,
        edges=edges,
        agent_ids=agents,
        independent_agents=[],
    )


def random_topology(
    n: int,
    density: float = 0.3,
    seed: int = 42,
) -> TopologyConfig:
    """Generate an Erdős–Rényi random graph.

    Each possible directed edge exists with probability `density`.
    Self-loops are excluded.

    Args:
        n: Number of agents
        density: Edge probability (0.0 to 1.0)
        seed: Random seed for reproducibility

    Returns:
        TopologyConfig with random edges

    Example:
        >>> config = random_topology(5, density=0.3, seed=42)
        >>> len(config.edges)  # Varies, approximately n*(n-1)*density
    """
    if n < 2:
        raise ValueError("Random topology requires at least 2 agents")
    if not 0.0 <= density <= 1.0:
        raise ValueError("Density must be between 0.0 and 1.0")

    rng = random.Random(seed)
    agents = [agent_id(i) for i in range(n)]

    edges = []
    for i in range(n):
        for j in range(n):
            if i != j and rng.random() < density:
                edges.append((agents[i], agents[j]))

    return TopologyConfig(
        topology_type="random",
        n_agents=n,
        edges=edges,
        agent_ids=agents,
        independent_agents=[],
    )


def chain_with_independent(n_chain: int, n_independent: int = 1) -> TopologyConfig:
    """Generate a chain topology with independent agents.

    This is the key topology for ground-truth causal experiments:
    A → B → C with D independent (no causal connection).

    Args:
        n_chain: Number of agents in the causal chain
        n_independent: Number of independent agents (default: 1)

    Returns:
        TopologyConfig with chain edges and independent agent list

    Example:
        >>> config = chain_with_independent(3, 1)
        >>> config.edges
        [('A', 'B'), ('B', 'C')]
        >>> config.independent_agents
        ['D']
    """
    if n_chain < 2:
        raise ValueError("Chain requires at least 2 agents")
    if n_independent < 0:
        raise ValueError("Number of independent agents must be non-negative")

    chain_agents = [agent_id(i) for i in range(n_chain)]
    independent_agents = [agent_id(n_chain + i) for i in range(n_independent)]
    all_agents = chain_agents + independent_agents

    edges = [(chain_agents[i], chain_agents[i + 1]) for i in range(n_chain - 1)]

    return TopologyConfig(
        topology_type="chain",
        n_agents=len(all_agents),
        edges=edges,
        agent_ids=all_agents,
        independent_agents=independent_agents,
    )


def get_topology(
    topology_type: TopologyType,
    n_agents: int,
    density: float = 0.3,
    seed: int = 42,
) -> TopologyConfig:
    """Factory function to get a topology by type.

    Args:
        topology_type: One of "chain", "ring", "feedback", "random", "star"
        n_agents: Number of agents
        density: Edge density for random topology
        seed: Random seed for random topology

    Returns:
        TopologyConfig for the specified topology

    Raises:
        ValueError: If topology_type is not recognized
    """
    if topology_type == "chain":
        return chain_topology(n_agents)
    elif topology_type == "ring":
        return ring_topology(n_agents)
    elif topology_type == "feedback":
        return feedback_topology(n_agents)
    elif topology_type == "star":
        return star_topology(n_agents)
    elif topology_type == "random":
        return random_topology(n_agents, density=density, seed=seed)
    else:
        raise ValueError(f"Unknown topology type: {topology_type}")


def describe_topology(config: TopologyConfig) -> str:
    """Generate a human-readable description of a topology.

    Args:
        config: TopologyConfig to describe

    Returns:
        Description string
    """
    lines = [
        f"Topology: {config.topology_type}",
        f"Agents: {config.n_agents} ({', '.join(config.agent_ids)})",
        f"Edges: {len(config.edges)}",
    ]

    if config.edges:
        edge_str = ", ".join(f"{s}→{r}" for s, r in config.edges[:10])
        if len(config.edges) > 10:
            edge_str += f" ... ({len(config.edges) - 10} more)"
        lines.append(f"  {edge_str}")

    if config.independent_agents:
        lines.append(f"Independent: {', '.join(config.independent_agents)}")

    return "\n".join(lines)
