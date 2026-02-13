"""NetworkX graph analytics for influence tracking."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from traceiq.models import InteractionEvent, ScoreResult


class InfluenceGraph:
    """Graph-based analytics for agent influence patterns."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._edge_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._edge_drifts: dict[tuple[str, str], list[float]] = defaultdict(list)

    def add_interaction(
        self,
        event: InteractionEvent,
        score: ScoreResult,
    ) -> None:
        """Add an interaction to the graph."""
        sender = event.sender_id
        receiver = event.receiver_id

        # Add nodes if they don't exist
        if not self._graph.has_node(sender):
            self._graph.add_node(sender, type="agent")
        if not self._graph.has_node(receiver):
            self._graph.add_node(receiver, type="agent")

        # Track both influence scores and drift for this edge
        self._edge_scores[(sender, receiver)].append(score.influence_score)
        self._edge_drifts[(sender, receiver)].append(score.drift_delta)

        # Update edge with aggregated weight (influence-based)
        avg_score = sum(self._edge_scores[(sender, receiver)]) / len(
            self._edge_scores[(sender, receiver)]
        )
        avg_drift = sum(self._edge_drifts[(sender, receiver)]) / len(
            self._edge_drifts[(sender, receiver)]
        )
        self._graph.add_edge(
            sender,
            receiver,
            weight=avg_score,
            avg_drift=avg_drift,
            interaction_count=len(self._edge_scores[(sender, receiver)]),
        )

    def build_from_events(
        self,
        events: list[InteractionEvent],
        scores: list[ScoreResult],
    ) -> None:
        """Build graph from lists of events and scores."""
        score_map = {s.event_id: s for s in scores}
        for event in events:
            if event.event_id in score_map:
                self.add_interaction(event, score_map[event.event_id])

    def influence_matrix(self) -> dict[str, dict[str, float]]:
        """Get influence matrix as nested dict (sender -> receiver -> score)."""
        matrix: dict[str, dict[str, float]] = {}
        for sender, receiver, data in self._graph.edges(data=True):
            if sender not in matrix:
                matrix[sender] = {}
            matrix[sender][receiver] = data.get("weight", 0.0)
        return matrix

    def top_influencers(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Get top N influencers ranked by sum of outgoing influence weights.

        Returns list of (agent_id, total_outgoing_influence).
        Sorted by score descending, then agent_id ascending for determinism.
        """
        influence_sums: dict[str, float] = defaultdict(float)
        for sender, _receiver, data in self._graph.edges(data=True):
            influence_sums[sender] += data.get("weight", 0.0)

        # Sort by score descending, then agent_id ascending for deterministic ordering
        sorted_agents = sorted(influence_sums.items(), key=lambda x: (-x[1], x[0]))
        return sorted_agents[:n]

    def top_susceptible(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Get top N susceptible agents ranked by sum of incoming drift.

        Susceptibility measures how much an agent's behavior changes
        (drifts) when receiving messages from others.

        Returns list of (agent_id, total_incoming_drift).
        Sorted by score descending, then agent_id ascending for determinism.
        """
        susceptibility: dict[str, float] = defaultdict(float)
        for (_sender, receiver), drifts in self._edge_drifts.items():
            susceptibility[receiver] += sum(drifts)

        # Sort by score descending, then agent_id ascending for deterministic ordering
        sorted_agents = sorted(susceptibility.items(), key=lambda x: (-x[1], x[0]))
        return sorted_agents[:n]

    def top_influenced(self, n: int = 10) -> list[tuple[str, float]]:
        """
        Get top N agents ranked by sum of incoming influence weights.

        This measures which agents have had their behavior most
        correlated with sender content (i.e., moved toward senders).

        Returns list of (agent_id, total_incoming_influence).
        Sorted by score descending, then agent_id ascending for determinism.
        """
        influenced: dict[str, float] = defaultdict(float)
        for (_sender, receiver), scores in self._edge_scores.items():
            influenced[receiver] += sum(scores)

        # Sort by score descending, then agent_id ascending for deterministic ordering
        sorted_agents = sorted(influenced.items(), key=lambda x: (-x[1], x[0]))
        return sorted_agents[:n]

    def find_influence_chains(
        self,
        source: str,
        min_weight: float = 0.1,
        max_length: int = 5,
    ) -> list[list[str]]:
        """
        Find influence chains originating from a source agent.

        Returns paths where each edge has weight >= min_weight.
        """
        if source not in self._graph:
            return []

        # Build subgraph with only edges above threshold
        filtered_edges = [
            (u, v)
            for u, v, d in self._graph.edges(data=True)
            if d.get("weight", 0.0) >= min_weight
        ]

        if not filtered_edges:
            return []

        subgraph = self._graph.edge_subgraph(filtered_edges)

        # Source must exist in the filtered subgraph
        if source not in subgraph:
            return []

        chains: list[list[str]] = []
        for target in subgraph.nodes():
            if target != source:
                try:
                    for path in nx.all_simple_paths(
                        subgraph, source, target, cutoff=max_length
                    ):
                        if len(path) > 1:
                            chains.append(path)
                except nx.NetworkXError:
                    continue

        # Sort by length (longer chains first)
        chains.sort(key=len, reverse=True)
        return chains

    def detect_cycles(self, min_weight: float = 0.1) -> list[list[str]]:
        """Detect influence cycles in the graph."""
        filtered_edges = [
            (u, v)
            for u, v, d in self._graph.edges(data=True)
            if d.get("weight", 0.0) >= min_weight
        ]
        subgraph = self._graph.edge_subgraph(filtered_edges)

        try:
            cycles = list(nx.simple_cycles(subgraph))
            return cycles
        except nx.NetworkXError:
            return []

    def get_stats(self) -> dict:
        """Get graph statistics."""
        return {
            "num_nodes": self._graph.number_of_nodes(),
            "num_edges": self._graph.number_of_edges(),
            "density": nx.density(self._graph)
            if self._graph.number_of_nodes() > 0
            else 0.0,
            "is_connected": nx.is_weakly_connected(self._graph)
            if self._graph.number_of_nodes() > 0
            else False,
        }

    @property
    def graph(self) -> nx.DiGraph:
        """Access the underlying NetworkX graph."""
        return self._graph
