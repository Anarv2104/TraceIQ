"""NetworkX graph analytics for influence tracking."""

from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

import networkx as nx
import numpy as np

from traceiq.metrics import IQX_CAP, build_adjacency_matrix, compute_propagation_risk
from traceiq.weights import bounded_iqx_weight, confidence_weight

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from traceiq.models import InteractionEvent, PropagationRiskResult, ScoreResult


class InfluenceGraph:
    """Graph-based analytics for agent influence patterns."""

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._edge_scores: dict[tuple[str, str], list[float]] = defaultdict(list)
        self._edge_drifts: dict[tuple[str, str], list[float]] = defaultdict(list)
        # IEEE metrics (v0.3.0)
        self._edge_iqx: dict[tuple[str, str], list[float]] = defaultdict(list)

    def add_interaction(
        self,
        event: InteractionEvent,
        score: ScoreResult,
    ) -> None:
        """Add an interaction to the graph.

        Note: Events with event_type="blocked" are skipped to exclude
        blocked events from graph computations (v0.4.0).
        """
        # Skip blocked events (v0.4.0)
        if hasattr(score, "event_type") and score.event_type == "blocked":
            return

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

        # Track IQx if available (v0.3.0) - cap extreme values for stability
        if score.IQx is not None:
            capped_iqx = min(score.IQx, IQX_CAP)
            self._edge_iqx[(sender, receiver)].append(capped_iqx)
        elif score.drift_l2_state is not None:
            # Fallback: use raw drift as weight when IQx is None (cold start)
            # This ensures PR can be computed even during warmup
            self._edge_iqx[(sender, receiver)].append(score.drift_l2_state)
        elif score.drift_l2_proxy is not None:
            # Secondary fallback: use proxy drift
            self._edge_iqx[(sender, receiver)].append(score.drift_l2_proxy)

        # Update edge with aggregated weight (influence-based)
        avg_score = sum(self._edge_scores[(sender, receiver)]) / len(
            self._edge_scores[(sender, receiver)]
        )
        avg_drift = sum(self._edge_drifts[(sender, receiver)]) / len(
            self._edge_drifts[(sender, receiver)]
        )

        # Compute average IQx if available
        avg_iqx = 0.0
        if self._edge_iqx[(sender, receiver)]:
            avg_iqx = sum(self._edge_iqx[(sender, receiver)]) / len(
                self._edge_iqx[(sender, receiver)]
            )

        self._graph.add_edge(
            sender,
            receiver,
            weight=avg_score,
            avg_drift=avg_drift,
            avg_iqx=avg_iqx,
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

    def iqx_matrix(self) -> dict[str, dict[str, float]]:
        """Get IQx matrix as nested dict (sender -> receiver -> avg_iqx)."""
        matrix: dict[str, dict[str, float]] = {}
        for sender, receiver, data in self._graph.edges(data=True):
            if sender not in matrix:
                matrix[sender] = {}
            matrix[sender][receiver] = data.get("avg_iqx", 0.0)
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

    # IEEE metrics methods (v0.3.0)

    def build_adjacency_matrix(
        self, weight_type: str = "iqx"
    ) -> tuple[NDArray[np.float64], list[str]]:
        """Build adjacency matrix from graph edges.

        Args:
            weight_type: Type of weight to use: "iqx", "influence", or "drift"

        Returns:
            Tuple of (NxN numpy array, list of agent IDs in matrix order)
        """
        agents = sorted(self._graph.nodes())
        if not agents:
            return np.array([], dtype=np.float64), []

        # Get edge weights based on type
        if weight_type == "iqx":
            edge_weights = self.get_mean_iqx_weights()
        elif weight_type == "drift":
            edge_weights = {
                edge: sum(drifts) / len(drifts)
                for edge, drifts in self._edge_drifts.items()
                if drifts
            }
        else:  # influence
            edge_weights = {
                edge: sum(scores) / len(scores)
                for edge, scores in self._edge_scores.items()
                if scores
            }

        matrix = build_adjacency_matrix(edge_weights, agents)
        return matrix, agents

    def compute_spectral_radius(self, weight_type: str = "iqx") -> float:
        """Compute spectral radius (propagation risk) of the adjacency matrix.

        Args:
            weight_type: Type of weight to use: "iqx", "influence", or "drift"

        Returns:
            Spectral radius (largest absolute eigenvalue)
        """
        matrix, _ = self.build_adjacency_matrix(weight_type)
        return compute_propagation_risk(matrix)

    def compute_propagation_risk_over_time(
        self,
        events: list[InteractionEvent],
        scores: list[ScoreResult],
        window_size: int = 10,
    ) -> list[PropagationRiskResult]:
        """Compute propagation risk over sliding time windows.

        Args:
            events: List of events ordered by timestamp
            scores: Corresponding scores
            window_size: Number of events per window

        Returns:
            List of PropagationRiskResult for each window
        """
        from traceiq.models import PropagationRiskResult

        if len(events) < window_size:
            return []

        score_map = {s.event_id: s for s in scores}
        results: list[PropagationRiskResult] = []

        # Slide window over events
        for i in range(0, len(events) - window_size + 1, window_size // 2 or 1):
            window_events = events[i : i + window_size]
            window_scores = [
                score_map[e.event_id] for e in window_events if e.event_id in score_map
            ]

            # Build temporary graph for this window
            temp_graph = InfluenceGraph()
            for event, score in zip(window_events, window_scores, strict=False):
                temp_graph.add_interaction(event, score)

            # Compute spectral radius
            matrix, agents = temp_graph.build_adjacency_matrix("iqx")
            spectral_radius = compute_propagation_risk(matrix)

            results.append(
                PropagationRiskResult(
                    window_start=window_events[0].timestamp,
                    window_end=window_events[-1].timestamp,
                    spectral_radius=spectral_radius,
                    edge_count=temp_graph._graph.number_of_edges(),
                    agent_count=len(agents),
                )
            )

        return results

    def get_mean_iqx_weights(self) -> dict[tuple[str, str], float]:
        """Get mean IQx for each edge, with capping for numerical stability.

        Returns:
            Dict mapping (sender, receiver) to mean IQx (capped to IQX_CAP)
        """
        result = {}
        for edge, iqx_values in self._edge_iqx.items():
            if iqx_values:
                # Cap each value to prevent extreme weights from early interactions
                capped_values = [min(v, IQX_CAP) for v in iqx_values]
                result[edge] = sum(capped_values) / len(capped_values)
        return result

    def get_edge_weight(self, sender: str, receiver: str) -> float:
        """Get sanitized edge weight for PR computation.

        Weight = clamp(mean_IQx, 0, IQX_CAP) if available
                 else clamp(mean_influence_score, 0, inf)

        Args:
            sender: Sender agent ID
            receiver: Receiver agent ID

        Returns:
            Sanitized edge weight
        """
        edge = (sender, receiver)

        if edge in self._edge_iqx and self._edge_iqx[edge]:
            raw = sum(self._edge_iqx[edge]) / len(self._edge_iqx[edge])
            return max(0, min(raw, IQX_CAP))
        elif edge in self._edge_scores and self._edge_scores[edge]:
            raw = sum(self._edge_scores[edge]) / len(self._edge_scores[edge])
            return max(0, raw)  # influence_score can be negative, clamp to 0
        return 0.0

    def top_iqx_influencers(self, n: int = 10) -> list[tuple[str, float]]:
        """Get top N agents ranked by sum of outgoing IQx.

        Returns list of (agent_id, total_outgoing_iqx).
        """
        iqx_sums: dict[str, float] = defaultdict(float)
        for (sender, _receiver), iqx_values in self._edge_iqx.items():
            iqx_sums[sender] += sum(iqx_values)

        sorted_agents = sorted(iqx_sums.items(), key=lambda x: (-x[1], x[0]))
        return sorted_agents[:n]

    def clear(self) -> None:
        """Clear all graph data."""
        self._graph.clear()
        self._edge_scores.clear()
        self._edge_drifts.clear()
        self._edge_iqx.clear()

    # v0.4.0 methods

    def build_windowed_adjacency(
        self,
        events: list[InteractionEvent],
        scores: list[ScoreResult],
        window_size: int = 50,
        use_bounded_weights: bool = True,
    ) -> tuple[NDArray[np.float64], list[str]]:
        """Build adjacency matrix from a window of events with bounded weights.

        This method filters to event_type=="applied" only and uses bounded
        weight transformations for numerical stability.

        Args:
            events: List of events (will be filtered to last window_size)
            scores: Corresponding scores
            window_size: Maximum number of events to include
            use_bounded_weights: Apply sigmoid transformation to weights

        Returns:
            Tuple of (NxN numpy array, list of agent IDs in matrix order)
        """
        # Filter to applied events only
        score_map = {s.event_id: s for s in scores}
        applied_events = []
        for event in events:
            score = score_map.get(event.event_id)
            if score is not None:
                event_type = getattr(score, "event_type", "applied")
                if event_type == "applied":
                    applied_events.append((event, score))

        # Take last window_size events
        windowed = applied_events[-window_size:]

        if not windowed:
            return np.array([], dtype=np.float64), []

        # Collect agents and edge weights
        agents: set[str] = set()
        edge_weights: dict[tuple[str, str], list[float]] = defaultdict(list)

        for event, score in windowed:
            agents.add(event.sender_id)
            agents.add(event.receiver_id)

            # Compute weight based on IQx
            if score.IQx is not None:
                if use_bounded_weights:
                    # Use bounded sigmoid transformation
                    conf = confidence_weight(getattr(score, "confidence", "medium"))
                    alignment = max(0, score.influence_score)  # Non-negative
                    weight = bounded_iqx_weight(score.IQx) * alignment * conf
                else:
                    weight = min(score.IQx, IQX_CAP)
            elif score.drift_l2_state is not None:
                weight = score.drift_l2_state
            elif score.drift_l2_proxy is not None:
                weight = score.drift_l2_proxy
            else:
                weight = 0.0

            edge_weights[(event.sender_id, event.receiver_id)].append(weight)

        # Build adjacency matrix
        agent_list = sorted(agents)
        mean_weights = {
            edge: sum(weights) / len(weights)
            for edge, weights in edge_weights.items()
            if weights
        }

        matrix = build_adjacency_matrix(mean_weights, agent_list)
        return matrix, agent_list

    def get_bounded_weights(self) -> dict[tuple[str, str], float]:
        """Get edge weights with bounded sigmoid transformation.

        Returns:
            Dict mapping (sender, receiver) to bounded weight in [0, 1]
        """
        result = {}
        for edge, iqx_values in self._edge_iqx.items():
            if iqx_values:
                mean_iqx = sum(iqx_values) / len(iqx_values)
                result[edge] = bounded_iqx_weight(mean_iqx)
        return result

    def compute_windowed_pr(
        self,
        events: list[InteractionEvent],
        scores: list[ScoreResult],
        window_size: int = 50,
    ) -> float:
        """Compute propagation risk for a window of events.

        Args:
            events: List of events
            scores: Corresponding scores
            window_size: Number of events to include

        Returns:
            Spectral radius of windowed adjacency matrix
        """
        matrix, _ = self.build_windowed_adjacency(
            events, scores, window_size, use_bounded_weights=True
        )
        return compute_propagation_risk(matrix)
