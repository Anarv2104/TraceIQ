"""Main InfluenceTracker class."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from traceiq.embeddings import MockEmbedder, SentenceTransformerEmbedder
from traceiq.export import export_combined_csv, export_combined_jsonl
from traceiq.graph import InfluenceGraph
from traceiq.models import InteractionEvent, ScoreResult, SummaryReport, TrackerConfig
from traceiq.scoring import ScoringEngine
from traceiq.storage import MemoryStorage, SQLiteStorage, StorageBackend

if TYPE_CHECKING:
    pass


class InfluenceTracker:
    """Main class for tracking AI-to-AI influence in multi-agent systems."""

    def __init__(
        self,
        config: TrackerConfig | None = None,
        use_mock_embedder: bool = False,
    ) -> None:
        """
        Initialize the InfluenceTracker.

        Args:
            config: Tracker configuration. Uses defaults if not provided.
            use_mock_embedder: Use mock embedder instead of sentence-transformers.
                              Useful for testing without the heavy dependency.
        """
        self.config = config or TrackerConfig()

        # Initialize storage backend
        self._storage: StorageBackend
        if self.config.storage_backend == "sqlite":
            if not self.config.storage_path:
                raise ValueError("storage_path required for sqlite backend")
            self._storage = SQLiteStorage(self.config.storage_path)
        else:
            self._storage = MemoryStorage()

        # Initialize embedder
        if use_mock_embedder:
            self._embedder = MockEmbedder(seed=self.config.random_seed)
        else:
            self._embedder = SentenceTransformerEmbedder(
                model_name=self.config.embedding_model,
                max_content_length=self.config.max_content_length,
                cache_size=self.config.embedding_cache_size,
            )

        # Initialize scoring engine
        self._scorer = ScoringEngine(
            baseline_window=self.config.baseline_window,
            drift_threshold=self.config.drift_threshold,
            influence_threshold=self.config.influence_threshold,
        )

        # Initialize graph
        self._graph = InfluenceGraph()

        # Set random seed if provided
        if self.config.random_seed is not None:
            np.random.seed(self.config.random_seed)

    def track_event(
        self,
        sender_id: str,
        receiver_id: str,
        sender_content: str,
        receiver_content: str,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Track a single interaction event.

        Args:
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent
            sender_content: Content from sender (message, output, etc.)
            receiver_content: Receiver's response or updated state
            metadata: Optional metadata dict

        Returns:
            Dict with event_id, scores, and flags
        """
        # Create event
        event = InteractionEvent(
            sender_id=sender_id,
            receiver_id=receiver_id,
            sender_content=sender_content,
            receiver_content=receiver_content,
            metadata=metadata or {},
        )

        # Compute embeddings
        sender_embedding = self._embedder.embed(sender_content)
        receiver_embedding = self._embedder.embed(receiver_content)

        # Compute scores
        score = self._scorer.compute_scores(
            event_id=event.event_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            sender_embedding=sender_embedding,
            receiver_embedding=receiver_embedding,
        )

        # Store
        self._storage.store_event(event)
        self._storage.store_score(score)

        # Update graph
        self._graph.add_interaction(event, score)

        return {
            "event_id": str(event.event_id),
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "influence_score": score.influence_score,
            "drift_delta": score.drift_delta,
            "flags": score.flags,
            "cold_start": score.cold_start,
        }

    def bulk_track(
        self,
        interactions: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Track multiple interactions.

        Args:
            interactions: List of dicts with keys:
                - sender_id, receiver_id, sender_content, receiver_content
                - Optional: metadata

        Returns:
            List of result dicts from track_event
        """
        results = []
        for interaction in interactions:
            result = self.track_event(
                sender_id=interaction["sender_id"],
                receiver_id=interaction["receiver_id"],
                sender_content=interaction["sender_content"],
                receiver_content=interaction["receiver_content"],
                metadata=interaction.get("metadata"),
            )
            results.append(result)
        return results

    def summary(self, top_n: int = 10) -> SummaryReport:
        """
        Generate a summary report of all tracked interactions.

        Args:
            top_n: Number of top agents to include in rankings

        Returns:
            SummaryReport with aggregated metrics
        """
        events = self._storage.get_all_events()
        scores = self._storage.get_all_scores()

        if not events:
            return SummaryReport(
                total_events=0,
                unique_senders=0,
                unique_receivers=0,
                avg_drift_delta=0.0,
                avg_influence_score=0.0,
                high_drift_count=0,
                high_influence_count=0,
                top_influencers=[],
                top_susceptible=[],
                influence_chains=[],
            )

        # Basic counts
        senders = {e.sender_id for e in events}
        receivers = {e.receiver_id for e in events}

        # Score aggregates
        non_cold_scores = [s for s in scores if not s.cold_start]
        if non_cold_scores:
            avg_drift = sum(s.drift_delta for s in non_cold_scores) / len(
                non_cold_scores
            )
            avg_influence = sum(s.influence_score for s in non_cold_scores) / len(
                non_cold_scores
            )
        else:
            avg_drift = 0.0
            avg_influence = 0.0

        high_drift = sum(1 for s in scores if "high_drift" in s.flags)
        high_influence = sum(1 for s in scores if "high_influence" in s.flags)

        # Graph analytics
        top_inf = self._graph.top_influencers(top_n)
        top_sus = self._graph.top_susceptible(top_n)

        # Find influence chains from top influencers
        chains: list[list[str]] = []
        for agent, _ in top_inf[:3]:  # Check top 3 influencers
            agent_chains = self._graph.find_influence_chains(
                agent,
                min_weight=self.config.influence_threshold / 2,
                max_length=4,
            )
            chains.extend(agent_chains[:5])  # Limit chains per agent

        return SummaryReport(
            total_events=len(events),
            unique_senders=len(senders),
            unique_receivers=len(receivers),
            avg_drift_delta=avg_drift,
            avg_influence_score=avg_influence,
            high_drift_count=high_drift,
            high_influence_count=high_influence,
            top_influencers=top_inf,
            top_susceptible=top_sus,
            influence_chains=chains[:10],  # Limit total chains
        )

    def export_csv(self, output_path: str | Path) -> None:
        """Export all data to CSV file."""
        events = self._storage.get_all_events()
        scores = self._storage.get_all_scores()
        export_combined_csv(events, scores, output_path)

    def export_jsonl(self, output_path: str | Path) -> None:
        """Export all data to JSONL file."""
        events = self._storage.get_all_events()
        scores = self._storage.get_all_scores()
        export_combined_jsonl(events, scores, output_path)

    def get_events(self) -> list[InteractionEvent]:
        """Get all tracked events."""
        return self._storage.get_all_events()

    def get_scores(self) -> list[ScoreResult]:
        """Get all computed scores."""
        return self._storage.get_all_scores()

    @property
    def graph(self) -> InfluenceGraph:
        """Access the influence graph for advanced analytics."""
        return self._graph

    @property
    def storage(self) -> StorageBackend:
        """Access the storage backend."""
        return self._storage

    def close(self) -> None:
        """Close the tracker and release resources."""
        self._storage.close()

    def __enter__(self) -> InfluenceTracker:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
