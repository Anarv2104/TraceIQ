"""Main InfluenceTracker class."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal
from uuid import uuid4

import numpy as np

from traceiq.capabilities import CapabilityRegistry
from traceiq.embeddings import MockEmbedder, SentenceTransformerEmbedder
from traceiq.export import export_combined_csv, export_combined_jsonl
from traceiq.graph import InfluenceGraph
from traceiq.metrics import compute_accumulated_influence
from traceiq.models import InteractionEvent, ScoreResult, SummaryReport, TrackerConfig
from traceiq.policy import PolicyEngine
from traceiq.risk import RiskResult, compute_risk_score
from traceiq.scoring import ScoringEngine
from traceiq.storage import MemoryStorage, SQLiteStorage, StorageBackend
from traceiq.validity import ValidityResult, check_validity

if TYPE_CHECKING:
    from traceiq.models import PropagationRiskResult
    from traceiq.schema import TraceIQEvent


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
            self._embedder = MockEmbedder(
                seed=self.config.random_seed,
                max_content_length=self.config.max_content_length,
            )
        else:
            self._embedder = SentenceTransformerEmbedder(
                model_name=self.config.embedding_model,
                max_content_length=self.config.max_content_length,
                cache_size=self.config.embedding_cache_size,
            )

        # Initialize scoring engine (with IEEE metrics parameters)
        self._scorer = ScoringEngine(
            baseline_window=self.config.baseline_window,
            drift_threshold=self.config.drift_threshold,
            influence_threshold=self.config.influence_threshold,
            epsilon=self.config.epsilon,
            anomaly_threshold=self.config.anomaly_threshold,
        )

        # Initialize graph
        self._graph = InfluenceGraph()

        # Initialize capability registry (v0.3.0)
        self._capabilities = CapabilityRegistry(
            weights=self.config.capability_weights or None
        )
        if self.config.capability_registry_path:
            self._capabilities.load_from_file(self.config.capability_registry_path)

        # Initialize policy engine (v0.4.0)
        self._policy: PolicyEngine | None = None
        if self.config.enable_policy:
            self._policy = PolicyEngine(
                enable_trust_decay=self.config.enable_trust_decay,
                trust_decay_rate=self.config.trust_decay_rate,
            )

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
        # NEW: Optional TraceIQEvent for full schema (v0.4.0)
        event: TraceIQEvent | None = None,
        # NEW: Run tracking (v0.4.0)
        run_id: str | None = None,
        task_id: str | None = None,
        # NEW: State quality hint (v0.4.0)
        state_quality: Literal["low", "medium", "high"] | None = None,
    ) -> dict[str, Any]:
        """
        Track a single interaction event.

        Args:
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent
            sender_content: Content from sender (message, output, etc.)
            receiver_content: Receiver's response or updated state
            metadata: Optional metadata dict
            event: Optional TraceIQEvent for full schema support
            run_id: Optional run identifier for experiment tracking
            task_id: Optional task identifier within a run
            state_quality: Optional state quality hint (auto-computed if not provided)

        Returns:
            Dict with event_id, scores, flags, and metrics (including v0.4.0 fields)
        """
        # Use provided event or create from params
        if event is not None:
            # Extract fields from TraceIQEvent
            sender_id = event.sender_id
            receiver_id = event.receiver_id
            sender_content = event.sender_content
            receiver_content = event.receiver_output
            metadata = event.metadata
            run_id = event.run_id
            task_id = event.task_id
            state_quality = event.state_quality

        # Create standard event
        interaction_event = InteractionEvent(
            sender_id=sender_id,
            receiver_id=receiver_id,
            sender_content=sender_content,
            receiver_content=receiver_content,
            metadata=metadata or {},
        )

        # Compute embeddings with truncation tracking
        sender_result = self._embedder.embed_with_info(sender_content)
        receiver_result = self._embedder.embed_with_info(receiver_content)

        # Get sender attack surface if registered
        sender_attack_surface = None
        if sender_id in self._capabilities:
            sender_attack_surface = self._capabilities.compute_attack_surface(sender_id)

        # Compute scores (including IEEE metrics)
        score = self._scorer.compute_scores(
            event_id=interaction_event.event_id,
            sender_id=sender_id,
            receiver_id=receiver_id,
            sender_embedding=sender_result.embedding,
            receiver_embedding=receiver_result.embedding,
            sender_attack_surface=sender_attack_surface,
        )

        # Add truncation flags if content was truncated
        if sender_result.was_truncated:
            score.flags.append("sender_truncated")
        if receiver_result.was_truncated:
            score.flags.append("receiver_truncated")

        # Compute validity (v0.4.0)
        baseline_samples = len(
            self._scorer._receiver_drift_history.get(receiver_id, [])
        )
        effective_state_quality = state_quality or "medium"
        validity = check_validity(
            baseline_samples=baseline_samples,
            state_quality=effective_state_quality,
            baseline_k=self.config.baseline_k,
        )

        # Update score with validity info
        score.valid = validity.valid
        score.invalid_reason = validity.invalid_reason
        score.confidence = validity.confidence

        # Override alert_flag if not valid (critical: no alerts on cold start)
        if not validity.valid:
            score.alert_flag = False
            if "anomaly_alert" in score.flags:
                score.flags.remove("anomaly_alert")

        # Compute risk score (v0.4.0)
        risk_result: RiskResult | None = None
        if self.config.enable_risk_scoring:
            # Get current PR for risk computation
            pr_window = self._graph.compute_spectral_radius("iqx")

            risk_result = compute_risk_score(
                robust_z=score.Z_score,
                drift=score.drift_l2_state or score.drift_l2_proxy,
                alignment=score.influence_score,
                pr_window=pr_window,
                valid=validity.valid,
                thresholds=self.config.risk_thresholds,
            )
            score.risk_score = risk_result.risk_score
            score.risk_level = risk_result.risk_level

        # Apply policy if enabled (v0.4.0)
        policy_action: str | None = None
        event_type: Literal["attempted", "applied", "blocked"] = "applied"

        if self._policy is not None and risk_result is not None:
            # Import here to avoid circular dependency
            from traceiq.schema import TraceIQEvent as FullEvent

            # Create TraceIQEvent for policy processing
            full_event = FullEvent(
                event_id=str(interaction_event.event_id),
                run_id=run_id or str(uuid4()),
                task_id=task_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                sender_content=sender_content,
                receiver_output=receiver_content,
                metadata=metadata or {},
            )

            # Apply policy
            updated_event = self._policy.apply_policy(full_event, risk_result)
            policy_action = updated_event.policy_action
            event_type = updated_event.event_type

            score.policy_action = policy_action
            score.event_type = event_type

        # Only store and update graph if event is applied (not blocked)
        if event_type != "blocked":
            self._storage.store_event(interaction_event)
            self._storage.store_score(score)
            self._graph.add_interaction(interaction_event, score)

        return {
            "event_id": str(interaction_event.event_id),
            "sender_id": sender_id,
            "receiver_id": receiver_id,
            "influence_score": score.influence_score,
            "drift_delta": score.drift_delta,
            "flags": score.flags,
            "cold_start": score.cold_start,
            # IEEE metrics (v0.3.0)
            "drift_l2_state": score.drift_l2_state,
            "drift_l2_proxy": score.drift_l2_proxy,
            "drift_l2": score.drift_l2,
            "IQx": score.IQx,
            "baseline_median": score.baseline_median,
            "RWI": score.RWI,
            "Z_score": score.Z_score,
            "alert": score.alert_flag,
            # NEW: v0.4.0 fields
            "valid": score.valid,
            "invalid_reason": score.invalid_reason,
            "confidence": score.confidence,
            "robust_z": score.Z_score,  # Alias for clarity
            "risk_score": score.risk_score,
            "risk_level": score.risk_level,
            "policy_action": policy_action,
            "event_type": event_type,
            # Run tracking
            "run_id": run_id,
            "task_id": task_id,
        }

    def bulk_track(
        self,
        interactions: list[dict[str, Any]],
        run_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """
        Track multiple interactions.

        Args:
            interactions: List of dicts with keys:
                - sender_id, receiver_id, sender_content, receiver_content
                - Optional: metadata, task_id
            run_id: Optional run identifier for all interactions

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
                run_id=run_id,
                task_id=interaction.get("task_id"),
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

    @property
    def capabilities(self) -> CapabilityRegistry:
        """Access the capability registry for agent security tracking."""
        return self._capabilities

    @property
    def policy(self) -> PolicyEngine | None:
        """Access the policy engine (if enabled)."""
        return self._policy

    # IEEE metrics methods (v0.3.0)

    def get_propagation_risk(self, weight_type: str = "iqx") -> float:
        """Get current propagation risk (spectral radius of influence graph).

        Args:
            weight_type: Type of weight to use: "iqx", "influence", or "drift"

        Returns:
            Spectral radius (values > 1.0 indicate potential influence amplification)
        """
        return self._graph.compute_spectral_radius(weight_type)

    def get_propagation_risk_over_time(
        self, window_size: int = 10
    ) -> list[PropagationRiskResult]:
        """Compute propagation risk over sliding time windows.

        Args:
            window_size: Number of events per window

        Returns:
            List of PropagationRiskResult for each window
        """
        events = self._storage.get_all_events()
        scores = self._storage.get_all_scores()
        return self._graph.compute_propagation_risk_over_time(
            events, scores, window_size
        )

    def get_accumulated_influence(self, agent_id: str) -> float:
        """Get accumulated influence (sum of IQx) for an agent.

        This sums all IQx values where agent_id was the sender.

        Args:
            agent_id: Agent identifier

        Returns:
            Sum of IQx values for interactions where this agent was sender
        """
        scores = self._storage.get_all_scores()
        events = self._storage.get_all_events()

        # Create event_id -> sender_id mapping
        sender_map = {e.event_id: e.sender_id for e in events}

        # Sum IQx for events where agent was sender
        iqx_values = [
            s.IQx
            for s in scores
            if s.IQx is not None and sender_map.get(s.event_id) == agent_id
        ]

        return compute_accumulated_influence(iqx_values)

    def get_alerts(
        self,
        threshold: float | None = None,
        valid_only: bool = True,
    ) -> list[ScoreResult]:
        """Get all anomaly alerts.

        Args:
            threshold: Optional minimum Z-score threshold (overrides config)
            valid_only: Only return alerts from valid metrics (default: True)

        Returns:
            List of ScoreResult with alert_flag=True, sorted by Z-score descending
        """
        scores = self._storage.get_all_scores()

        # Filter for alerts
        alerts = [s for s in scores if s.alert_flag]

        # Filter for valid metrics only (v0.4.0)
        if valid_only:
            alerts = [s for s in alerts if s.valid]

        # Apply threshold filter if provided
        if threshold is not None:
            alerts = [
                s
                for s in alerts
                if s.Z_score is not None and abs(s.Z_score) > threshold
            ]

        # Sort by absolute Z-score descending
        alerts.sort(key=lambda s: abs(s.Z_score or 0), reverse=True)

        return alerts

    def get_risky_agents(self, top_n: int = 10) -> list[tuple[str, float, float]]:
        """Get agents ranked by risk (RWI * accumulated influence).

        Args:
            top_n: Number of top agents to return

        Returns:
            List of (agent_id, total_rwi, attack_surface) tuples
        """
        scores = self._storage.get_all_scores()
        events = self._storage.get_all_events()

        # Create event_id -> sender_id mapping
        sender_map = {e.event_id: e.sender_id for e in events}

        # Sum RWI for each agent
        rwi_sums: dict[str, float] = {}
        for s in scores:
            if s.RWI is not None:
                sender = sender_map.get(s.event_id)
                if sender:
                    rwi_sums[sender] = rwi_sums.get(sender, 0.0) + s.RWI

        # Get attack surface for each agent
        results = []
        for agent_id, total_rwi in rwi_sums.items():
            attack_surface = self._capabilities.compute_attack_surface(agent_id)
            results.append((agent_id, total_rwi, attack_surface))

        # Sort by total RWI descending
        results.sort(key=lambda x: -x[1])

        return results[:top_n]

    # v0.4.0 methods

    def get_trust_score(self, agent_id: str) -> float:
        """Get trust score for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Trust score in [0, 1], or 1.0 if policy not enabled
        """
        if self._policy is None:
            return 1.0
        return self._policy.get_trust(agent_id)

    def get_low_trust_agents(
        self, threshold: float = 0.5
    ) -> list[tuple[str, float]]:
        """Get agents with low trust scores.

        Args:
            threshold: Trust threshold (default: 0.5)

        Returns:
            List of (agent_id, trust_score) tuples for agents below threshold
        """
        if self._policy is None:
            return []
        return self._policy.get_low_trust_agents(threshold)

    def close(self) -> None:
        """Close the tracker and release resources."""
        self._storage.close()

    def __enter__(self) -> InfluenceTracker:
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
