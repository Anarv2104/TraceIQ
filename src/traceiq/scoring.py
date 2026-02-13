"""Drift and influence score calculations.

Baseline Definition (v1):
    The baseline for a receiver is the rolling mean of their recent
    receiver_content embeddings. This represents the receiver's "typical"
    response style over the baseline_window most recent interactions.

Drift Detection:
    drift_delta = 1 - cosine_similarity(current_embedding, baseline_embedding)

    Higher drift indicates the receiver deviated more from their typical
    behavior in this response.

Influence Scoring:
    baseline_shift = baseline_after - baseline_before
    influence_score = cosine_similarity(sender_embedding, baseline_shift)

    This measures how aligned the sender's content is with the direction
    of change in the receiver's baseline. A positive score indicates the
    receiver's behavior shifted toward the sender's content direction.
    Negative scores indicate counter-alignment (receiver shifted away).

    Range: -1.0 to +1.0
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from traceiq.models import ScoreResult

if TYPE_CHECKING:
    from numpy.typing import NDArray


def cosine_similarity(a: NDArray[np.float32], b: NDArray[np.float32]) -> float:
    """Compute cosine similarity between two vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


class ScoringEngine:
    """Computes drift and influence scores."""

    def __init__(
        self,
        baseline_window: int = 10,
        drift_threshold: float = 0.3,
        influence_threshold: float = 0.5,
    ) -> None:
        self.baseline_window = baseline_window
        self.drift_threshold = drift_threshold
        self.influence_threshold = influence_threshold

        # Per-receiver baseline tracking
        # Maps receiver_id -> list of recent embeddings
        self._receiver_baselines: dict[str, list[NDArray[np.float32]]] = {}

    def _get_baseline(self, receiver_id: str) -> NDArray[np.float32] | None:
        """Get mean baseline embedding for a receiver."""
        embeddings = self._receiver_baselines.get(receiver_id, [])
        if not embeddings:
            return None
        return np.mean(embeddings, axis=0).astype(np.float32)

    def _update_baseline(
        self, receiver_id: str, embedding: NDArray[np.float32]
    ) -> None:
        """Add embedding to receiver's rolling baseline."""
        if receiver_id not in self._receiver_baselines:
            self._receiver_baselines[receiver_id] = []
        self._receiver_baselines[receiver_id].append(embedding.copy())
        # Keep only the most recent embeddings
        if len(self._receiver_baselines[receiver_id]) > self.baseline_window:
            self._receiver_baselines[receiver_id] = self._receiver_baselines[
                receiver_id
            ][-self.baseline_window :]

    def compute_scores(
        self,
        event_id: UUID,
        sender_id: str,
        receiver_id: str,
        sender_embedding: NDArray[np.float32],
        receiver_embedding: NDArray[np.float32],
    ) -> ScoreResult:
        """
        Compute influence and drift scores for an interaction.

        Args:
            event_id: Unique event identifier
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent
            sender_embedding: Embedding of sender's content
            receiver_embedding: Embedding of receiver's response/state

        Returns:
            ScoreResult with computed metrics
        """
        baseline_before = self._get_baseline(receiver_id)
        flags: list[str] = []
        cold_start = False

        if baseline_before is None:
            # Cold start: no baseline exists yet
            cold_start = True
            drift_delta = 0.0
            influence_score = 0.0
            receiver_baseline_drift = 0.0
        else:
            # Compute drift: how much did receiver deviate from baseline?
            drift_delta = 1.0 - cosine_similarity(receiver_embedding, baseline_before)

            # Update baseline with new embedding
            self._update_baseline(receiver_id, receiver_embedding)
            baseline_after = self._get_baseline(receiver_id)

            # Baseline shift vector
            baseline_shift = baseline_after - baseline_before

            # Influence score: how aligned is sender with the baseline shift?
            shift_norm = np.linalg.norm(baseline_shift)
            if shift_norm > 1e-8:
                influence_score = cosine_similarity(sender_embedding, baseline_shift)
            else:
                influence_score = 0.0

            # Overall baseline drift (for tracking)
            receiver_baseline_drift = float(shift_norm)

            # Flag high values
            if drift_delta > self.drift_threshold:
                flags.append("high_drift")
            if influence_score > self.influence_threshold:
                flags.append("high_influence")

        # Still update baseline on cold start
        if cold_start:
            self._update_baseline(receiver_id, receiver_embedding)

        return ScoreResult(
            event_id=event_id,
            influence_score=influence_score,
            drift_delta=drift_delta,
            receiver_baseline_drift=receiver_baseline_drift,
            flags=flags,
            cold_start=cold_start,
        )

    def reset_baseline(self, receiver_id: str) -> None:
        """Reset a receiver's baseline."""
        if receiver_id in self._receiver_baselines:
            del self._receiver_baselines[receiver_id]

    def reset_all_baselines(self) -> None:
        """Reset all baselines."""
        self._receiver_baselines.clear()
