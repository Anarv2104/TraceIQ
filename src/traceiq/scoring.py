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

IEEE Metrics (v0.3.0):
    L2 Drift: ||s_j(t+) - s_j(t-)||_2
    IQx: drift / (baseline_median + epsilon)
    RWI: IQx * attack_surface
    Z-score: (IQx - mean) / (std + epsilon)
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np

from traceiq.metrics import (
    compute_drift_l2,
    compute_IQx,
    compute_RWI,
    compute_z_score,
    rolling_mean,
    rolling_median,
    rolling_std,
)
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
        epsilon: float = 1e-6,
        anomaly_threshold: float = 2.0,
    ) -> None:
        self.baseline_window = baseline_window
        self.drift_threshold = drift_threshold
        self.influence_threshold = influence_threshold
        self.epsilon = epsilon
        self.anomaly_threshold = anomaly_threshold

        # Per-receiver baseline tracking
        # Maps receiver_id -> list of recent embeddings
        self._receiver_baselines: dict[str, list[NDArray[np.float32]]] = {}

        # IEEE metrics tracking (v0.3.0)
        # Maps receiver_id -> list of recent L2 drift values (for median baseline)
        self._receiver_drift_history: dict[str, list[float]] = {}
        # Maps receiver_id -> list of recent IQx values (for Z-score computation)
        self._receiver_iqx_history: dict[str, list[float]] = {}

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

    def _get_drift_baseline_median(self, receiver_id: str) -> float:
        """Get median of recent L2 drift values for a receiver.

        This represents the receiver's "typical" drift magnitude.
        """
        history = self._receiver_drift_history.get(receiver_id, [])
        return rolling_median(history)

    def _get_iqx_stats(self, receiver_id: str) -> tuple[float, float]:
        """Get mean and std of recent IQx values for Z-score computation.

        Returns:
            Tuple of (mean, std)
        """
        history = self._receiver_iqx_history.get(receiver_id, [])
        return rolling_mean(history), rolling_std(history)

    def _update_drift_history(self, receiver_id: str, drift: float) -> None:
        """Add drift value to receiver's history."""
        if receiver_id not in self._receiver_drift_history:
            self._receiver_drift_history[receiver_id] = []
        self._receiver_drift_history[receiver_id].append(drift)
        # Keep only recent values
        if len(self._receiver_drift_history[receiver_id]) > self.baseline_window:
            self._receiver_drift_history[receiver_id] = self._receiver_drift_history[
                receiver_id
            ][-self.baseline_window :]

    def _update_iqx_history(self, receiver_id: str, iqx: float) -> None:
        """Add IQx value to receiver's history."""
        if receiver_id not in self._receiver_iqx_history:
            self._receiver_iqx_history[receiver_id] = []
        self._receiver_iqx_history[receiver_id].append(iqx)
        # Keep only recent values
        if len(self._receiver_iqx_history[receiver_id]) > self.baseline_window:
            self._receiver_iqx_history[receiver_id] = self._receiver_iqx_history[
                receiver_id
            ][-self.baseline_window :]

    def compute_scores(
        self,
        event_id: UUID,
        sender_id: str,
        receiver_id: str,
        sender_embedding: NDArray[np.float32],
        receiver_embedding: NDArray[np.float32],
        sender_attack_surface: float | None = None,
    ) -> ScoreResult:
        """
        Compute influence and drift scores for an interaction.

        Args:
            event_id: Unique event identifier
            sender_id: ID of the sending agent
            receiver_id: ID of the receiving agent
            sender_embedding: Embedding of sender's content
            receiver_embedding: Embedding of receiver's response/state
            sender_attack_surface: Optional attack surface score for RWI computation

        Returns:
            ScoreResult with computed metrics (including IEEE metrics)
        """
        baseline_before = self._get_baseline(receiver_id)
        flags: list[str] = []
        cold_start = False

        # IEEE metrics
        drift_l2: float | None = None
        iqx: float | None = None
        baseline_median: float | None = None
        rwi: float | None = None
        z_score: float | None = None
        alert_flag = False

        if baseline_before is None:
            # Cold start: no baseline exists yet
            cold_start = True
            drift_delta = 0.0
            influence_score = 0.0
            receiver_baseline_drift = 0.0
        else:
            # Compute drift: how much did receiver deviate from baseline?
            drift_delta = 1.0 - cosine_similarity(receiver_embedding, baseline_before)

            # Compute L2 drift (IEEE metric)
            drift_l2 = compute_drift_l2(baseline_before, receiver_embedding)

            # Get baseline median for IQx computation
            baseline_median = self._get_drift_baseline_median(receiver_id)

            # Compute IQx
            iqx = compute_IQx(drift_l2, baseline_median, self.epsilon)

            # Compute RWI if attack surface is provided
            if sender_attack_surface is not None:
                rwi = compute_RWI(iqx, sender_attack_surface)

            # Get IQx stats and compute Z-score
            iqx_mean, iqx_std = self._get_iqx_stats(receiver_id)
            if iqx_mean > 0 or iqx_std > 0:  # Have some history
                z_score = compute_z_score(iqx, iqx_mean, iqx_std, self.epsilon)
                if abs(z_score) > self.anomaly_threshold:
                    alert_flag = True
                    flags.append("anomaly_alert")

            # Update histories for future computations
            self._update_drift_history(receiver_id, drift_l2)
            self._update_iqx_history(receiver_id, iqx)

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
            # IEEE metrics
            drift_l2=drift_l2,
            IQx=iqx,
            baseline_median=baseline_median,
            RWI=rwi,
            Z_score=z_score,
            alert_flag=alert_flag,
        )

    def reset_baseline(self, receiver_id: str) -> None:
        """Reset a receiver's baseline."""
        if receiver_id in self._receiver_baselines:
            del self._receiver_baselines[receiver_id]
        if receiver_id in self._receiver_drift_history:
            del self._receiver_drift_history[receiver_id]
        if receiver_id in self._receiver_iqx_history:
            del self._receiver_iqx_history[receiver_id]

    def reset_all_baselines(self) -> None:
        """Reset all baselines."""
        self._receiver_baselines.clear()
        self._receiver_drift_history.clear()
        self._receiver_iqx_history.clear()

    def get_receiver_drift_history(self, receiver_id: str) -> list[float]:
        """Get drift history for a receiver (for analysis/export)."""
        return list(self._receiver_drift_history.get(receiver_id, []))

    def get_receiver_iqx_history(self, receiver_id: str) -> list[float]:
        """Get IQx history for a receiver (for analysis/export)."""
        return list(self._receiver_iqx_history.get(receiver_id, []))
