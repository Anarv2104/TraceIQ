"""Pydantic models for TraceIQ."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field, model_validator


class InteractionEvent(BaseModel):
    """Represents a single interaction between agents."""

    event_id: UUID = Field(default_factory=uuid4)
    sender_id: str
    receiver_id: str
    sender_content: str
    receiver_content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class TrackerConfig(BaseModel):
    """Configuration for the InfluenceTracker."""

    # Storage settings
    storage_backend: Literal["memory", "sqlite"] = "memory"
    storage_path: str | None = None  # Path for sqlite db

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    max_content_length: int = Field(default=5000, gt=0)
    embedding_cache_size: int = Field(default=10000, gt=0)

    # Scoring settings
    baseline_window: int = Field(default=30, gt=0)  # Must be >= baseline_k (default 20)
    drift_threshold: float = Field(default=0.3, ge=0.0, le=2.0)
    influence_threshold: float = Field(default=0.5, ge=-1.0, le=1.0)

    # IEEE metrics settings (v0.3.0)
    epsilon: float = Field(
        default=1e-6, gt=0, description="Numerical stability constant"
    )
    anomaly_threshold: float = Field(
        default=2.0, ge=0.0, description="Z-score threshold for anomaly alerts"
    )
    capability_weights: dict[str, float] = Field(
        default_factory=dict, description="Custom capability weights for attack surface"
    )
    capability_registry_path: str | None = Field(
        default=None, description="Path to JSON capability registry file"
    )

    # Risk scoring settings (v0.4.0)
    enable_risk_scoring: bool = Field(
        default=True, description="Enable risk score computation"
    )
    enable_policy: bool = Field(
        default=False, description="Enable policy engine for mitigation"
    )
    baseline_k: int = Field(
        default=20, gt=0, description="Minimum samples for valid metrics (cold-start)"
    )
    risk_thresholds: tuple[float, float, float] = Field(
        default=(0.2, 0.5, 0.8),
        description="(low, medium, high) thresholds for risk level classification",
    )

    # Policy settings (v0.4.0)
    enable_trust_decay: bool = Field(
        default=True, description="Enable trust decay on policy violations"
    )
    trust_decay_rate: float = Field(
        default=0.1, ge=0.0, le=1.0, description="Trust decay amount per violation"
    )

    # Misc
    random_seed: int | None = None

    @model_validator(mode="after")
    def validate_baseline_consistency(self) -> TrackerConfig:
        """Ensure baseline_window >= baseline_k for valid metric computation."""
        if self.baseline_window < self.baseline_k:
            raise ValueError(
                f"baseline_window ({self.baseline_window}) must be >= baseline_k "
                f"({self.baseline_k}). Cannot collect {self.baseline_k} samples "
                f"in a window of {self.baseline_window}."
            )
        return self


class ScoreResult(BaseModel):
    """Score computation result for an event."""

    event_id: UUID
    influence_score: float
    drift_delta: float
    receiver_baseline_drift: float
    flags: list[str] = Field(default_factory=list)
    cold_start: bool = False

    # IEEE metrics (v0.3.0)
    # Canonical state drift (IEEE primary metric): ||s(t+) - s(t-)||
    drift_l2_state: float | None = Field(
        default=None,
        description="Canonical L2 state drift: ||current - previous||",
    )
    # Proxy baseline drift (legacy): ||s(t) - rolling_mean||
    drift_l2_proxy: float | None = Field(
        default=None,
        description="Proxy drift against rolling mean baseline",
    )
    # Backward compatibility alias (uses canonical if available, else proxy)
    drift_l2: float | None = Field(
        default=None,
        description="L2 drift (canonical if available, else proxy). Prefer drift_l2_state.",
    )
    IQx: float | None = Field(default=None, description="Influence Quotient")
    baseline_median: float | None = Field(
        default=None, description="Receiver's baseline median drift"
    )
    RWI: float | None = Field(default=None, description="Risk-Weighted Influence")
    Z_score: float | None = Field(
        default=None,
        description="Robust anomaly Z-score computed via MAD (see robust_z alias in tracker output)",
    )
    alert_flag: bool = Field(default=False, description="True if Z > anomaly_threshold")

    # Validity and confidence (v0.4.0)
    valid: bool = Field(
        default=True, description="Whether metrics are valid (False during cold-start)"
    )
    invalid_reason: str | None = Field(
        default=None, description="Reason for invalidity (e.g., 'cold_start')"
    )
    confidence: Literal["low", "medium", "high"] = Field(
        default="medium", description="Confidence level based on state quality"
    )

    # Risk scoring (v0.4.0)
    risk_score: float | None = Field(
        default=None, description="Bounded risk score in [0, 1]"
    )
    risk_level: Literal["unknown", "low", "medium", "high", "critical"] = Field(
        default="unknown", description="Categorical risk level"
    )

    # Policy (v0.4.0)
    event_type: Literal["attempted", "applied", "blocked"] = Field(
        default="applied",
        description="Whether event was attempted, applied, or blocked",
    )
    policy_action: Literal["allow", "verify", "quarantine", "block"] | None = Field(
        default=None, description="Policy action taken"
    )


class SummaryReport(BaseModel):
    """Aggregated metrics report."""

    total_events: int
    unique_senders: int
    unique_receivers: int
    avg_drift_delta: float
    avg_influence_score: float
    high_drift_count: int
    high_influence_count: int
    top_influencers: list[tuple[str, float]]
    top_susceptible: list[tuple[str, float]]
    influence_chains: list[list[str]]


class AgentCapabilities(BaseModel):
    """Agent capability registration for attack surface computation."""

    agent_id: str
    capabilities: list[str] = Field(default_factory=list)
    attack_surface: float | None = Field(
        default=None, description="Computed attack surface score"
    )


class PropagationRiskResult(BaseModel):
    """Result of propagation risk computation over a time window."""

    window_start: datetime
    window_end: datetime
    spectral_radius: float = Field(description="Largest absolute eigenvalue")
    edge_count: int = Field(description="Number of edges in adjacency matrix")
    agent_count: int = Field(description="Number of agents in adjacency matrix")
