"""Pydantic models for TraceIQ."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


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
    baseline_window: int = Field(default=10, gt=0)
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

    # Misc
    random_seed: int | None = None


class ScoreResult(BaseModel):
    """Score computation result for an event."""

    event_id: UUID
    influence_score: float
    drift_delta: float
    receiver_baseline_drift: float
    flags: list[str] = Field(default_factory=list)
    cold_start: bool = False

    # IEEE metrics (v0.3.0)
    drift_l2: float | None = Field(default=None, description="L2 norm drift")
    IQx: float | None = Field(default=None, description="Influence Quotient")
    baseline_median: float | None = Field(
        default=None, description="Receiver's baseline median drift"
    )
    RWI: float | None = Field(default=None, description="Risk-Weighted Influence")
    Z_score: float | None = Field(default=None, description="Anomaly Z-score")
    alert_flag: bool = Field(default=False, description="True if Z > anomaly_threshold")


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
