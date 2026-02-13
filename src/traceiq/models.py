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
