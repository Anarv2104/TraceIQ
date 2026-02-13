"""Pydantic models for TraceIQ."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class InteractionEvent(BaseModel):
    """Represents a single interaction between agents."""

    event_id: UUID = Field(default_factory=uuid4)
    sender_id: str
    receiver_id: str
    content: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class TrackerConfig(BaseModel):
    """Configuration for the InfluenceTracker."""

    # Storage settings
    storage_backend: str = "memory"  # "memory" or "sqlite"
    storage_path: str | None = None  # Path for sqlite db

    # Embedding settings
    embedding_model: str = "all-MiniLM-L6-v2"
    max_content_length: int = 512
    embedding_cache_size: int = 10000

    # Scoring settings
    baseline_window: int = 10  # Number of recent embeddings for baseline
    drift_threshold: float = 0.3  # Flag if drift_delta > threshold
    influence_threshold: float = 0.5  # Flag if influence_score > threshold

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
