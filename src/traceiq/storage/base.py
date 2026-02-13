"""Abstract storage interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING
from uuid import UUID

if TYPE_CHECKING:
    from traceiq.models import InteractionEvent, ScoreResult


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    def store_event(self, event: InteractionEvent) -> None:
        """Store an interaction event."""

    @abstractmethod
    def store_score(self, score: ScoreResult) -> None:
        """Store a score result."""

    @abstractmethod
    def get_event(self, event_id: UUID) -> InteractionEvent | None:
        """Retrieve an event by ID."""

    @abstractmethod
    def get_score(self, event_id: UUID) -> ScoreResult | None:
        """Retrieve a score by event ID."""

    @abstractmethod
    def get_events_by_sender(self, sender_id: str) -> list[InteractionEvent]:
        """Get all events from a sender."""

    @abstractmethod
    def get_events_by_receiver(self, receiver_id: str) -> list[InteractionEvent]:
        """Get all events received by a receiver."""

    @abstractmethod
    def get_all_events(self) -> list[InteractionEvent]:
        """Get all stored events."""

    @abstractmethod
    def get_all_scores(self) -> list[ScoreResult]:
        """Get all stored scores."""

    @abstractmethod
    def get_recent_events_for_receiver(
        self, receiver_id: str, limit: int
    ) -> list[InteractionEvent]:
        """Get most recent events for a receiver, ordered by timestamp desc."""

    @abstractmethod
    def close(self) -> None:
        """Close the storage backend."""
