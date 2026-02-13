"""In-memory storage backend."""

from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID

from traceiq.storage.base import StorageBackend

if TYPE_CHECKING:
    from traceiq.models import InteractionEvent, ScoreResult


class MemoryStorage(StorageBackend):
    """In-memory storage using dicts and lists."""

    def __init__(self) -> None:
        self._events: dict[UUID, InteractionEvent] = {}
        self._scores: dict[UUID, ScoreResult] = {}
        self._events_by_sender: dict[str, list[UUID]] = {}
        self._events_by_receiver: dict[str, list[UUID]] = {}

    def store_event(self, event: InteractionEvent) -> None:
        self._events[event.event_id] = event

        if event.sender_id not in self._events_by_sender:
            self._events_by_sender[event.sender_id] = []
        self._events_by_sender[event.sender_id].append(event.event_id)

        if event.receiver_id not in self._events_by_receiver:
            self._events_by_receiver[event.receiver_id] = []
        self._events_by_receiver[event.receiver_id].append(event.event_id)

    def store_score(self, score: ScoreResult) -> None:
        self._scores[score.event_id] = score

    def get_event(self, event_id: UUID) -> InteractionEvent | None:
        return self._events.get(event_id)

    def get_score(self, event_id: UUID) -> ScoreResult | None:
        return self._scores.get(event_id)

    def get_events_by_sender(self, sender_id: str) -> list[InteractionEvent]:
        event_ids = self._events_by_sender.get(sender_id, [])
        return [self._events[eid] for eid in event_ids if eid in self._events]

    def get_events_by_receiver(self, receiver_id: str) -> list[InteractionEvent]:
        event_ids = self._events_by_receiver.get(receiver_id, [])
        return [self._events[eid] for eid in event_ids if eid in self._events]

    def get_all_events(self) -> list[InteractionEvent]:
        return sorted(self._events.values(), key=lambda e: e.timestamp)

    def get_all_scores(self) -> list[ScoreResult]:
        return list(self._scores.values())

    def get_recent_events_for_receiver(
        self, receiver_id: str, limit: int
    ) -> list[InteractionEvent]:
        events = self.get_events_by_receiver(receiver_id)
        sorted_events = sorted(events, key=lambda e: e.timestamp, reverse=True)
        return sorted_events[:limit]

    def close(self) -> None:
        pass
