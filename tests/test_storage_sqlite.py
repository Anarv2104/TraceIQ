"""Tests for SQLite storage backend."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import pytest

from traceiq.models import InteractionEvent, ScoreResult
from traceiq.storage import SQLiteStorage


class TestSQLiteStorage:
    """Tests for SQLite storage backend."""

    def test_create_tables(self, tmp_path: Path) -> None:
        """Test that tables are created on initialization."""
        db_path = tmp_path / "test.db"
        storage = SQLiteStorage(db_path)

        # Check tables exist
        cursor = storage._conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='events'"
        )
        assert cursor.fetchone() is not None

        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scores'"
        )
        assert cursor.fetchone() is not None

        storage.close()

    def test_store_and_get_event(self, sqlite_storage: SQLiteStorage) -> None:
        """Test storing and retrieving an event."""
        event = InteractionEvent(
            event_id=uuid4(),
            sender_id="agent_a",
            receiver_id="agent_b",
            sender_content="Hello world",
            receiver_content="Hi there",
            timestamp=datetime.now(timezone.utc),
            metadata={"key": "value"},
        )

        sqlite_storage.store_event(event)
        retrieved = sqlite_storage.get_event(event.event_id)

        assert retrieved is not None
        assert retrieved.event_id == event.event_id
        assert retrieved.sender_id == event.sender_id
        assert retrieved.receiver_id == event.receiver_id
        assert retrieved.sender_content == event.sender_content
        assert retrieved.receiver_content == event.receiver_content
        assert retrieved.metadata == event.metadata

    def test_store_and_get_score(self, sqlite_storage: SQLiteStorage) -> None:
        """Test storing and retrieving a score."""
        score = ScoreResult(
            event_id=uuid4(),
            influence_score=0.75,
            drift_delta=0.25,
            receiver_baseline_drift=0.1,
            flags=["high_influence"],
            cold_start=False,
        )

        sqlite_storage.store_score(score)
        retrieved = sqlite_storage.get_score(score.event_id)

        assert retrieved is not None
        assert retrieved.event_id == score.event_id
        assert retrieved.influence_score == pytest.approx(0.75)
        assert retrieved.drift_delta == pytest.approx(0.25)
        assert retrieved.flags == ["high_influence"]
        assert retrieved.cold_start is False

    def test_get_nonexistent_event(self, sqlite_storage: SQLiteStorage) -> None:
        """Test retrieving a non-existent event returns None."""
        result = sqlite_storage.get_event(uuid4())
        assert result is None

    def test_get_events_by_sender(self, sqlite_storage: SQLiteStorage) -> None:
        """Test getting events filtered by sender."""
        for i in range(5):
            event = InteractionEvent(
                sender_id="sender_a" if i % 2 == 0 else "sender_b",
                receiver_id="receiver",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
            )
            sqlite_storage.store_event(event)

        sender_a_events = sqlite_storage.get_events_by_sender("sender_a")
        assert len(sender_a_events) == 3

        sender_b_events = sqlite_storage.get_events_by_sender("sender_b")
        assert len(sender_b_events) == 2

    def test_get_events_by_receiver(self, sqlite_storage: SQLiteStorage) -> None:
        """Test getting events filtered by receiver."""
        for i in range(5):
            event = InteractionEvent(
                sender_id="sender",
                receiver_id="receiver_a" if i < 3 else "receiver_b",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
            )
            sqlite_storage.store_event(event)

        receiver_a_events = sqlite_storage.get_events_by_receiver("receiver_a")
        assert len(receiver_a_events) == 3

    def test_get_all_events(self, sqlite_storage: SQLiteStorage) -> None:
        """Test getting all events."""
        for i in range(10):
            event = InteractionEvent(
                sender_id=f"sender_{i}",
                receiver_id=f"receiver_{i}",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
            )
            sqlite_storage.store_event(event)

        all_events = sqlite_storage.get_all_events()
        assert len(all_events) == 10

    def test_get_all_scores(self, sqlite_storage: SQLiteStorage) -> None:
        """Test getting all scores."""
        for i in range(5):
            score = ScoreResult(
                event_id=uuid4(),
                influence_score=0.1 * i,
                drift_delta=0.2,
                receiver_baseline_drift=0.05,
            )
            sqlite_storage.store_score(score)

        all_scores = sqlite_storage.get_all_scores()
        assert len(all_scores) == 5

    def test_get_recent_events_for_receiver(
        self, sqlite_storage: SQLiteStorage
    ) -> None:
        """Test getting recent events for a receiver with limit."""
        from datetime import timedelta

        base_time = datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc)

        for i in range(10):
            event = InteractionEvent(
                sender_id="sender",
                receiver_id="receiver",
                sender_content=f"Message {i}",
                receiver_content=f"Response {i}",
                timestamp=base_time + timedelta(minutes=i),
            )
            sqlite_storage.store_event(event)

        recent = sqlite_storage.get_recent_events_for_receiver("receiver", limit=3)
        assert len(recent) == 3

        # Should be in descending timestamp order
        assert "Message 9" in recent[0].sender_content
        assert "Message 8" in recent[1].sender_content
        assert "Message 7" in recent[2].sender_content

    def test_upsert_event(self, sqlite_storage: SQLiteStorage) -> None:
        """Test that storing an event with same ID updates it."""
        event_id = uuid4()

        event1 = InteractionEvent(
            event_id=event_id,
            sender_id="sender",
            receiver_id="receiver",
            sender_content="Original content",
            receiver_content="Original response",
        )
        sqlite_storage.store_event(event1)

        event2 = InteractionEvent(
            event_id=event_id,
            sender_id="sender",
            receiver_id="receiver",
            sender_content="Updated content",
            receiver_content="Updated response",
        )
        sqlite_storage.store_event(event2)

        retrieved = sqlite_storage.get_event(event_id)
        assert retrieved is not None
        assert retrieved.sender_content == "Updated content"
        assert retrieved.receiver_content == "Updated response"

    def test_persistence_across_connections(self, tmp_path: Path) -> None:
        """Test that data persists after closing and reopening."""
        db_path = tmp_path / "persist_test.db"

        # Write data
        storage1 = SQLiteStorage(db_path)
        event = InteractionEvent(
            sender_id="sender",
            receiver_id="receiver",
            sender_content="Persistent message",
            receiver_content="Persistent response",
        )
        sqlite_storage_event_id = event.event_id
        storage1.store_event(event)
        storage1.close()

        # Read data with new connection
        storage2 = SQLiteStorage(db_path)
        retrieved = storage2.get_event(sqlite_storage_event_id)
        storage2.close()

        assert retrieved is not None
        assert retrieved.sender_content == "Persistent message"
        assert retrieved.receiver_content == "Persistent response"
