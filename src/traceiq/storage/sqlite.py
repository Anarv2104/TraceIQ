"""SQLite storage backend."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from traceiq.models import InteractionEvent, ScoreResult
from traceiq.storage.base import StorageBackend

if TYPE_CHECKING:
    pass


class SQLiteStorage(StorageBackend):
    """SQLite-based persistent storage."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._check_schema_compatibility()
        self._create_tables()

    def _create_tables(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS events (
                event_id TEXT PRIMARY KEY,
                sender_id TEXT NOT NULL,
                receiver_id TEXT NOT NULL,
                sender_content TEXT NOT NULL,
                receiver_content TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                metadata_json TEXT NOT NULL
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_sender ON events(sender_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_receiver ON events(receiver_id)"
        )
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp)"
        )

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scores (
                event_id TEXT PRIMARY KEY,
                influence_score REAL NOT NULL,
                drift_delta REAL NOT NULL,
                receiver_baseline_drift REAL NOT NULL,
                flags_json TEXT NOT NULL,
                cold_start INTEGER NOT NULL
            )
        """)
        self._conn.commit()

    def _check_schema_compatibility(self) -> None:
        """Check database schema is compatible with v0.2.0.

        Raises RuntimeError if an incompatible old schema is detected.
        """
        cursor = self._conn.cursor()
        cursor.execute("PRAGMA table_info(events)")
        columns = {row[1] for row in cursor.fetchall()}

        # Check if old schema (single 'content' column) exists
        if "content" in columns and "sender_content" not in columns:
            raise RuntimeError(
                "Database schema is from an older TraceIQ version and is incompatible. "
                "Export data and re-ingest with receiver_content."
            )

    def _parse_timestamp(self, ts_str: str) -> datetime:
        """Parse timestamp string correctly handling timezone info."""
        dt = datetime.fromisoformat(ts_str)
        if dt.tzinfo is None:
            # Naive datetime - assume UTC
            return dt.replace(tzinfo=timezone.utc)
        # Already has timezone info - convert to UTC
        return dt.astimezone(timezone.utc)

    def store_event(self, event: InteractionEvent) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO events
            (event_id, sender_id, receiver_id, sender_content, receiver_content, timestamp, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(event.event_id),
                event.sender_id,
                event.receiver_id,
                event.sender_content,
                event.receiver_content,
                event.timestamp.isoformat(),
                json.dumps(event.metadata),
            ),
        )
        self._conn.commit()

    def store_score(self, score: ScoreResult) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO scores
            (event_id, influence_score, drift_delta, receiver_baseline_drift, flags_json, cold_start)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                str(score.event_id),
                score.influence_score,
                score.drift_delta,
                score.receiver_baseline_drift,
                json.dumps(score.flags),
                1 if score.cold_start else 0,
            ),
        )
        self._conn.commit()

    def _row_to_event(self, row: sqlite3.Row) -> InteractionEvent:
        return InteractionEvent(
            event_id=UUID(row["event_id"]),
            sender_id=row["sender_id"],
            receiver_id=row["receiver_id"],
            sender_content=row["sender_content"],
            receiver_content=row["receiver_content"],
            timestamp=self._parse_timestamp(row["timestamp"]),
            metadata=json.loads(row["metadata_json"]),
        )

    def _row_to_score(self, row: sqlite3.Row) -> ScoreResult:
        return ScoreResult(
            event_id=UUID(row["event_id"]),
            influence_score=row["influence_score"],
            drift_delta=row["drift_delta"],
            receiver_baseline_drift=row["receiver_baseline_drift"],
            flags=json.loads(row["flags_json"]),
            cold_start=bool(row["cold_start"]),
        )

    def get_event(self, event_id: UUID) -> InteractionEvent | None:
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM events WHERE event_id = ?", (str(event_id),))
        row = cursor.fetchone()
        return self._row_to_event(row) if row else None

    def get_score(self, event_id: UUID) -> ScoreResult | None:
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM scores WHERE event_id = ?", (str(event_id),))
        row = cursor.fetchone()
        return self._row_to_score(row) if row else None

    def get_events_by_sender(self, sender_id: str) -> list[InteractionEvent]:
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM events WHERE sender_id = ? ORDER BY timestamp",
            (sender_id,),
        )
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def get_events_by_receiver(self, receiver_id: str) -> list[InteractionEvent]:
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT * FROM events WHERE receiver_id = ? ORDER BY timestamp",
            (receiver_id,),
        )
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def get_all_events(self) -> list[InteractionEvent]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM events ORDER BY timestamp")
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def get_all_scores(self) -> list[ScoreResult]:
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM scores")
        return [self._row_to_score(row) for row in cursor.fetchall()]

    def get_recent_events_for_receiver(
        self, receiver_id: str, limit: int
    ) -> list[InteractionEvent]:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            SELECT * FROM events
            WHERE receiver_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """,
            (receiver_id, limit),
        )
        return [self._row_to_event(row) for row in cursor.fetchall()]

    def close(self) -> None:
        self._conn.close()
