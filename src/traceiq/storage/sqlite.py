"""SQLite storage backend."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING
from uuid import UUID

from traceiq.models import InteractionEvent, PropagationRiskResult, ScoreResult
from traceiq.storage.base import StorageBackend

if TYPE_CHECKING:
    pass

# Schema version for migration tracking
SCHEMA_VERSION = 3


class SQLiteStorage(StorageBackend):
    """SQLite-based persistent storage."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self._conn = sqlite3.connect(str(self.db_path))
        self._conn.row_factory = sqlite3.Row
        self._check_schema_compatibility()
        self._create_tables()
        self._migrate_schema()

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
                cold_start INTEGER NOT NULL,
                drift_l2 REAL,
                drift_l2_state REAL,
                drift_l2_proxy REAL,
                IQx REAL,
                baseline_median REAL,
                RWI REAL,
                Z_score REAL,
                alert_flag INTEGER DEFAULT 0
            )
        """)

        # IEEE metrics tables (v0.3.0)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS propagation_risk (
                window_id TEXT PRIMARY KEY,
                window_start TEXT NOT NULL,
                window_end TEXT NOT NULL,
                spectral_radius REAL NOT NULL,
                edge_count INTEGER NOT NULL,
                agent_count INTEGER NOT NULL
            )
        """)
        cursor.execute(
            "CREATE INDEX IF NOT EXISTS idx_pr_window_start "
            "ON propagation_risk(window_start)"
        )

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS accumulated_influence (
                agent_id TEXT PRIMARY KEY,
                ai_value REAL NOT NULL,
                window_size INTEGER NOT NULL,
                last_updated TEXT NOT NULL
            )
        """)

        # Schema version tracking
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS schema_meta (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
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

    def _migrate_schema(self) -> None:
        """Migrate schema from older versions to current version.

        This handles backward compatibility with v0.2.0 databases.
        """
        cursor = self._conn.cursor()

        # Get current schema version
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = {row[0] for row in cursor.fetchall()}

        # Check if we have schema_meta table
        current_version = 1
        if "schema_meta" in tables:
            cursor.execute("SELECT value FROM schema_meta WHERE key = 'schema_version'")
            row = cursor.fetchone()
            if row:
                current_version = int(row[0])

        if current_version >= SCHEMA_VERSION:
            return

        # Migration from v1 to v2: Add IEEE metric columns to scores
        if current_version < 2:
            cursor.execute("PRAGMA table_info(scores)")
            score_columns = {row[1] for row in cursor.fetchall()}

            # Add missing columns
            new_columns = [
                ("drift_l2", "REAL"),
                ("IQx", "REAL"),
                ("baseline_median", "REAL"),
                ("RWI", "REAL"),
                ("Z_score", "REAL"),
                ("alert_flag", "INTEGER DEFAULT 0"),
            ]

            for col_name, col_type in new_columns:
                if col_name not in score_columns:
                    cursor.execute(
                        f"ALTER TABLE scores ADD COLUMN {col_name} {col_type}"
                    )

        # Migration from v2 to v3: Add canonical/proxy drift columns
        if current_version < 3:
            cursor.execute("PRAGMA table_info(scores)")
            score_columns = {row[1] for row in cursor.fetchall()}

            # Add new canonical and proxy drift columns
            new_columns = [
                ("drift_l2_state", "REAL"),
                ("drift_l2_proxy", "REAL"),
            ]

            for col_name, col_type in new_columns:
                if col_name not in score_columns:
                    cursor.execute(
                        f"ALTER TABLE scores ADD COLUMN {col_name} {col_type}"
                    )

        # Update schema version
        cursor.execute(
            "INSERT OR REPLACE INTO schema_meta (key, value) VALUES (?, ?)",
            ("schema_version", str(SCHEMA_VERSION)),
        )

        self._conn.commit()

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
            (event_id, influence_score, drift_delta, receiver_baseline_drift, flags_json, cold_start,
             drift_l2, drift_l2_state, drift_l2_proxy, IQx, baseline_median, RWI, Z_score, alert_flag)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                str(score.event_id),
                score.influence_score,
                score.drift_delta,
                score.receiver_baseline_drift,
                json.dumps(score.flags),
                1 if score.cold_start else 0,
                score.drift_l2,
                score.drift_l2_state,
                score.drift_l2_proxy,
                score.IQx,
                score.baseline_median,
                score.RWI,
                score.Z_score,
                1 if score.alert_flag else 0,
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
        # Handle both old and new schema
        keys = row.keys()
        return ScoreResult(
            event_id=UUID(row["event_id"]),
            influence_score=row["influence_score"],
            drift_delta=row["drift_delta"],
            receiver_baseline_drift=row["receiver_baseline_drift"],
            flags=json.loads(row["flags_json"]),
            cold_start=bool(row["cold_start"]),
            # IEEE metrics (may be None for old records)
            drift_l2_state=row["drift_l2_state"] if "drift_l2_state" in keys else None,
            drift_l2_proxy=row["drift_l2_proxy"] if "drift_l2_proxy" in keys else None,
            drift_l2=row["drift_l2"] if "drift_l2" in keys else None,
            IQx=row["IQx"] if "IQx" in keys else None,
            baseline_median=row["baseline_median"]
            if "baseline_median" in keys
            else None,
            RWI=row["RWI"] if "RWI" in keys else None,
            Z_score=row["Z_score"] if "Z_score" in keys else None,
            alert_flag=bool(row["alert_flag"]) if "alert_flag" in keys else False,
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

    # IEEE metrics storage methods (v0.3.0)

    def store_propagation_risk(
        self, window_id: str, result: PropagationRiskResult
    ) -> None:
        """Store a propagation risk computation result."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO propagation_risk
            (window_id, window_start, window_end, spectral_radius, edge_count, agent_count)
            VALUES (?, ?, ?, ?, ?, ?)
        """,
            (
                window_id,
                result.window_start.isoformat(),
                result.window_end.isoformat(),
                result.spectral_radius,
                result.edge_count,
                result.agent_count,
            ),
        )
        self._conn.commit()

    def get_propagation_risk_history(self) -> list[PropagationRiskResult]:
        """Get all stored propagation risk results."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT * FROM propagation_risk ORDER BY window_start")
        results = []
        for row in cursor.fetchall():
            results.append(
                PropagationRiskResult(
                    window_start=self._parse_timestamp(row["window_start"]),
                    window_end=self._parse_timestamp(row["window_end"]),
                    spectral_radius=row["spectral_radius"],
                    edge_count=row["edge_count"],
                    agent_count=row["agent_count"],
                )
            )
        return results

    def store_accumulated_influence(
        self, agent_id: str, ai_value: float, window_size: int
    ) -> None:
        """Store accumulated influence for an agent."""
        cursor = self._conn.cursor()
        cursor.execute(
            """
            INSERT OR REPLACE INTO accumulated_influence
            (agent_id, ai_value, window_size, last_updated)
            VALUES (?, ?, ?, ?)
        """,
            (
                agent_id,
                ai_value,
                window_size,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        self._conn.commit()

    def get_accumulated_influence(self, agent_id: str) -> float | None:
        """Get accumulated influence for an agent."""
        cursor = self._conn.cursor()
        cursor.execute(
            "SELECT ai_value FROM accumulated_influence WHERE agent_id = ?",
            (agent_id,),
        )
        row = cursor.fetchone()
        return row["ai_value"] if row else None

    def get_all_accumulated_influence(self) -> dict[str, float]:
        """Get accumulated influence for all agents."""
        cursor = self._conn.cursor()
        cursor.execute("SELECT agent_id, ai_value FROM accumulated_influence")
        return {row["agent_id"]: row["ai_value"] for row in cursor.fetchall()}

    def get_alerts(self, threshold: float | None = None) -> list[ScoreResult]:
        """Get all scores with alert_flag=True, optionally filtered by Z_score threshold."""
        cursor = self._conn.cursor()
        if threshold is not None:
            cursor.execute(
                """
                SELECT * FROM scores
                WHERE alert_flag = 1 AND ABS(Z_score) > ?
                ORDER BY ABS(Z_score) DESC
            """,
                (threshold,),
            )
        else:
            cursor.execute(
                """
                SELECT * FROM scores
                WHERE alert_flag = 1
                ORDER BY ABS(Z_score) DESC
            """
            )
        return [self._row_to_score(row) for row in cursor.fetchall()]

    def close(self) -> None:
        self._conn.close()
