"""Tests for export module."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from traceiq.export import (
    export_combined_csv,
    export_combined_jsonl,
    export_events_csv,
    export_events_jsonl,
    export_scores_csv,
    export_scores_jsonl,
)
from traceiq.models import InteractionEvent, ScoreResult


class TestExportCSV:
    """Tests for CSV export functions."""

    def test_export_events_csv(
        self, sample_events: list[InteractionEvent], tmp_path: Path
    ) -> None:
        """Test exporting events to CSV."""
        output_path = tmp_path / "events.csv"
        export_events_csv(sample_events, output_path)

        assert output_path.exists()

        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == len(sample_events)
        assert rows[0]["sender_id"] == sample_events[0].sender_id
        assert rows[0]["receiver_id"] == sample_events[0].receiver_id
        assert rows[0]["sender_content"] == sample_events[0].sender_content
        assert rows[0]["receiver_content"] == sample_events[0].receiver_content

    def test_export_scores_csv(
        self, sample_scores: list[ScoreResult], tmp_path: Path
    ) -> None:
        """Test exporting scores to CSV."""
        output_path = tmp_path / "scores.csv"
        export_scores_csv(sample_scores, output_path)

        assert output_path.exists()

        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == len(sample_scores)
        assert float(rows[0]["influence_score"]) == pytest.approx(
            sample_scores[0].influence_score
        )

    def test_export_combined_csv(
        self,
        sample_events: list[InteractionEvent],
        sample_scores: list[ScoreResult],
        tmp_path: Path,
    ) -> None:
        """Test exporting combined events and scores to CSV."""
        output_path = tmp_path / "combined.csv"
        export_combined_csv(sample_events, sample_scores, output_path)

        assert output_path.exists()

        with open(output_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == len(sample_events)

        # Check that both event and score data are present
        assert "sender_id" in rows[0]
        assert "influence_score" in rows[0]
        assert "drift_delta" in rows[0]

    def test_export_creates_parent_dirs(
        self, sample_events: list[InteractionEvent], tmp_path: Path
    ) -> None:
        """Test that export creates parent directories."""
        output_path = tmp_path / "nested" / "dir" / "events.csv"
        export_events_csv(sample_events, output_path)
        assert output_path.exists()


class TestExportJSONL:
    """Tests for JSONL export functions."""

    def test_export_events_jsonl(
        self, sample_events: list[InteractionEvent], tmp_path: Path
    ) -> None:
        """Test exporting events to JSONL."""
        output_path = tmp_path / "events.jsonl"
        export_events_jsonl(sample_events, output_path)

        assert output_path.exists()

        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == len(sample_events)

        first_record = json.loads(lines[0])
        assert first_record["sender_id"] == sample_events[0].sender_id
        assert first_record["receiver_id"] == sample_events[0].receiver_id

    def test_export_scores_jsonl(
        self, sample_scores: list[ScoreResult], tmp_path: Path
    ) -> None:
        """Test exporting scores to JSONL."""
        output_path = tmp_path / "scores.jsonl"
        export_scores_jsonl(sample_scores, output_path)

        assert output_path.exists()

        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == len(sample_scores)

        first_record = json.loads(lines[0])
        assert first_record["influence_score"] == pytest.approx(
            sample_scores[0].influence_score
        )

    def test_export_combined_jsonl(
        self,
        sample_events: list[InteractionEvent],
        sample_scores: list[ScoreResult],
        tmp_path: Path,
    ) -> None:
        """Test exporting combined events and scores to JSONL."""
        output_path = tmp_path / "combined.jsonl"
        export_combined_jsonl(sample_events, sample_scores, output_path)

        assert output_path.exists()

        with open(output_path, encoding="utf-8") as f:
            lines = f.readlines()

        assert len(lines) == len(sample_events)

        first_record = json.loads(lines[0])
        assert "sender_id" in first_record
        assert "influence_score" in first_record

    def test_jsonl_valid_json_per_line(
        self, sample_events: list[InteractionEvent], tmp_path: Path
    ) -> None:
        """Test that each line in JSONL is valid JSON."""
        output_path = tmp_path / "events.jsonl"
        export_events_jsonl(sample_events, output_path)

        with open(output_path, encoding="utf-8") as f:
            for line in f:
                # Should not raise
                record = json.loads(line)
                assert isinstance(record, dict)

    def test_export_empty_list(self, tmp_path: Path) -> None:
        """Test exporting empty lists."""
        events_path = tmp_path / "empty_events.csv"
        export_events_csv([], events_path)

        with open(events_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        assert len(rows) == 0

    def test_export_preserves_metadata(
        self, sample_events: list[InteractionEvent], tmp_path: Path
    ) -> None:
        """Test that metadata is preserved in export."""
        output_path = tmp_path / "events.jsonl"
        export_events_jsonl(sample_events, output_path)

        with open(output_path, encoding="utf-8") as f:
            first_record = json.loads(f.readline())

        assert "metadata" in first_record
        assert first_record["metadata"] == sample_events[0].metadata
