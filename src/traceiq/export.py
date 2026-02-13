"""CSV and JSONL export functions."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from traceiq.models import InteractionEvent, ScoreResult


def export_events_csv(
    events: list[InteractionEvent],
    output_path: str | Path,
) -> None:
    """Export events to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["event_id", "sender_id", "receiver_id", "content", "timestamp", "metadata"]
        )
        for event in events:
            writer.writerow([
                str(event.event_id),
                event.sender_id,
                event.receiver_id,
                event.content,
                event.timestamp.isoformat(),
                json.dumps(event.metadata),
            ])


def export_scores_csv(
    scores: list[ScoreResult],
    output_path: str | Path,
) -> None:
    """Export scores to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_id",
            "influence_score",
            "drift_delta",
            "receiver_baseline_drift",
            "flags",
            "cold_start",
        ])
        for score in scores:
            writer.writerow([
                str(score.event_id),
                score.influence_score,
                score.drift_delta,
                score.receiver_baseline_drift,
                ",".join(score.flags),
                score.cold_start,
            ])


def export_combined_csv(
    events: list[InteractionEvent],
    scores: list[ScoreResult],
    output_path: str | Path,
) -> None:
    """Export combined events and scores to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score_map = {s.event_id: s for s in scores}

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "event_id",
            "sender_id",
            "receiver_id",
            "content",
            "timestamp",
            "influence_score",
            "drift_delta",
            "receiver_baseline_drift",
            "flags",
            "cold_start",
            "metadata",
        ])
        for event in events:
            score = score_map.get(event.event_id)
            writer.writerow([
                str(event.event_id),
                event.sender_id,
                event.receiver_id,
                event.content,
                event.timestamp.isoformat(),
                score.influence_score if score else "",
                score.drift_delta if score else "",
                score.receiver_baseline_drift if score else "",
                ",".join(score.flags) if score else "",
                score.cold_start if score else "",
                json.dumps(event.metadata),
            ])


def export_events_jsonl(
    events: list[InteractionEvent],
    output_path: str | Path,
) -> None:
    """Export events to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for event in events:
            record = {
                "event_id": str(event.event_id),
                "sender_id": event.sender_id,
                "receiver_id": event.receiver_id,
                "content": event.content,
                "timestamp": event.timestamp.isoformat(),
                "metadata": event.metadata,
            }
            f.write(json.dumps(record) + "\n")


def export_scores_jsonl(
    scores: list[ScoreResult],
    output_path: str | Path,
) -> None:
    """Export scores to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for score in scores:
            record = {
                "event_id": str(score.event_id),
                "influence_score": score.influence_score,
                "drift_delta": score.drift_delta,
                "receiver_baseline_drift": score.receiver_baseline_drift,
                "flags": score.flags,
                "cold_start": score.cold_start,
            }
            f.write(json.dumps(record) + "\n")


def export_combined_jsonl(
    events: list[InteractionEvent],
    scores: list[ScoreResult],
    output_path: str | Path,
) -> None:
    """Export combined events and scores to JSONL file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    score_map = {s.event_id: s for s in scores}

    with open(output_path, "w", encoding="utf-8") as f:
        for event in events:
            score = score_map.get(event.event_id)
            record = {
                "event_id": str(event.event_id),
                "sender_id": event.sender_id,
                "receiver_id": event.receiver_id,
                "content": event.content,
                "timestamp": event.timestamp.isoformat(),
                "metadata": event.metadata,
            }
            if score:
                record.update({
                    "influence_score": score.influence_score,
                    "drift_delta": score.drift_delta,
                    "receiver_baseline_drift": score.receiver_baseline_drift,
                    "flags": score.flags,
                    "cold_start": score.cold_start,
                })
            f.write(json.dumps(record) + "\n")
