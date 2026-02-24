#!/usr/bin/env python3
"""MCP Server for TraceIQ.

A simple JSON-over-stdio server that exposes TraceIQ functionality
via a request/response protocol.

Protocol:
- Each request/response is a single JSON line
- Request format: {"method": "method_name", "params": {...}}
- Response format: {"result": ...} or {"error": "message"}

Available methods:
- log_interaction: Track an interaction event
- summary: Get summary report
- export_csv: Export data to CSV file
- get_alerts: Get anomaly alerts
- propagation_risk: Get current propagation risk

Usage:
    echo '{"method":"propagation_risk"}' | python mcp_server_traceiq.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from traceiq import InfluenceTracker, TrackerConfig


def create_server_tracker() -> InfluenceTracker:
    """Create an InfluenceTracker for the MCP server.

    Returns:
        Configured InfluenceTracker instance
    """
    config = TrackerConfig(
        storage_backend="memory",
        baseline_window=10,
        epsilon=1e-6,
        anomaly_threshold=2.0,
        random_seed=42,
    )
    return InfluenceTracker(config=config, use_mock_embedder=True)


def handle_log_interaction(tracker: InfluenceTracker, params: dict) -> dict:
    """Handle log_interaction method.

    Args:
        tracker: InfluenceTracker instance
        params: Request parameters

    Returns:
        Result dict from track_event
    """
    required = ["sender_id", "receiver_id", "sender_content", "receiver_content"]
    for key in required:
        if key not in params:
            raise ValueError(f"Missing required parameter: {key}")

    result = tracker.track_event(
        sender_id=params["sender_id"],
        receiver_id=params["receiver_id"],
        sender_content=params["sender_content"],
        receiver_content=params["receiver_content"],
        metadata=params.get("metadata"),
    )

    return result


def handle_summary(tracker: InfluenceTracker, params: dict) -> dict:
    """Handle summary method.

    Args:
        tracker: InfluenceTracker instance
        params: Request parameters

    Returns:
        Summary report dict
    """
    top_n = params.get("top_n", 10)
    summary = tracker.summary(top_n=top_n)

    return {
        "total_events": summary.total_events,
        "unique_senders": summary.unique_senders,
        "unique_receivers": summary.unique_receivers,
        "avg_drift_delta": summary.avg_drift_delta,
        "avg_influence_score": summary.avg_influence_score,
        "high_drift_count": summary.high_drift_count,
        "high_influence_count": summary.high_influence_count,
        "top_influencers": summary.top_influencers,
        "top_susceptible": summary.top_susceptible,
        "influence_chains": summary.influence_chains,
    }


def handle_export_csv(tracker: InfluenceTracker, params: dict) -> dict:
    """Handle export_csv method.

    Args:
        tracker: InfluenceTracker instance
        params: Request parameters

    Returns:
        Success dict with path
    """
    path = params.get("path", "traceiq_export.csv")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    tracker.export_csv(path)

    return {"success": True, "path": str(path.absolute())}


def handle_get_alerts(tracker: InfluenceTracker, params: dict) -> dict:
    """Handle get_alerts method.

    Args:
        tracker: InfluenceTracker instance
        params: Request parameters

    Returns:
        Dict with alerts list
    """
    threshold = params.get("threshold")
    alerts = tracker.get_alerts(threshold=threshold)

    return {
        "alerts": [
            {
                "event_id": str(a.event_id),
                "sender_id": a.sender_id,
                "receiver_id": a.receiver_id,
                "IQx": a.IQx,
                "Z_score": a.Z_score,
                "alert_flag": a.alert_flag,
            }
            for a in alerts
        ]
    }


def handle_propagation_risk(tracker: InfluenceTracker, params: dict) -> dict:
    """Handle propagation_risk method.

    Args:
        tracker: InfluenceTracker instance
        params: Request parameters

    Returns:
        Dict with propagation_risk value
    """
    pr = tracker.get_propagation_risk()
    return {"propagation_risk": pr}


def handle_request(request: dict, tracker: InfluenceTracker) -> dict:
    """Route and handle a request.

    Args:
        request: Request dict with "method" and optional "params"
        tracker: InfluenceTracker instance

    Returns:
        Response dict with "result" or "error"
    """
    method = request.get("method")
    params = request.get("params", {})

    handlers = {
        "log_interaction": handle_log_interaction,
        "summary": handle_summary,
        "export_csv": handle_export_csv,
        "get_alerts": handle_get_alerts,
        "propagation_risk": handle_propagation_risk,
    }

    if method not in handlers:
        return {"error": f"Unknown method: {method}. Available: {list(handlers.keys())}"}

    try:
        result = handlers[method](tracker, params)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}


def main() -> None:
    """Run the MCP server."""
    tracker = create_server_tracker()

    # Process stdin line by line
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError as e:
            response = {"error": f"Invalid JSON: {e}"}
            print(json.dumps(response), flush=True)
            continue

        response = handle_request(request, tracker)
        print(json.dumps(response), flush=True)

    tracker.close()


if __name__ == "__main__":
    main()
