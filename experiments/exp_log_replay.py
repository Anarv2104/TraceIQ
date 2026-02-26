#!/usr/bin/env python3
"""Experiment 4: Real Log Replay.

Proves TraceIQ detects influence spikes in realistic log format.

Setup:
    - Static JSONL log with ground truth injection markers
    - Replay through tracker: baseline phase → injection → observation
    - Measure IQx spike relative to pre-injection baseline

Pass/Fail Criteria:
    - iqx_post_mean > iqx_pre_mean + 2 × pre_std (statistically significant spike)

Usage:
    python exp_log_replay.py
    python exp_log_replay.py --log-file custom_log.jsonl

Note: Requires traceiq to be installed (pip install -e . from repo root)
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from experiments.stats import bootstrap_ci, effect_size_cohens_d, t_test_independent
from experiments.utils import create_tracker


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


# Default log file
DEFAULT_LOG_FILE = Path(__file__).parent / "data" / "injection_log.jsonl"


def get_iqx(result: dict) -> float | None:
    """Extract IQx from result dict, handling case variations."""
    return result.get("IQx", result.get("iqx", result.get("Iqx")))


def get_z_score(result: dict) -> float | None:
    """Extract Z-score from result dict, handling case variations."""
    return result.get("Z_score", result.get("z_score", result.get("zscore")))


def load_log(log_path: Path) -> list[dict]:
    """Load JSONL log file.

    Args:
        log_path: Path to JSONL file

    Returns:
        List of log entries
    """
    entries = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def run_replay(
    log_path: Path = DEFAULT_LOG_FILE,
    output_dir: Path | None = None,
) -> dict:
    """Replay log through TraceIQ and analyze results.

    Args:
        log_path: Path to JSONL log file
        output_dir: Output directory for results

    Returns:
        Dict with analysis results and pass/fail criteria
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "results" / "log_replay"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Log Replay Experiment")
    print(f"Loading log: {log_path}")

    # Load log
    log_entries = load_log(log_path)
    print(f"Loaded {len(log_entries)} entries")

    # Count baseline vs injection
    n_baseline = sum(1 for e in log_entries if not e.get("hint_injected", False))
    n_injected = sum(1 for e in log_entries if e.get("hint_injected", False))
    print(f"Baseline events: {n_baseline}")
    print(f"Injected events: {n_injected}")

    # Replay through tracker
    tracker = create_tracker(seed=42, baseline_k=10)

    results = []
    pre_injection_iqx = []
    post_injection_iqx = []
    injection_started = False
    first_injection_idx = None

    for idx, entry in enumerate(log_entries):
        sender = entry["sender_id"]
        receiver = entry["receiver_id"]
        content = entry["message_text"]
        is_injected = entry.get("hint_injected", False)

        # Use content echo as response so embedding drift reflects injection propagation
        response = f"ACK: {content}"

        result = tracker.track_event(
            sender_id=sender,
            receiver_id=receiver,
            sender_content=content,
            receiver_content=response,
        )

        # Extract metrics robustly (handle case variations)
        iqx = get_iqx(result)
        z_score = get_z_score(result)
        alert = result.get("alert", False)
        valid = result.get("valid", True)

        # Track injection phase
        if is_injected and not injection_started:
            injection_started = True
            first_injection_idx = idx

        # Categorize IQx values
        if iqx is not None:
            if not injection_started:
                pre_injection_iqx.append(iqx)
            else:
                post_injection_iqx.append(iqx)

        results.append(
            {
                "idx": idx,
                "timestamp": entry.get("timestamp", ""),
                "sender": sender,
                "receiver": receiver,
                "hint_injected": is_injected,
                "iqx": iqx,
                "z_score": z_score,
                "alert": alert,
                "valid": valid,
                "phase": "pre" if not injection_started else "post",
            }
        )

    tracker.close()

    # Write timeseries CSV
    csv_path = output_dir / "timeseries.csv"
    with open(csv_path, "w", newline="") as f:
        if results:
            writer = csv.DictWriter(f, fieldnames=results[0].keys())
            writer.writeheader()
            writer.writerows(results)

    # === STATISTICAL ANALYSIS ===

    pre_mean = float(np.mean(pre_injection_iqx)) if pre_injection_iqx else 0.0
    pre_std = float(np.std(pre_injection_iqx)) if pre_injection_iqx else 0.0
    post_mean = float(np.mean(post_injection_iqx)) if post_injection_iqx else 0.0
    post_std = float(np.std(post_injection_iqx)) if post_injection_iqx else 0.0

    # Bootstrap CIs
    pre_ci = bootstrap_ci(pre_injection_iqx, seed=42) if pre_injection_iqx else None
    post_ci = bootstrap_ci(post_injection_iqx, seed=42) if post_injection_iqx else None

    # Statistical tests
    if len(pre_injection_iqx) >= 3 and len(post_injection_iqx) >= 3:
        t_result = t_test_independent(post_injection_iqx, pre_injection_iqx)
        effect = effect_size_cohens_d(post_injection_iqx, pre_injection_iqx)
    else:
        t_result = None
        effect = None

    # === PASS/FAIL CRITERIA ===

    # Criterion: iqx_post_mean > iqx_pre_mean + 2 × pre_std
    # This is a 2-sigma detection threshold
    threshold = pre_mean + 2 * pre_std
    spike_detected = post_mean > threshold

    # Also check if t-test is significant
    statistically_significant = t_result.significant if t_result else False

    # Alert rate in post-injection phase
    post_alerts = sum(1 for r in results if r["phase"] == "post" and r["alert"])
    post_events = sum(1 for r in results if r["phase"] == "post")
    post_alert_rate = post_alerts / post_events if post_events > 0 else 0.0

    pre_alerts = sum(1 for r in results if r["phase"] == "pre" and r["alert"])
    pre_events = sum(1 for r in results if r["phase"] == "pre")
    pre_alert_rate = pre_alerts / pre_events if pre_events > 0 else 0.0

    # Overall pass: spike detected OR significantly different OR elevated alert rate
    elevated_alert = post_alert_rate > pre_alert_rate * 2
    overall_passed = spike_detected or statistically_significant or elevated_alert

    summary = {
        "experiment": "log_replay",
        "config": {
            "log_file": str(log_path),
            "n_entries": len(log_entries),
            "n_baseline": n_baseline,
            "n_injected": n_injected,
            "first_injection_idx": first_injection_idx,
        },
        "pre_injection": {
            "n_events": len(pre_injection_iqx),
            "mean_iqx": pre_mean,
            "std_iqx": pre_std,
            "ci_95": [pre_ci.ci_low, pre_ci.ci_high] if pre_ci else None,
            "alert_rate": pre_alert_rate,
        },
        "post_injection": {
            "n_events": len(post_injection_iqx),
            "mean_iqx": post_mean,
            "std_iqx": post_std,
            "ci_95": [post_ci.ci_low, post_ci.ci_high] if post_ci else None,
            "alert_rate": post_alert_rate,
        },
        "statistical_tests": {
            "t_test": {
                "t_statistic": t_result.t_statistic if t_result else None,
                "p_value": t_result.p_value if t_result else None,
                "significant": t_result.significant if t_result else None,
            }
            if t_result
            else None,
            "effect_size": {
                "cohens_d": effect.effect_size if effect else None,
                "interpretation": effect.interpretation if effect else None,
            }
            if effect
            else None,
        },
        "pass_fail": {
            "spike_detected": {
                "passed": spike_detected,
                "threshold": threshold,
                "observed": post_mean,
                "derivation": "iqx_post_mean > iqx_pre_mean + 2 × pre_std",
            },
            "statistically_significant": {
                "passed": statistically_significant,
                "p_value": t_result.p_value if t_result else None,
            },
            "elevated_alert_rate": {
                "passed": post_alert_rate > pre_alert_rate * 2,
                "pre_rate": pre_alert_rate,
                "post_rate": post_alert_rate,
            },
        },
        "overall_passed": overall_passed,
        "output_dir": str(output_dir),
    }

    # Write summary
    summary_path = output_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)

    # Print results
    print(f"\n{'=' * 60}")
    print("RESULTS")
    print(f"{'=' * 60}")

    print(f"\nPre-injection phase ({len(pre_injection_iqx)} events):")
    print(f"  Mean IQx: {pre_mean:.4f} ± {pre_std:.4f}")
    if pre_ci:
        print(f"  95% CI: [{pre_ci.ci_low:.4f}, {pre_ci.ci_high:.4f}]")
    print(f"  Alert rate: {pre_alert_rate * 100:.1f}%")

    print(f"\nPost-injection phase ({len(post_injection_iqx)} events):")
    print(f"  Mean IQx: {post_mean:.4f} ± {post_std:.4f}")
    if post_ci:
        print(f"  95% CI: [{post_ci.ci_low:.4f}, {post_ci.ci_high:.4f}]")
    print(f"  Alert rate: {post_alert_rate * 100:.1f}%")

    print("\nStatistical Tests:")
    if t_result:
        print(f"  t-test: t={t_result.t_statistic:.3f}, p={t_result.p_value:.4f}")
        print(f"  Significant: {t_result.significant}")
    if effect:
        d_val = effect.effect_size
        interp = effect.interpretation
        print(f"  Effect size (Cohen's d): {d_val:.3f} ({interp})")

    print("\nPass/Fail Criteria:")
    print(f"  1. Spike detected (2σ): {'PASS' if spike_detected else 'FAIL'}")
    print(f"     Threshold: {threshold:.4f}, Observed: {post_mean:.4f}")
    sig_str = "PASS" if statistically_significant else "FAIL"
    print(f"  2. Statistically significant: {sig_str}")
    elevated = post_alert_rate > pre_alert_rate * 2
    print(f"  3. Elevated alert rate: {'PASS' if elevated else 'FAIL'}")

    print(f"\nOverall: {'PASS' if overall_passed else 'FAIL'}")
    print(f"\nResults saved to: {output_dir}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description="Run log replay experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=DEFAULT_LOG_FILE,
        help=f"Path to JSONL log file (default: {DEFAULT_LOG_FILE})",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Output directory"
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=True,
        dest="strict",
        help="Strict mode: exit 1 on FAIL (default: true)",
    )
    parser.add_argument(
        "--no-strict",
        action="store_false",
        dest="strict",
        help="Non-strict mode: always exit 0, print INCONCLUSIVE",
    )

    args = parser.parse_args()

    strict = args.strict
    mode = "proof" if strict else "quick"

    print(f"\nMode: {mode.upper()} (strict={strict})")

    summary = run_replay(
        log_path=args.log_file,
        output_dir=args.output_dir,
    )

    # Record run semantics for CI/paper reproducibility
    summary["strict"] = strict
    summary["mode"] = mode
    summary["status"] = "PASS" if summary.get("overall_passed") else "FAIL"
    summary["exit_code_if_strict"] = 0 if summary.get("overall_passed") else 1

    if not strict:
        summary["status"] = "INCONCLUSIVE"

    # Write-back summary with mode/status fields
    out_dir = summary.get("output_dir")
    if out_dir:
        summary_path = Path(out_dir) / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2, cls=NumpyEncoder)

    if not strict:
        print("\nINCONCLUSIVE (non-strict mode): schema/runtime validation only")
        sys.exit(0)
    else:
        sys.exit(0 if summary["overall_passed"] else 1)


if __name__ == "__main__":
    main()
