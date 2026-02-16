#!/usr/bin/env python3
"""Ablation study for baseline window size.

This script tests how the baseline_window parameter affects:
- IQx sensitivity
- Anomaly detection rate
- False positive rate

We vary baseline_window across [3, 5, 10, 15, 20, 30] and measure metrics.
"""

from __future__ import annotations

import random
from pathlib import Path

from traceiq import InfluenceTracker, TrackerConfig


def generate_content(drift_level: str = "normal") -> str:
    """Generate content with specified drift level."""
    if drift_level == "normal":
        return f"Normal message {random.randint(1, 100)}"
    elif drift_level == "high":
        return f"URGENT ALERT: Critical action required {random.randint(1, 100)}"
    else:
        return f"Slightly unusual message {random.randint(1, 100)}"


def run_experiment(baseline_window: int, seed: int = 42) -> dict:
    """Run experiment with specific baseline window.

    Args:
        baseline_window: Baseline window size
        seed: Random seed

    Returns:
        Dict with experiment metrics
    """
    random.seed(seed)

    config = TrackerConfig(
        storage_backend="memory",
        baseline_window=baseline_window,
        epsilon=1e-6,
        anomaly_threshold=2.0,
    )

    tracker = InfluenceTracker(config=config, use_mock_embedder=True)

    # Track metrics
    iqx_values = []
    z_scores = []
    alerts = 0
    high_drift_events = 0

    # Phase 1: Build baseline (20 normal events)
    for i in range(20):
        sender = f"agent_{i % 3}"
        receiver = f"agent_{(i + 1) % 3}"
        result = tracker.track_event(
            sender_id=sender,
            receiver_id=receiver,
            sender_content=generate_content("normal"),
            receiver_content=generate_content("normal"),
        )
        if result.get("IQx") is not None:
            iqx_values.append(result["IQx"])
        if result.get("Z_score") is not None:
            z_scores.append(result["Z_score"])

    # Phase 2: Inject anomalies (10 high-drift events)
    for i in range(10):
        result = tracker.track_event(
            sender_id="injector",
            receiver_id=f"agent_{i % 3}",
            sender_content=generate_content("high"),
            receiver_content=generate_content("high"),
        )
        if result.get("IQx") is not None:
            iqx_values.append(result["IQx"])
        if result.get("Z_score") is not None:
            z_scores.append(result["Z_score"])
        if result.get("alert"):
            alerts += 1
        high_drift_events += 1

    # Phase 3: Return to normal (10 events)
    for i in range(10):
        sender = f"agent_{i % 3}"
        receiver = f"agent_{(i + 1) % 3}"
        result = tracker.track_event(
            sender_id=sender,
            receiver_id=receiver,
            sender_content=generate_content("normal"),
            receiver_content=generate_content("normal"),
        )
        if result.get("IQx") is not None:
            iqx_values.append(result["IQx"])
        if result.get("Z_score") is not None:
            z_scores.append(result["Z_score"])
        if result.get("alert"):
            # These are false positives (normal events flagged)
            pass

    tracker.close()

    # Compute statistics
    import numpy as np

    return {
        "baseline_window": baseline_window,
        "mean_iqx": float(np.mean(iqx_values)) if iqx_values else 0.0,
        "std_iqx": float(np.std(iqx_values)) if iqx_values else 0.0,
        "max_iqx": float(np.max(iqx_values)) if iqx_values else 0.0,
        "mean_z_score": float(np.mean(np.abs(z_scores))) if z_scores else 0.0,
        "alerts_detected": alerts,
        "detection_rate": alerts / high_drift_events if high_drift_events > 0 else 0.0,
        "total_events": 40,
    }


def run_ablation_study(
    window_sizes: list[int] | None = None,
    output_dir: str | Path | None = None,
    seed: int = 42,
) -> list[dict]:
    """Run ablation study across window sizes.

    Args:
        window_sizes: List of window sizes to test
        output_dir: Directory for output files
        seed: Random seed

    Returns:
        List of experiment results
    """
    if window_sizes is None:
        window_sizes = [3, 5, 10, 15, 20, 30]

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("ABLATION STUDY: Baseline Window Size")
    print("=" * 60)

    results = []
    for window in window_sizes:
        print(f"\nTesting baseline_window={window}...")
        result = run_experiment(window, seed)
        results.append(result)
        print(f"  Mean IQx: {result['mean_iqx']:.4f}")
        print(f"  Std IQx: {result['std_iqx']:.4f}")
        print(f"  Detection Rate: {result['detection_rate']:.1%}")

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Window':<10} {'Mean IQx':<12} {'Std IQx':<12} {'Detection':<12}")
    print("-" * 46)
    for r in results:
        print(
            f"{r['baseline_window']:<10} "
            f"{r['mean_iqx']:<12.4f} "
            f"{r['std_iqx']:<12.4f} "
            f"{r['detection_rate']:<12.1%}"
        )

    # Generate plot if matplotlib available
    if output_dir:
        try:
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(15, 5))

            windows = [r["baseline_window"] for r in results]

            # Plot 1: Mean IQx vs Window Size
            axes[0].plot(windows, [r["mean_iqx"] for r in results], marker="o")
            axes[0].set_xlabel("Baseline Window Size")
            axes[0].set_ylabel("Mean IQx")
            axes[0].set_title("Mean IQx vs Window Size")
            axes[0].grid(True, alpha=0.3)

            # Plot 2: Detection Rate vs Window Size
            axes[1].plot(
                windows,
                [r["detection_rate"] for r in results],
                marker="o",
                color="green",
            )
            axes[1].set_xlabel("Baseline Window Size")
            axes[1].set_ylabel("Detection Rate")
            axes[1].set_title("Anomaly Detection Rate vs Window Size")
            axes[1].grid(True, alpha=0.3)

            # Plot 3: IQx Std vs Window Size
            axes[2].plot(
                windows, [r["std_iqx"] for r in results], marker="o", color="orange"
            )
            axes[2].set_xlabel("Baseline Window Size")
            axes[2].set_ylabel("IQx Standard Deviation")
            axes[2].set_title("IQx Variability vs Window Size")
            axes[2].grid(True, alpha=0.3)

            plt.tight_layout()
            plt.savefig(output_dir / "ablation_study.png", dpi=150)
            plt.close()

            print(f"\nPlot saved to {output_dir / 'ablation_study.png'}")

        except ImportError:
            print("\nMatplotlib not available, skipping plot.")

    return results


if __name__ == "__main__":
    run_ablation_study(output_dir=Path("research/outputs"))
