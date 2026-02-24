#!/usr/bin/env python3
"""Verify mathematical properties of TraceIQ metrics.

This script validates the theoretical properties of TraceIQ's influence metrics:
- IQx non-negativity
- IQx monotonicity
- IQx zero-influence property
- Propagation Risk bounds
- Z-score standardization

These property tests provide empirical verification of the mathematical
guarantees described in THEORY.md.

Usage:
    python experiments/run_metric_properties.py

Outputs:
    - Console: Property test results (PASS/FAIL)
    - experiments/results/metric_properties.csv
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from traceiq.metrics import (
    compute_drift_l2,
    compute_IQx,
    compute_propagation_risk,
    compute_z_score,
)


@dataclass
class PropertyTestResult:
    """Result of a property test."""

    name: str
    passed: bool
    n_trials: int
    failures: int
    description: str
    error_message: str | None = None


def test_iqx_non_negativity(n_trials: int = 10000, seed: int = 42) -> PropertyTestResult:
    """Property 1: IQx >= 0 for all valid inputs.

    IQx = drift / (baseline + epsilon)

    Since drift = ||·||_2 >= 0 and baseline + epsilon > 0,
    the quotient must be non-negative.
    """
    rng = np.random.default_rng(seed)
    failures = 0

    for _ in range(n_trials):
        # Generate random drift (always non-negative as L2 norm)
        drift = abs(rng.normal(0, 10))
        # Generate random baseline (can be zero or positive)
        baseline = abs(rng.normal(5, 2))
        epsilon = 1e-6

        iqx = compute_IQx(drift, baseline, epsilon)

        if iqx < 0:
            failures += 1

    return PropertyTestResult(
        name="IQx Non-Negativity",
        passed=failures == 0,
        n_trials=n_trials,
        failures=failures,
        description="IQx >= 0 for all inputs (drift >= 0, baseline >= 0, epsilon > 0)",
        error_message=f"Found {failures} negative IQx values" if failures > 0 else None,
    )


def test_iqx_zero_influence(n_trials: int = 1000, seed: int = 42) -> PropertyTestResult:
    """Property 2: If drift = 0, then IQx = 0.

    When there is no state change (drift = 0), influence should be zero
    regardless of baseline.
    """
    rng = np.random.default_rng(seed)
    failures = 0
    epsilon = 1e-6
    tolerance = 1e-10

    for _ in range(n_trials):
        baseline = abs(rng.normal(5, 2))
        iqx = compute_IQx(0.0, baseline, epsilon)

        if abs(iqx) > tolerance:
            failures += 1

    return PropertyTestResult(
        name="IQx Zero Influence",
        passed=failures == 0,
        n_trials=n_trials,
        failures=failures,
        description="IQx = 0 when drift = 0 (no state change implies no influence)",
        error_message=f"Found {failures} non-zero IQx with zero drift" if failures > 0 else None,
    )


def test_iqx_monotonicity_drift(n_trials: int = 1000, seed: int = 42) -> PropertyTestResult:
    """Property 3: IQx is monotonically increasing in drift.

    For fixed baseline B:
        drift_1 < drift_2 => IQx(drift_1) < IQx(drift_2)
    """
    rng = np.random.default_rng(seed)
    failures = 0
    epsilon = 1e-6

    for _ in range(n_trials):
        baseline = abs(rng.normal(5, 2)) + 0.1  # Ensure positive
        drift1 = abs(rng.uniform(0, 10))
        drift2 = drift1 + abs(rng.uniform(0.1, 5))  # drift2 > drift1

        iqx1 = compute_IQx(drift1, baseline, epsilon)
        iqx2 = compute_IQx(drift2, baseline, epsilon)

        if iqx2 <= iqx1:
            failures += 1

    return PropertyTestResult(
        name="IQx Monotonicity (Drift)",
        passed=failures == 0,
        n_trials=n_trials,
        failures=failures,
        description="Larger drift => larger IQx (for fixed baseline)",
        error_message=f"Found {failures} monotonicity violations" if failures > 0 else None,
    )


def test_iqx_monotonicity_baseline(n_trials: int = 1000, seed: int = 42) -> PropertyTestResult:
    """Property 4: IQx is monotonically decreasing in baseline.

    For fixed drift D:
        baseline_1 < baseline_2 => IQx(baseline_1) > IQx(baseline_2)

    Higher baseline means the agent is more "responsive" normally,
    so the same drift indicates less relative influence.
    """
    rng = np.random.default_rng(seed)
    failures = 0
    epsilon = 1e-6

    for _ in range(n_trials):
        drift = abs(rng.normal(5, 2)) + 0.1
        baseline1 = abs(rng.uniform(0.1, 5))
        baseline2 = baseline1 + abs(rng.uniform(0.1, 5))  # baseline2 > baseline1

        iqx1 = compute_IQx(drift, baseline1, epsilon)
        iqx2 = compute_IQx(drift, baseline2, epsilon)

        if iqx2 >= iqx1:
            failures += 1

    return PropertyTestResult(
        name="IQx Monotonicity (Baseline)",
        passed=failures == 0,
        n_trials=n_trials,
        failures=failures,
        description="Larger baseline => smaller IQx (for fixed drift)",
        error_message=f"Found {failures} monotonicity violations" if failures > 0 else None,
    )


def test_pr_empty_network() -> PropertyTestResult:
    """Property 5: PR = 0 for empty network.

    An empty adjacency matrix has no eigenvalues, so PR should be 0.
    """
    empty_matrix = np.array([], dtype=np.float64).reshape(0, 0)
    pr = compute_propagation_risk(empty_matrix)

    passed = abs(pr) < 1e-10

    return PropertyTestResult(
        name="PR Empty Network",
        passed=passed,
        n_trials=1,
        failures=0 if passed else 1,
        description="PR = 0 for empty adjacency matrix",
        error_message=f"PR = {pr}, expected 0" if not passed else None,
    )


def test_pr_zero_matrix() -> PropertyTestResult:
    """Property 6: PR = 0 for zero adjacency matrix.

    A network with no influence edges should have PR = 0.
    """
    sizes = [3, 5, 10]
    failures = 0

    for n in sizes:
        zero_matrix = np.zeros((n, n), dtype=np.float64)
        pr = compute_propagation_risk(zero_matrix)
        if abs(pr) > 1e-10:
            failures += 1

    return PropertyTestResult(
        name="PR Zero Matrix",
        passed=failures == 0,
        n_trials=len(sizes),
        failures=failures,
        description="PR = 0 for zero adjacency matrix (no influence)",
        error_message=f"Failed for {failures} matrix sizes" if failures > 0 else None,
    )


def test_pr_identity_matrix() -> PropertyTestResult:
    """Property 7: PR = 1 for identity matrix.

    Identity matrix has all eigenvalues = 1, so spectral radius = 1.
    This represents a network where each agent perfectly preserves influence.
    """
    sizes = [3, 5, 10]
    failures = 0
    tolerance = 1e-10

    for n in sizes:
        identity = np.eye(n, dtype=np.float64)
        pr = compute_propagation_risk(identity)
        if abs(pr - 1.0) > tolerance:
            failures += 1

    return PropertyTestResult(
        name="PR Identity Matrix",
        passed=failures == 0,
        n_trials=len(sizes),
        failures=failures,
        description="PR = 1 for identity matrix (eigenvalues all equal 1)",
        error_message=f"Failed for {failures} matrix sizes" if failures > 0 else None,
    )


def test_pr_scaling(n_trials: int = 100, seed: int = 42) -> PropertyTestResult:
    """Property 8: PR scales with matrix entries.

    For scalar c > 1: PR(cW) = c * PR(W)
    This is because eigenvalues scale linearly with matrix scaling.
    """
    rng = np.random.default_rng(seed)
    failures = 0
    tolerance = 1e-8

    for _ in range(n_trials):
        n = rng.integers(2, 6)
        W = rng.random((n, n))
        c = rng.uniform(1.1, 3.0)

        pr_W = compute_propagation_risk(W)
        pr_cW = compute_propagation_risk(c * W)

        expected = c * pr_W
        if abs(pr_cW - expected) > tolerance * max(1.0, abs(expected)):
            failures += 1

    return PropertyTestResult(
        name="PR Scaling",
        passed=failures == 0,
        n_trials=n_trials,
        failures=failures,
        description="PR(cW) = c * PR(W) for scalar c",
        error_message=f"Found {failures} scaling violations" if failures > 0 else None,
    )


def test_pr_non_negativity(n_trials: int = 100, seed: int = 42) -> PropertyTestResult:
    """Property 9: PR >= 0 for all matrices.

    Spectral radius is defined as max absolute eigenvalue, so always non-negative.
    """
    rng = np.random.default_rng(seed)
    failures = 0

    for _ in range(n_trials):
        n = rng.integers(2, 10)
        # Generate random matrix (can have negative entries)
        W = rng.normal(0, 1, (n, n))
        pr = compute_propagation_risk(W)

        if pr < 0:
            failures += 1

    return PropertyTestResult(
        name="PR Non-Negativity",
        passed=failures == 0,
        n_trials=n_trials,
        failures=failures,
        description="PR >= 0 for all adjacency matrices",
        error_message=f"Found {failures} negative PR values" if failures > 0 else None,
    )


def test_zscore_standardization(n_trials: int = 100, seed: int = 42) -> PropertyTestResult:
    """Property 10: Z-score transforms to approximately standard normal.

    For a sample from any distribution, Z-scores should have mean ≈ 0 and std ≈ 1.
    """
    rng = np.random.default_rng(seed)
    failures = 0
    mean_tolerance = 0.1
    std_tolerance = 0.1

    for _ in range(n_trials):
        # Generate IQx-like values
        n = rng.integers(50, 200)
        iqx_values = abs(rng.normal(1.0, 0.5, n))

        mean = np.mean(iqx_values)
        std = np.std(iqx_values, ddof=1)

        if std < 1e-10:
            continue

        z_scores = [compute_z_score(x, mean, std) for x in iqx_values]

        z_mean = np.mean(z_scores)
        z_std = np.std(z_scores, ddof=1)

        if abs(z_mean) > mean_tolerance or abs(z_std - 1.0) > std_tolerance:
            failures += 1

    return PropertyTestResult(
        name="Z-Score Standardization",
        passed=failures == 0,
        n_trials=n_trials,
        failures=failures,
        description="Z-scores have mean ≈ 0 and std ≈ 1",
        error_message=f"Found {failures} standardization violations" if failures > 0 else None,
    )


def test_zscore_symmetry() -> PropertyTestResult:
    """Property 11: Z-score is symmetric around the mean.

    z(mean + d) = -z(mean - d) for any displacement d.
    """
    mean = 1.0
    std = 0.5
    epsilon = 1e-6

    displacements = [0.1, 0.2, 0.5, 1.0, 2.0]
    failures = 0
    tolerance = 1e-10

    for d in displacements:
        z_plus = compute_z_score(mean + d, mean, std, epsilon)
        z_minus = compute_z_score(mean - d, mean, std, epsilon)

        if abs(z_plus + z_minus) > tolerance:
            failures += 1

    return PropertyTestResult(
        name="Z-Score Symmetry",
        passed=failures == 0,
        n_trials=len(displacements),
        failures=failures,
        description="z(μ + d) = -z(μ - d) for displacement d",
        error_message=f"Found {failures} symmetry violations" if failures > 0 else None,
    )


def test_drift_l2_triangle_inequality(n_trials: int = 100, seed: int = 42) -> PropertyTestResult:
    """Property 12: L2 drift satisfies triangle inequality.

    ||A - C||_2 <= ||A - B||_2 + ||B - C||_2
    """
    rng = np.random.default_rng(seed)
    failures = 0
    tolerance = 1e-10

    for _ in range(n_trials):
        dim = rng.integers(32, 128)
        A = rng.normal(0, 1, dim).astype(np.float32)
        B = rng.normal(0, 1, dim).astype(np.float32)
        C = rng.normal(0, 1, dim).astype(np.float32)

        d_AC = compute_drift_l2(A, C)
        d_AB = compute_drift_l2(A, B)
        d_BC = compute_drift_l2(B, C)

        if d_AC > d_AB + d_BC + tolerance:
            failures += 1

    return PropertyTestResult(
        name="Drift L2 Triangle Inequality",
        passed=failures == 0,
        n_trials=n_trials,
        failures=failures,
        description="||A - C|| <= ||A - B|| + ||B - C||",
        error_message=f"Found {failures} triangle inequality violations" if failures > 0 else None,
    )


def test_drift_l2_non_negativity(n_trials: int = 100, seed: int = 42) -> PropertyTestResult:
    """Property 13: L2 drift is non-negative.

    ||A - B||_2 >= 0 for all A, B.
    """
    rng = np.random.default_rng(seed)
    failures = 0

    for _ in range(n_trials):
        dim = rng.integers(32, 128)
        A = rng.normal(0, 1, dim).astype(np.float32)
        B = rng.normal(0, 1, dim).astype(np.float32)

        drift = compute_drift_l2(A, B)

        if drift < 0:
            failures += 1

    return PropertyTestResult(
        name="Drift L2 Non-Negativity",
        passed=failures == 0,
        n_trials=n_trials,
        failures=failures,
        description="||A - B|| >= 0 for all embeddings",
        error_message=f"Found {failures} negative drift values" if failures > 0 else None,
    )


def test_drift_l2_identity(n_trials: int = 100, seed: int = 42) -> PropertyTestResult:
    """Property 14: L2 drift is zero for identical embeddings.

    ||A - A||_2 = 0
    """
    rng = np.random.default_rng(seed)
    failures = 0
    tolerance = 1e-10

    for _ in range(n_trials):
        dim = rng.integers(32, 128)
        A = rng.normal(0, 1, dim).astype(np.float32)

        drift = compute_drift_l2(A, A)

        if drift > tolerance:
            failures += 1

    return PropertyTestResult(
        name="Drift L2 Identity",
        passed=failures == 0,
        n_trials=n_trials,
        failures=failures,
        description="||A - A|| = 0 for identical embeddings",
        error_message=f"Found {failures} non-zero self-drift values" if failures > 0 else None,
    )


def test_drift_l2_symmetry(n_trials: int = 100, seed: int = 42) -> PropertyTestResult:
    """Property 15: L2 drift is symmetric.

    ||A - B||_2 = ||B - A||_2
    """
    rng = np.random.default_rng(seed)
    failures = 0
    tolerance = 1e-10

    for _ in range(n_trials):
        dim = rng.integers(32, 128)
        A = rng.normal(0, 1, dim).astype(np.float32)
        B = rng.normal(0, 1, dim).astype(np.float32)

        d_AB = compute_drift_l2(A, B)
        d_BA = compute_drift_l2(B, A)

        if abs(d_AB - d_BA) > tolerance:
            failures += 1

    return PropertyTestResult(
        name="Drift L2 Symmetry",
        passed=failures == 0,
        n_trials=n_trials,
        failures=failures,
        description="||A - B|| = ||B - A|| (symmetric)",
        error_message=f"Found {failures} symmetry violations" if failures > 0 else None,
    )


def run_all_tests() -> list[PropertyTestResult]:
    """Run all property tests and return results."""
    tests: list[Callable[[], PropertyTestResult]] = [
        # IQx properties
        test_iqx_non_negativity,
        test_iqx_zero_influence,
        test_iqx_monotonicity_drift,
        test_iqx_monotonicity_baseline,
        # PR properties
        test_pr_empty_network,
        test_pr_zero_matrix,
        test_pr_identity_matrix,
        test_pr_scaling,
        test_pr_non_negativity,
        # Z-score properties
        test_zscore_standardization,
        test_zscore_symmetry,
        # Drift properties
        test_drift_l2_triangle_inequality,
        test_drift_l2_non_negativity,
        test_drift_l2_identity,
        test_drift_l2_symmetry,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            results.append(
                PropertyTestResult(
                    name=test.__name__,
                    passed=False,
                    n_trials=0,
                    failures=1,
                    description="Test execution failed",
                    error_message=str(e),
                )
            )

    return results


def save_results(results: list[PropertyTestResult], output_path: Path) -> None:
    """Save results to CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["name", "passed", "n_trials", "failures", "description", "error_message"]

    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            writer.writerow({
                "name": result.name,
                "passed": result.passed,
                "n_trials": result.n_trials,
                "failures": result.failures,
                "description": result.description,
                "error_message": result.error_message or "",
            })

    print(f"Results saved to {output_path}")


def print_results(results: list[PropertyTestResult]) -> None:
    """Print results in a formatted table."""
    print("\n" + "=" * 80)
    print("METRIC PROPERTY VERIFICATION RESULTS")
    print("=" * 80)

    passed_count = sum(1 for r in results if r.passed)
    total_count = len(results)

    # Group by metric type
    iqx_tests = [r for r in results if r.name.startswith("IQx")]
    pr_tests = [r for r in results if r.name.startswith("PR")]
    zscore_tests = [r for r in results if r.name.startswith("Z-Score")]
    drift_tests = [r for r in results if r.name.startswith("Drift")]

    for group_name, tests in [
        ("IQx Properties", iqx_tests),
        ("Propagation Risk Properties", pr_tests),
        ("Z-Score Properties", zscore_tests),
        ("Drift Properties", drift_tests),
    ]:
        if not tests:
            continue

        print(f"\n{group_name}:")
        print("-" * 60)

        for result in tests:
            status = "PASS" if result.passed else "FAIL"
            status_color = "\033[92m" if result.passed else "\033[91m"
            reset = "\033[0m"

            print(f"  [{status_color}{status}{reset}] {result.name}")
            print(f"        {result.description}")
            if not result.passed and result.error_message:
                print(f"        Error: {result.error_message}")

    print("\n" + "=" * 80)
    print(f"SUMMARY: {passed_count}/{total_count} properties verified")

    if passed_count == total_count:
        print("\033[92mAll metric properties PASS\033[0m")
    else:
        print(f"\033[91m{total_count - passed_count} properties FAILED\033[0m")

    print("=" * 80)


def main() -> None:
    """Run metric property verification."""
    print("Metric Property Verification")
    print("-" * 40)

    results = run_all_tests()

    # Save results
    output_path = Path("experiments/results/metric_properties.csv")
    save_results(results, output_path)

    # Print results
    print_results(results)

    # Exit with error code if any tests failed
    if not all(r.passed for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
