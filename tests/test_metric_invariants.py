"""Metric invariant tests - CI-friendly (<5s)."""

import numpy as np
import pytest

from traceiq.metrics import (
    IQX_CAP,
    compute_drift_l2,
    compute_IQx,
    compute_propagation_risk,
)


class TestIQxBoundedness:
    """IQx is bounded [0, IQX_CAP]."""

    @pytest.mark.parametrize("drift", [0.0, 0.1, 1.0, 10.0, 100.0])
    @pytest.mark.parametrize("baseline", [0.01, 0.1, 1.0, 10.0])
    def test_iqx_in_bounds(self, drift: float, baseline: float) -> None:
        iqx = compute_IQx(drift, baseline)
        assert 0.0 <= iqx <= IQX_CAP


class TestIQxMonotonicity:
    """IQx monotonic in drift (for fixed baseline)."""

    def test_iqx_increases_with_drift(self) -> None:
        baseline = 0.5
        prev_iqx = -1.0
        for drift in [0.0, 0.1, 0.5, 1.0, 2.0, 5.0]:
            iqx = compute_IQx(drift, baseline)
            assert iqx >= prev_iqx
            prev_iqx = iqx


class TestDriftL2Properties:
    """Drift L2 is a metric (non-negative, identity, symmetric)."""

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(50):
            a = rng.normal(0, 1, 64).astype(np.float32)
            b = rng.normal(0, 1, 64).astype(np.float32)
            assert compute_drift_l2(a, b) >= 0.0

    def test_identity(self) -> None:
        a = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        assert compute_drift_l2(a, a) == pytest.approx(0.0, abs=1e-10)

    def test_symmetry(self) -> None:
        a = np.array([1.0, 2.0], dtype=np.float32)
        b = np.array([3.0, 4.0], dtype=np.float32)
        assert compute_drift_l2(a, b) == pytest.approx(compute_drift_l2(b, a))


class TestPRBounds:
    """PR >= 0, PR(0) = 0, PR(I) = 1."""

    def test_zero_matrix(self) -> None:
        W = np.zeros((3, 3))
        assert compute_propagation_risk(W) == pytest.approx(0.0)

    def test_identity_matrix(self) -> None:
        W = np.eye(3)
        assert compute_propagation_risk(W) == pytest.approx(1.0)

    def test_non_negative(self) -> None:
        rng = np.random.default_rng(42)
        for _ in range(20):
            W = rng.random((5, 5))
            assert compute_propagation_risk(W) >= 0.0
