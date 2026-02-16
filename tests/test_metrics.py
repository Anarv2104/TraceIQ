"""Tests for IEEE metrics module."""

import numpy as np
import pytest

from traceiq.metrics import (
    build_adjacency_matrix,
    compute_accumulated_influence,
    compute_attack_surface,
    compute_drift_l2,
    compute_IQx,
    compute_propagation_risk,
    compute_RWI,
    compute_z_score,
    rolling_mean,
    rolling_median,
    rolling_std,
)


class TestComputeDriftL2:
    """Tests for compute_drift_l2."""

    def test_identical_embeddings_zero_drift(self):
        """Identical embeddings should have zero drift."""
        emb = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        drift = compute_drift_l2(emb, emb)
        assert drift == 0.0

    def test_different_embeddings_positive_drift(self):
        """Different embeddings should have positive drift."""
        emb1 = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        emb2 = np.array([3.0, 4.0, 0.0], dtype=np.float32)
        drift = compute_drift_l2(emb1, emb2)
        assert drift == pytest.approx(5.0)  # sqrt(9 + 16) = 5

    def test_unit_vectors(self):
        """Test with unit vectors."""
        emb1 = np.array([1.0, 0.0], dtype=np.float32)
        emb2 = np.array([0.0, 1.0], dtype=np.float32)
        drift = compute_drift_l2(emb1, emb2)
        assert drift == pytest.approx(np.sqrt(2))


class TestComputeIQx:
    """Tests for compute_IQx."""

    def test_basic_computation(self):
        """Test basic IQx computation."""
        drift = 1.0
        baseline_median = 0.5
        iqx = compute_IQx(drift, baseline_median)
        assert iqx == pytest.approx(2.0, rel=1e-5)

    def test_zero_baseline_uses_epsilon(self):
        """Zero baseline should use epsilon to avoid division by zero."""
        drift = 1.0
        iqx = compute_IQx(drift, 0.0, epsilon=1e-6)
        assert iqx == pytest.approx(1e6, rel=1e-2)

    def test_scale_invariance(self):
        """IQx should be proportional to drift."""
        baseline = 0.5
        iqx1 = compute_IQx(1.0, baseline)
        iqx2 = compute_IQx(2.0, baseline)
        assert iqx2 == pytest.approx(2 * iqx1, rel=1e-5)


class TestComputeAccumulatedInfluence:
    """Tests for compute_accumulated_influence."""

    def test_empty_list(self):
        """Empty list should return 0."""
        assert compute_accumulated_influence([]) == 0.0

    def test_single_value(self):
        """Single value should return that value."""
        assert compute_accumulated_influence([5.0]) == 5.0

    def test_sum_of_values(self):
        """Should return sum of all values."""
        values = [1.0, 2.0, 3.0, 4.0]
        assert compute_accumulated_influence(values) == 10.0


class TestComputePropagationRisk:
    """Tests for compute_propagation_risk."""

    def test_empty_matrix(self):
        """Empty matrix should return 0."""
        matrix = np.array([], dtype=np.float64)
        assert compute_propagation_risk(matrix) == 0.0

    def test_identity_matrix(self):
        """Identity matrix should have spectral radius 1.0."""
        matrix = np.eye(3, dtype=np.float64)
        pr = compute_propagation_risk(matrix)
        assert pr == pytest.approx(1.0)

    def test_zero_matrix(self):
        """Zero matrix should have spectral radius 0."""
        matrix = np.zeros((3, 3), dtype=np.float64)
        pr = compute_propagation_risk(matrix)
        assert pr == pytest.approx(0.0)

    def test_increases_with_edge_weights(self):
        """Spectral radius should increase with edge weights."""
        matrix1 = np.array([[0, 0.5], [0.5, 0]], dtype=np.float64)
        matrix2 = np.array([[0, 1.0], [1.0, 0]], dtype=np.float64)

        pr1 = compute_propagation_risk(matrix1)
        pr2 = compute_propagation_risk(matrix2)

        assert pr2 > pr1

    def test_single_element(self):
        """1x1 matrix should return that element's absolute value."""
        matrix = np.array([[5.0]], dtype=np.float64)
        assert compute_propagation_risk(matrix) == 5.0


class TestComputeAttackSurface:
    """Tests for compute_attack_surface."""

    def test_empty_capabilities(self):
        """Empty capabilities should return 0."""
        assert compute_attack_surface([], {"execute_code": 1.0}) == 0.0

    def test_single_capability(self):
        """Single capability should return its weight."""
        caps = ["execute_code"]
        weights = {"execute_code": 1.0}
        assert compute_attack_surface(caps, weights) == 1.0

    def test_multiple_capabilities(self):
        """Multiple capabilities should sum their weights."""
        caps = ["execute_code", "admin"]
        weights = {"execute_code": 1.0, "admin": 1.5}
        assert compute_attack_surface(caps, weights) == 2.5

    def test_unknown_capability_ignored(self):
        """Unknown capabilities should be ignored (weight 0)."""
        caps = ["execute_code", "unknown"]
        weights = {"execute_code": 1.0}
        assert compute_attack_surface(caps, weights) == 1.0


class TestComputeRWI:
    """Tests for compute_RWI."""

    def test_basic_computation(self):
        """Test basic RWI computation."""
        iqx = 2.0
        attack_surface = 1.5
        rwi = compute_RWI(iqx, attack_surface)
        assert rwi == 3.0

    def test_zero_attack_surface(self):
        """Zero attack surface should return 0."""
        assert compute_RWI(5.0, 0.0) == 0.0

    def test_zero_iqx(self):
        """Zero IQx should return 0."""
        assert compute_RWI(0.0, 5.0) == 0.0


class TestComputeZScore:
    """Tests for compute_z_score."""

    def test_at_mean(self):
        """Value at mean should have Z-score 0."""
        z = compute_z_score(5.0, mean=5.0, std=1.0)
        assert z == pytest.approx(0.0)

    def test_one_std_above(self):
        """One std above mean should have Z-score 1."""
        z = compute_z_score(6.0, mean=5.0, std=1.0)
        assert z == pytest.approx(1.0)

    def test_two_std_below(self):
        """Two std below mean should have Z-score -2."""
        z = compute_z_score(3.0, mean=5.0, std=1.0)
        assert z == pytest.approx(-2.0)

    def test_zero_std_uses_epsilon(self):
        """Zero std should use epsilon."""
        z = compute_z_score(6.0, mean=5.0, std=0.0, epsilon=1.0)
        assert z == pytest.approx(1.0)

    def test_anomaly_detection_works(self):
        """Anomalies should have high absolute Z-score."""
        z = compute_z_score(100.0, mean=5.0, std=2.0)
        assert abs(z) > 2.0  # Clear anomaly


class TestRollingFunctions:
    """Tests for rolling statistical functions."""

    def test_rolling_median_empty(self):
        """Empty list should return 0."""
        assert rolling_median([]) == 0.0

    def test_rolling_median_single(self):
        """Single value should return that value."""
        assert rolling_median([5.0]) == 5.0

    def test_rolling_median_odd(self):
        """Odd-length list should return middle value."""
        assert rolling_median([1.0, 2.0, 3.0]) == 2.0

    def test_rolling_median_even(self):
        """Even-length list should return average of middle values."""
        assert rolling_median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_rolling_mean_empty(self):
        """Empty list should return 0."""
        assert rolling_mean([]) == 0.0

    def test_rolling_mean_basic(self):
        """Test basic mean computation."""
        assert rolling_mean([2.0, 4.0, 6.0]) == 4.0

    def test_rolling_std_empty(self):
        """Empty list should return 0."""
        assert rolling_std([]) == 0.0

    def test_rolling_std_single(self):
        """Single value should return 0."""
        assert rolling_std([5.0]) == 0.0

    def test_rolling_std_basic(self):
        """Test basic std computation."""
        std = rolling_std([2.0, 4.0, 6.0])
        assert std == pytest.approx(2.0)


class TestBuildAdjacencyMatrix:
    """Tests for build_adjacency_matrix."""

    def test_empty_input(self):
        """Empty agents should return empty matrix."""
        matrix = build_adjacency_matrix({}, [])
        assert matrix.size == 0

    def test_single_agent_no_edges(self):
        """Single agent with no edges should be zero matrix."""
        matrix = build_adjacency_matrix({}, ["agent_0"])
        assert matrix.shape == (1, 1)
        assert matrix[0, 0] == 0.0

    def test_two_agents_one_edge(self):
        """Two agents with one edge."""
        edges = {("a", "b"): 0.5}
        agents = ["a", "b"]
        matrix = build_adjacency_matrix(edges, agents)

        assert matrix.shape == (2, 2)
        assert matrix[0, 1] == 0.5  # a -> b
        assert matrix[1, 0] == 0.0  # b -> a (no edge)

    def test_symmetric_edges(self):
        """Symmetric edges should produce symmetric matrix."""
        edges = {("a", "b"): 0.5, ("b", "a"): 0.5}
        agents = ["a", "b"]
        matrix = build_adjacency_matrix(edges, agents)

        assert matrix[0, 1] == matrix[1, 0] == 0.5

    def test_unknown_agent_ignored(self):
        """Edges with unknown agents should be ignored."""
        edges = {("a", "b"): 0.5, ("c", "d"): 0.5}
        agents = ["a", "b"]
        matrix = build_adjacency_matrix(edges, agents)

        assert matrix.shape == (2, 2)
        assert matrix[0, 1] == 0.5
