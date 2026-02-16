"""Core mathematical metrics for IEEE research framework.

This module implements the mathematical formulas for influence quantification
as described in the TraceIQ IEEE research paper.

Metrics:
    - Drift (L2): ||s_j(t+) - s_j(t-)||_2
    - IQx: I / (B_j + epsilon)
    - Accumulated Influence: Sum of IQx over window
    - Propagation Risk: spectral_radius(adjacency_matrix)
    - Attack Surface: Sum of capability weights
    - Risk-Weighted Influence: IQx * attack_surface
    - Z-score: (IQx - mean) / (std + epsilon)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from numpy.typing import NDArray


def compute_drift_l2(
    emb_before: NDArray[np.float32],
    emb_after: NDArray[np.float32],
) -> float:
    """Compute L2 norm drift between two embeddings.

    Formula: D_j(t) = ||s_j(t+) - s_j(t-)||_2

    Args:
        emb_before: Embedding state before interaction
        emb_after: Embedding state after interaction

    Returns:
        L2 norm distance (drift magnitude)
    """
    diff = emb_after - emb_before
    return float(np.linalg.norm(diff))


def compute_IQx(
    drift: float,
    baseline_median: float,
    epsilon: float = 1e-6,
) -> float:
    """Compute Influence Quotient (IQx).

    Formula: IQx = I / (B_j + epsilon)

    The IQx normalizes influence by the receiver's baseline responsiveness,
    making influence scores comparable across different agents.

    Args:
        drift: The drift value (influence proxy)
        baseline_median: Median drift for this receiver
        epsilon: Small constant for numerical stability

    Returns:
        Normalized influence quotient
    """
    return drift / (baseline_median + epsilon)


def compute_accumulated_influence(iqx_values: list[float]) -> float:
    """Compute accumulated influence over a window.

    Formula: AI_j(W) = Sum(IQx) over window W

    Args:
        iqx_values: List of IQx values in the window

    Returns:
        Sum of IQx values
    """
    if not iqx_values:
        return 0.0
    return float(sum(iqx_values))


def compute_propagation_risk(adjacency_matrix: NDArray[np.float64]) -> float:
    """Compute propagation risk as spectral radius of adjacency matrix.

    Formula: PR(W) = spectral_radius(W) = max(|eigenvalues|)

    The spectral radius indicates network instability - values > 1.0
    suggest influence can amplify through the network.

    Args:
        adjacency_matrix: NxN weighted adjacency matrix

    Returns:
        Spectral radius (largest absolute eigenvalue)
    """
    if adjacency_matrix.size == 0:
        return 0.0

    # Handle 1x1 matrix case
    if adjacency_matrix.shape == (1, 1):
        return float(np.abs(adjacency_matrix[0, 0]))

    try:
        eigenvalues = np.linalg.eigvals(adjacency_matrix)
        return float(np.max(np.abs(eigenvalues)))
    except np.linalg.LinAlgError:
        return 0.0


def compute_attack_surface(
    capabilities: list[str],
    weights: dict[str, float],
) -> float:
    """Compute attack surface for an agent based on capabilities.

    Formula: AS_i = Sum(p(c)) for all capabilities c

    Args:
        capabilities: List of capability names
        weights: Dict mapping capability names to weights

    Returns:
        Sum of capability weights (attack surface score)
    """
    total = 0.0
    for cap in capabilities:
        total += weights.get(cap, 0.0)
    return total


def compute_RWI(iqx: float, attack_surface: float) -> float:
    """Compute Risk-Weighted Influence.

    Formula: RWI = IQx * AS_i

    Combines influence magnitude with sender's attack surface
    to prioritize monitoring of high-capability influencers.

    Args:
        iqx: Influence quotient
        attack_surface: Sender's attack surface score

    Returns:
        Risk-weighted influence score
    """
    return iqx * attack_surface


def compute_z_score(
    iqx: float,
    mean: float,
    std: float,
    epsilon: float = 1e-6,
) -> float:
    """Compute Z-score for anomaly detection.

    Formula: Z = (IQx - mean) / (std + epsilon)

    Args:
        iqx: Current IQx value
        mean: Historical mean IQx
        std: Historical standard deviation
        epsilon: Small constant for numerical stability

    Returns:
        Z-score (number of standard deviations from mean)
    """
    return (iqx - mean) / (std + epsilon)


def rolling_median(values: list[float]) -> float:
    """Compute rolling median of values.

    Args:
        values: List of numeric values

    Returns:
        Median value, or 0.0 if empty
    """
    if not values:
        return 0.0
    return float(np.median(values))


def rolling_mean(values: list[float]) -> float:
    """Compute rolling mean of values.

    Args:
        values: List of numeric values

    Returns:
        Mean value, or 0.0 if empty
    """
    if not values:
        return 0.0
    return float(np.mean(values))


def rolling_std(values: list[float]) -> float:
    """Compute rolling standard deviation of values.

    Args:
        values: List of numeric values

    Returns:
        Standard deviation, or 0.0 if fewer than 2 values
    """
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=1))


def build_adjacency_matrix(
    edge_weights: dict[tuple[str, str], float],
    agents: list[str],
) -> NDArray[np.float64]:
    """Build adjacency matrix from edge weights.

    Args:
        edge_weights: Dict mapping (sender, receiver) to weight
        agents: Ordered list of agent IDs (determines matrix indices)

    Returns:
        NxN numpy array where M[i,j] = weight from agent i to agent j
    """
    n = len(agents)
    if n == 0:
        return np.array([], dtype=np.float64)

    agent_idx = {agent: i for i, agent in enumerate(agents)}
    matrix = np.zeros((n, n), dtype=np.float64)

    for (sender, receiver), weight in edge_weights.items():
        if sender in agent_idx and receiver in agent_idx:
            matrix[agent_idx[sender], agent_idx[receiver]] = weight

    return matrix
