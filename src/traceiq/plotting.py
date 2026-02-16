"""Matplotlib-based plotting functions."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from traceiq.capabilities import CapabilityRegistry
    from traceiq.graph import InfluenceGraph
    from traceiq.models import InteractionEvent, PropagationRiskResult, ScoreResult


def _check_matplotlib() -> None:
    """Check if matplotlib is available."""
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install it with: pip install matplotlib "
            "or pip install traceiq[plot]"
        ) from e


def plot_drift_over_time(
    events: list[InteractionEvent],
    scores: list[ScoreResult],
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Plot drift delta over time per agent.

    Args:
        events: List of interaction events
        scores: List of score results
        output_path: Optional path to save figure
        figsize: Figure size tuple
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    score_map = {s.event_id: s for s in scores}

    # Group by receiver
    receiver_data: dict[str, list[tuple]] = {}
    for event in sorted(events, key=lambda e: e.timestamp):
        score = score_map.get(event.event_id)
        if score and not score.cold_start:
            if event.receiver_id not in receiver_data:
                receiver_data[event.receiver_id] = []
            receiver_data[event.receiver_id].append(
                (event.timestamp, score.drift_delta)
            )

    fig, ax = plt.subplots(figsize=figsize)

    for receiver_id, data in receiver_data.items():
        timestamps = [d[0] for d in data]
        drifts = [d[1] for d in data]
        ax.plot(timestamps, drifts, marker="o", label=receiver_id, alpha=0.7)

    ax.set_xlabel("Time")
    ax.set_ylabel("Drift Delta")
    ax.set_title("Agent Drift Over Time")
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_influence_heatmap(
    influence_graph: InfluenceGraph,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Plot influence matrix as a heatmap.

    Args:
        influence_graph: InfluenceGraph instance
        output_path: Optional path to save figure
        figsize: Figure size tuple
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = influence_graph.influence_matrix()
    if not matrix:
        print("No influence data to plot")
        return

    # Get all unique agents
    agents = sorted(
        set(matrix.keys()) | {r for senders in matrix.values() for r in senders}
    )

    # Build numpy matrix
    n = len(agents)
    agent_idx = {a: i for i, a in enumerate(agents)}
    data = np.zeros((n, n))

    for sender, receivers in matrix.items():
        for receiver, score in receivers.items():
            data[agent_idx[sender], agent_idx[receiver]] = score

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.set_yticklabels(agents)
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    ax.set_title("Influence Score Matrix")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Influence Score")

    # Add value annotations for small matrices
    if n <= 10:
        for i in range(n):
            for j in range(n):
                if data[i, j] > 0:
                    text_color = "white" if data[i, j] > 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{data[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_top_influencers(
    influence_graph: InfluenceGraph,
    n: int = 10,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot horizontal bar chart of top influencers.

    Args:
        influence_graph: InfluenceGraph instance
        n: Number of top influencers to show
        output_path: Optional path to save figure
        figsize: Figure size tuple
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    top = influence_graph.top_influencers(n)
    if not top:
        print("No influencer data to plot")
        return

    agents = [t[0] for t in top]
    scores = [t[1] for t in top]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.viridis([s / max(scores) if max(scores) > 0 else 0 for s in scores])
    bars = ax.barh(agents, scores, color=colors)

    ax.set_xlabel("Total Influence Score")
    ax.set_ylabel("Agent")
    ax.set_title(f"Top {len(agents)} Influencers")
    ax.invert_yaxis()  # Highest at top

    # Add value labels
    for bar, score in zip(bars, scores, strict=True):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_top_susceptible(
    influence_graph: InfluenceGraph,
    n: int = 10,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot horizontal bar chart of most susceptible agents.

    Args:
        influence_graph: InfluenceGraph instance
        n: Number of top susceptible agents to show
        output_path: Optional path to save figure
        figsize: Figure size tuple
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    top = influence_graph.top_susceptible(n)
    if not top:
        print("No susceptibility data to plot")
        return

    agents = [t[0] for t in top]
    scores = [t[1] for t in top]

    fig, ax = plt.subplots(figsize=figsize)

    colors = plt.cm.plasma([s / max(scores) if max(scores) > 0 else 0 for s in scores])
    bars = ax.barh(agents, scores, color=colors)

    ax.set_xlabel("Total Incoming Drift")
    ax.set_ylabel("Agent")
    ax.set_title(f"Top {len(agents)} Susceptible Agents")
    ax.invert_yaxis()

    for bar, score in zip(bars, scores, strict=True):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{score:.2f}",
            va="center",
            fontsize=9,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_influence_network(
    influence_graph: InfluenceGraph,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 10),
    min_edge_weight: float = 0.1,
) -> None:
    """
    Plot the influence network graph.

    Args:
        influence_graph: InfluenceGraph instance
        output_path: Optional path to save figure
        figsize: Figure size tuple
        min_edge_weight: Minimum edge weight to display
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    import networkx as nx

    G = influence_graph.graph
    if G.number_of_nodes() == 0:
        print("No network data to plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Filter edges by weight
    edges_to_draw = [
        (u, v)
        for u, v, d in G.edges(data=True)
        if d.get("weight", 0) >= min_edge_weight
    ]

    if not edges_to_draw:
        print(f"No edges with weight >= {min_edge_weight}")
        return

    # Create subgraph with filtered edges
    subgraph = G.edge_subgraph(edges_to_draw)

    # Layout
    pos = nx.spring_layout(subgraph, seed=42, k=2)

    # Draw nodes
    node_sizes = [300 + 100 * subgraph.degree(n) for n in subgraph.nodes()]
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        ax=ax,
        node_size=node_sizes,
        node_color="lightblue",
        edgecolors="black",
    )

    # Draw edges with varying width based on weight
    edge_weights = [subgraph[u][v].get("weight", 0.1) for u, v in subgraph.edges()]
    max_weight = max(edge_weights) if edge_weights else 1
    edge_widths = [2 + 3 * (w / max_weight) for w in edge_weights]

    nx.draw_networkx_edges(
        subgraph,
        pos,
        ax=ax,
        width=edge_widths,
        alpha=0.6,
        edge_color=edge_weights,
        edge_cmap=plt.cm.Reds,
        arrows=True,
        arrowsize=15,
        connectionstyle="arc3,rad=0.1",
    )

    # Draw labels
    nx.draw_networkx_labels(subgraph, pos, ax=ax, font_size=9)

    ax.set_title("Agent Influence Network")
    ax.axis("off")

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


# IEEE Metrics Plotting Functions (v0.3.0)


def plot_iqx_heatmap(
    influence_graph: InfluenceGraph,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 8),
) -> None:
    """
    Plot IQx matrix as a heatmap.

    Args:
        influence_graph: InfluenceGraph instance
        output_path: Optional path to save figure
        figsize: Figure size tuple
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    matrix = influence_graph.iqx_matrix()
    if not matrix:
        print("No IQx data to plot")
        return

    # Get all unique agents
    agents = sorted(
        set(matrix.keys()) | {r for senders in matrix.values() for r in senders}
    )

    # Build numpy matrix
    n = len(agents)
    agent_idx = {a: i for i, a in enumerate(agents)}
    data = np.zeros((n, n))

    for sender, receivers in matrix.items():
        for receiver, iqx in receivers.items():
            data[agent_idx[sender], agent_idx[receiver]] = iqx

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(data, cmap="Blues", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.set_yticklabels(agents)
    ax.set_xlabel("Receiver")
    ax.set_ylabel("Sender")
    ax.set_title("IQx (Influence Quotient) Matrix")

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("IQx")

    # Add value annotations for small matrices
    if n <= 10:
        max_val = np.max(data) if np.max(data) > 0 else 1
        for i in range(n):
            for j in range(n):
                if data[i, j] > 0:
                    text_color = "white" if data[i, j] > max_val * 0.5 else "black"
                    ax.text(
                        j,
                        i,
                        f"{data[i, j]:.2f}",
                        ha="center",
                        va="center",
                        color=text_color,
                        fontsize=8,
                    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_propagation_risk_over_time(
    pr_results: list[PropagationRiskResult],
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Plot propagation risk (spectral radius) over time.

    Args:
        pr_results: List of PropagationRiskResult from compute_propagation_risk_over_time
        output_path: Optional path to save figure
        figsize: Figure size tuple
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    if not pr_results:
        print("No propagation risk data to plot")
        return

    timestamps = [pr.window_end for pr in pr_results]
    spectral_radii = [pr.spectral_radius for pr in pr_results]

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(timestamps, spectral_radii, marker="o", linewidth=2, color="red", alpha=0.7)

    # Add danger threshold line
    ax.axhline(y=1.0, color="orange", linestyle="--", label="Amplification threshold")

    ax.set_xlabel("Time")
    ax.set_ylabel("Spectral Radius")
    ax.set_title("Propagation Risk Over Time")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Highlight dangerous regions
    ax.fill_between(
        timestamps,
        spectral_radii,
        1.0,
        where=[sr > 1.0 for sr in spectral_radii],
        alpha=0.3,
        color="red",
        label="High risk",
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_top_risky_agents(
    influence_graph: InfluenceGraph,
    capabilities: CapabilityRegistry,
    n: int = 10,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Plot top risky agents combining IQx influence with attack surface.

    Args:
        influence_graph: InfluenceGraph instance
        capabilities: CapabilityRegistry instance
        n: Number of top agents to show
        output_path: Optional path to save figure
        figsize: Figure size tuple
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    top_iqx = influence_graph.top_iqx_influencers(n * 2)  # Get more for filtering
    if not top_iqx:
        print("No IQx data to plot")
        return

    # Compute RWI for each agent
    agents_data = []
    for agent_id, total_iqx in top_iqx:
        attack_surface = capabilities.compute_attack_surface(agent_id)
        rwi = total_iqx * attack_surface if attack_surface > 0 else total_iqx
        agents_data.append((agent_id, total_iqx, attack_surface, rwi))

    # Sort by RWI and take top n
    agents_data.sort(key=lambda x: -x[3])
    agents_data = agents_data[:n]

    if not agents_data:
        print("No risky agents to plot")
        return

    agents = [d[0] for d in agents_data]
    iqx_values = [d[1] for d in agents_data]
    attack_surfaces = [d[2] for d in agents_data]
    rwi_values = [d[3] for d in agents_data]

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(agents))
    width = 0.35

    ax.bar(x - width / 2, iqx_values, width, label="Total IQx", color="steelblue")
    bars2 = ax.bar(x + width / 2, rwi_values, width, label="RWI", color="crimson")

    ax.set_xlabel("Agent")
    ax.set_ylabel("Score")
    ax.set_title("Top Risky Agents (IQx vs Risk-Weighted Influence)")
    ax.set_xticks(x)
    ax.set_xticklabels(agents, rotation=45, ha="right")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    # Add attack surface annotation
    for bar, as_val in zip(bars2, attack_surfaces, strict=True):
        ax.annotate(
            f"AS={as_val:.1f}",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=7,
        )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_z_score_distribution(
    scores: list[ScoreResult],
    threshold: float = 2.0,
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (10, 6),
) -> None:
    """
    Plot distribution of Z-scores with anomaly threshold.

    Args:
        scores: List of ScoreResult
        threshold: Z-score threshold for anomalies
        output_path: Optional path to save figure
        figsize: Figure size tuple
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    z_scores = [s.Z_score for s in scores if s.Z_score is not None]
    if not z_scores:
        print("No Z-score data to plot")
        return

    fig, ax = plt.subplots(figsize=figsize)

    # Histogram
    n_bins = min(50, len(z_scores) // 5 + 1)
    counts, bins, patches = ax.hist(z_scores, bins=n_bins, edgecolor="black", alpha=0.7)

    # Color bars beyond threshold
    for patch, left, right in zip(patches, bins[:-1], bins[1:], strict=True):
        if abs(left) > threshold or abs(right) > threshold:
            patch.set_facecolor("red")
        else:
            patch.set_facecolor("steelblue")

    # Add threshold lines
    ax.axvline(
        x=threshold, color="red", linestyle="--", label=f"Threshold ({threshold})"
    )
    ax.axvline(x=-threshold, color="red", linestyle="--")

    # Statistics
    mean_z = np.mean(z_scores)
    std_z = np.std(z_scores)
    n_anomalies = sum(1 for z in z_scores if abs(z) > threshold)

    ax.set_xlabel("Z-Score")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Z-Score Distribution (anomalies: {n_anomalies}/{len(z_scores)})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = f"Mean: {mean_z:.2f}\nStd: {std_z:.2f}"
    ax.text(
        0.95,
        0.95,
        stats_text,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="right",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_drift_comparison(
    events: list[InteractionEvent],
    scores: list[ScoreResult],
    output_path: str | Path | None = None,
    figsize: tuple[int, int] = (12, 6),
) -> None:
    """
    Plot comparison of cosine drift vs L2 drift over time.

    Args:
        events: List of interaction events
        scores: List of score results
        output_path: Optional path to save figure
        figsize: Figure size tuple
    """
    _check_matplotlib()
    import matplotlib.pyplot as plt

    score_map = {s.event_id: s for s in scores}

    # Collect data points
    timestamps = []
    cosine_drifts = []
    l2_drifts = []

    for event in sorted(events, key=lambda e: e.timestamp):
        score = score_map.get(event.event_id)
        if score and not score.cold_start and score.drift_l2 is not None:
            timestamps.append(event.timestamp)
            cosine_drifts.append(score.drift_delta)
            l2_drifts.append(score.drift_l2)

    if not timestamps:
        print("No drift data to plot")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True)

    # Cosine drift
    ax1.plot(
        timestamps,
        cosine_drifts,
        marker=".",
        color="blue",
        alpha=0.6,
        label="Cosine Drift",
    )
    ax1.set_ylabel("Cosine Drift (1 - similarity)")
    ax1.set_title("Drift Comparison: Cosine vs L2")
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # L2 drift
    ax2.plot(
        timestamps, l2_drifts, marker=".", color="green", alpha=0.6, label="L2 Drift"
    )
    ax2.set_xlabel("Time")
    ax2.set_ylabel("L2 Drift (Euclidean)")
    ax2.legend(loc="upper right")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
