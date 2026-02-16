"""Click-based CLI for TraceIQ."""

from __future__ import annotations

import json

import click
from rich.console import Console
from rich.table import Table

from traceiq.models import TrackerConfig
from traceiq.tracker import InfluenceTracker

console = Console()


@click.group()
@click.version_option(prog_name="traceiq")
def cli() -> None:
    """TraceIQ: Measure AI-to-AI influence in multi-agent systems."""
    pass


@cli.command()
@click.option(
    "--db",
    type=click.Path(),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--config",
    type=click.Path(exists=True),
    help="Path to JSON config file",
)
def init(db: str, config: str | None) -> None:
    """Initialize a new TraceIQ database."""
    if config:
        with open(config) as f:
            config_data = json.load(f)
        tracker_config = TrackerConfig(**config_data)
    else:
        tracker_config = TrackerConfig(
            storage_backend="sqlite",
            storage_path=db,
        )

    # Initialize tracker to create database
    tracker = InfluenceTracker(config=tracker_config, use_mock_embedder=True)
    tracker.close()

    console.print(f"[green]Initialized TraceIQ database at:[/green] {db}")


@cli.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.option(
    "--db",
    type=click.Path(),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--mock-embedder",
    is_flag=True,
    help="Use mock embedder (no sentence-transformers required)",
)
def ingest(input_file: str, db: str, mock_embedder: bool) -> None:
    """Ingest interactions from a JSONL file.

    Each line should be a JSON object with:
    sender_id, receiver_id, sender_content, receiver_content
    """
    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    # Required fields for each interaction
    required_fields = ["sender_id", "receiver_id", "sender_content", "receiver_content"]

    # Parse and validate all lines first
    interactions = []
    errors = []
    with open(input_file) as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: Invalid JSON - {e}")
                continue

            missing = [f for f in required_fields if f not in record]
            if missing:
                errors.append(
                    f"Line {line_num}: Missing required field(s): {', '.join(missing)}"
                )
                continue

            interactions.append(record)

    if errors:
        console.print("[red]Validation errors found:[/red]")
        for err in errors:
            console.print(f"  {err}")
        raise SystemExit(1)

    if not interactions:
        console.print("[yellow]No interactions found in file[/yellow]")
        return

    console.print(f"Ingesting {len(interactions)} interactions...")

    with InfluenceTracker(config=config, use_mock_embedder=mock_embedder) as tracker:
        results = tracker.bulk_track(interactions)

        high_drift = sum(1 for r in results if "high_drift" in r["flags"])
        high_influence = sum(1 for r in results if "high_influence" in r["flags"])

        console.print(f"[green]Ingested {len(results)} interactions[/green]")
        console.print(f"  High drift events: {high_drift}")
        console.print(f"  High influence events: {high_influence}")


@cli.command()
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--top-n",
    type=int,
    default=10,
    help="Number of top agents to show",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
def summary(db: str, top_n: int, as_json: bool) -> None:
    """Show summary of tracked interactions."""
    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        # Rebuild graph from stored data
        events = tracker.get_events()
        scores = tracker.get_scores()
        tracker.graph.build_from_events(events, scores)

        report = tracker.summary(top_n=top_n)

        if as_json:
            console.print(report.model_dump_json(indent=2))
            return

        # Rich formatted output
        console.print("\n[bold]TraceIQ Summary Report[/bold]\n")

        # Overview table
        overview = Table(title="Overview")
        overview.add_column("Metric", style="cyan")
        overview.add_column("Value", style="green")

        overview.add_row("Total Events", str(report.total_events))
        overview.add_row("Unique Senders", str(report.unique_senders))
        overview.add_row("Unique Receivers", str(report.unique_receivers))
        overview.add_row("Avg Drift Delta", f"{report.avg_drift_delta:.4f}")
        overview.add_row("Avg Influence Score", f"{report.avg_influence_score:.4f}")
        overview.add_row("High Drift Count", str(report.high_drift_count))
        overview.add_row("High Influence Count", str(report.high_influence_count))

        console.print(overview)

        # Top influencers
        if report.top_influencers:
            console.print()
            inf_table = Table(title="Top Influencers")
            inf_table.add_column("Rank", style="dim")
            inf_table.add_column("Agent", style="cyan")
            inf_table.add_column("Score", style="green")

            for i, (agent, score) in enumerate(report.top_influencers, 1):
                inf_table.add_row(str(i), agent, f"{score:.4f}")

            console.print(inf_table)

        # Top susceptible
        if report.top_susceptible:
            console.print()
            sus_table = Table(title="Most Susceptible Agents")
            sus_table.add_column("Rank", style="dim")
            sus_table.add_column("Agent", style="cyan")
            sus_table.add_column("Score", style="yellow")

            for i, (agent, score) in enumerate(report.top_susceptible, 1):
                sus_table.add_row(str(i), agent, f"{score:.4f}")

            console.print(sus_table)

        # Influence chains
        if report.influence_chains:
            console.print("\n[bold]Influence Chains[/bold]")
            for chain in report.influence_chains[:5]:
                console.print("  [dim]->[/dim] " + " -> ".join(chain))


@cli.command()
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output file path",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["csv", "jsonl"]),
    default="csv",
    help="Export format",
)
def export(db: str, output: str, fmt: str) -> None:
    """Export data to CSV or JSONL."""
    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        if fmt == "csv":
            tracker.export_csv(output)
        else:
            tracker.export_jsonl(output)

    console.print(f"[green]Exported data to:[/green] {output}")


# IEEE Metrics Commands (v0.3.0)


@cli.command("propagation-risk")
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--window",
    type=int,
    default=10,
    help="Window size for time-based analysis",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
def propagation_risk(db: str, window: int, as_json: bool) -> None:
    """Show propagation risk (spectral radius of influence graph)."""
    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        events = tracker.get_events()
        scores = tracker.get_scores()
        tracker.graph.build_from_events(events, scores)

        # Current propagation risk
        current_pr = tracker.get_propagation_risk()

        # Over time
        pr_history = tracker.get_propagation_risk_over_time(window_size=window)

        if as_json:
            output = {
                "current_propagation_risk": current_pr,
                "history": [
                    {
                        "window_start": pr.window_start.isoformat(),
                        "window_end": pr.window_end.isoformat(),
                        "spectral_radius": pr.spectral_radius,
                        "edge_count": pr.edge_count,
                        "agent_count": pr.agent_count,
                    }
                    for pr in pr_history
                ],
            }
            console.print(json.dumps(output, indent=2))
            return

        console.print("\n[bold]Propagation Risk Analysis[/bold]\n")

        console.print(f"Current Propagation Risk: [cyan]{current_pr:.4f}[/cyan]")
        if current_pr > 1.0:
            console.print(
                "[yellow]Warning: PR > 1.0 indicates potential influence amplification[/yellow]"
            )

        if pr_history:
            console.print(f"\nHistory (window size={window}):")
            table = Table()
            table.add_column("Window Start", style="dim")
            table.add_column("Window End", style="dim")
            table.add_column("Spectral Radius", style="cyan")
            table.add_column("Edges", style="green")
            table.add_column("Agents", style="green")

            for pr in pr_history[-10:]:  # Show last 10
                table.add_row(
                    pr.window_start.strftime("%Y-%m-%d %H:%M"),
                    pr.window_end.strftime("%Y-%m-%d %H:%M"),
                    f"{pr.spectral_radius:.4f}",
                    str(pr.edge_count),
                    str(pr.agent_count),
                )

            console.print(table)


@cli.command("alerts")
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--threshold",
    type=float,
    default=2.0,
    help="Minimum Z-score threshold for alerts",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
def alerts(db: str, threshold: float, as_json: bool) -> None:
    """Show anomaly alerts (high Z-score events)."""
    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        alert_list = tracker.get_alerts(threshold=threshold)

        if as_json:
            output = [
                {
                    "event_id": str(a.event_id),
                    "Z_score": a.Z_score,
                    "IQx": a.IQx,
                    "RWI": a.RWI,
                    "drift_l2": a.drift_l2,
                }
                for a in alert_list
            ]
            console.print(json.dumps(output, indent=2))
            return

        console.print(f"\n[bold]Anomaly Alerts (Z > {threshold})[/bold]\n")

        if not alert_list:
            console.print("[green]No alerts found[/green]")
            return

        table = Table()
        table.add_column("Event ID", style="dim")
        table.add_column("Z-Score", style="red")
        table.add_column("IQx", style="cyan")
        table.add_column("RWI", style="yellow")
        table.add_column("Drift L2", style="green")

        for alert in alert_list[:20]:  # Limit display
            table.add_row(
                str(alert.event_id)[:8] + "...",
                f"{alert.Z_score:.2f}" if alert.Z_score else "-",
                f"{alert.IQx:.4f}" if alert.IQx else "-",
                f"{alert.RWI:.4f}" if alert.RWI else "-",
                f"{alert.drift_l2:.4f}" if alert.drift_l2 else "-",
            )

        console.print(table)
        console.print(f"\nTotal alerts: {len(alert_list)}")


@cli.command("risky-agents")
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--top-n",
    type=int,
    default=10,
    help="Number of top risky agents to show",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
def risky_agents(db: str, top_n: int, as_json: bool) -> None:
    """Show agents ranked by risk-weighted influence."""
    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        events = tracker.get_events()
        scores = tracker.get_scores()
        tracker.graph.build_from_events(events, scores)

        risky = tracker.get_risky_agents(top_n=top_n)

        if as_json:
            output = [
                {
                    "agent_id": agent_id,
                    "total_rwi": total_rwi,
                    "attack_surface": attack_surface,
                }
                for agent_id, total_rwi, attack_surface in risky
            ]
            console.print(json.dumps(output, indent=2))
            return

        console.print("\n[bold]Risky Agents (by RWI)[/bold]\n")

        if not risky:
            console.print("[green]No risky agents found[/green]")
            return

        table = Table()
        table.add_column("Rank", style="dim")
        table.add_column("Agent", style="cyan")
        table.add_column("Total RWI", style="red")
        table.add_column("Attack Surface", style="yellow")

        for i, (agent_id, total_rwi, attack_surface) in enumerate(risky, 1):
            table.add_row(
                str(i),
                agent_id,
                f"{total_rwi:.4f}",
                f"{attack_surface:.2f}",
            )

        console.print(table)


@cli.group()
def capabilities() -> None:
    """Manage agent capability registry."""
    pass


@capabilities.command("load")
@click.argument("registry_file", type=click.Path(exists=True))
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
def capabilities_load(registry_file: str, db: str) -> None:
    """Load capability registry from JSON file."""
    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
        capability_registry_path=registry_file,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        agents = tracker.capabilities.get_all_agents()
        console.print(f"[green]Loaded {len(agents)} agents from registry[/green]")

        for agent_id in agents:
            caps = tracker.capabilities.get_capabilities(agent_id)
            surface = tracker.capabilities.compute_attack_surface(agent_id)
            console.print(f"  {agent_id}: {caps} (AS={surface:.2f})")


@capabilities.command("show")
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--json",
    "as_json",
    is_flag=True,
    help="Output as JSON",
)
def capabilities_show(db: str, as_json: bool) -> None:
    """Show current capability weights and registered agents."""
    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        weights = tracker.capabilities.weights
        agents = tracker.capabilities.get_all_capabilities_models()

        if as_json:
            output = {
                "weights": weights,
                "agents": [a.model_dump() for a in agents],
            }
            console.print(json.dumps(output, indent=2))
            return

        console.print("\n[bold]Capability Weights[/bold]\n")
        weight_table = Table()
        weight_table.add_column("Capability", style="cyan")
        weight_table.add_column("Weight", style="green")

        for cap, weight in sorted(weights.items()):
            weight_table.add_row(cap, f"{weight:.2f}")

        console.print(weight_table)

        if agents:
            console.print("\n[bold]Registered Agents[/bold]\n")
            agent_table = Table()
            agent_table.add_column("Agent", style="cyan")
            agent_table.add_column("Capabilities", style="dim")
            agent_table.add_column("Attack Surface", style="yellow")

            for agent in agents:
                agent_table.add_row(
                    agent.agent_id,
                    ", ".join(agent.capabilities) or "-",
                    f"{agent.attack_surface:.2f}" if agent.attack_surface else "-",
                )

            console.print(agent_table)


@cli.group()
def plot() -> None:
    """Generate plots from tracked data."""
    pass


@plot.command("drift")
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output image path",
)
def plot_drift(db: str, output: str) -> None:
    """Plot drift over time."""
    from traceiq.plotting import plot_drift_over_time

    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        events = tracker.get_events()
        scores = tracker.get_scores()

        plot_drift_over_time(events, scores, output_path=output)

    console.print(f"[green]Saved drift plot to:[/green] {output}")


@plot.command("heatmap")
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output image path",
)
def plot_heatmap(db: str, output: str) -> None:
    """Plot influence heatmap."""
    from traceiq.plotting import plot_influence_heatmap

    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        events = tracker.get_events()
        scores = tracker.get_scores()
        tracker.graph.build_from_events(events, scores)

        plot_influence_heatmap(tracker.graph, output_path=output)

    console.print(f"[green]Saved heatmap to:[/green] {output}")


@plot.command("influencers")
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output image path",
)
@click.option(
    "--top-n",
    type=int,
    default=10,
    help="Number of top influencers to show",
)
def plot_influencers(db: str, output: str, top_n: int) -> None:
    """Plot top influencers bar chart."""
    from traceiq.plotting import plot_top_influencers

    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        events = tracker.get_events()
        scores = tracker.get_scores()
        tracker.graph.build_from_events(events, scores)

        plot_top_influencers(tracker.graph, n=top_n, output_path=output)

    console.print(f"[green]Saved influencers plot to:[/green] {output}")


@plot.command("network")
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output image path",
)
@click.option(
    "--min-weight",
    type=float,
    default=0.1,
    help="Minimum edge weight to display",
)
def plot_network(db: str, output: str, min_weight: float) -> None:
    """Plot influence network graph."""
    from traceiq.plotting import plot_influence_network

    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        events = tracker.get_events()
        scores = tracker.get_scores()
        tracker.graph.build_from_events(events, scores)

        plot_influence_network(
            tracker.graph,
            output_path=output,
            min_edge_weight=min_weight,
        )

    console.print(f"[green]Saved network plot to:[/green] {output}")


@plot.command("propagation-risk")
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output image path",
)
@click.option(
    "--window",
    type=int,
    default=10,
    help="Window size for time-based analysis",
)
def plot_propagation_risk_cmd(db: str, output: str, window: int) -> None:
    """Plot propagation risk over time."""
    from traceiq.plotting import plot_propagation_risk_over_time

    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        pr_results = tracker.get_propagation_risk_over_time(window_size=window)

        if not pr_results:
            console.print("[yellow]Not enough data for propagation risk plot[/yellow]")
            return

        plot_propagation_risk_over_time(pr_results, output_path=output)

    console.print(f"[green]Saved propagation risk plot to:[/green] {output}")


@plot.command("iqx-heatmap")
@click.option(
    "--db",
    type=click.Path(exists=True),
    default="traceiq.db",
    help="Path to SQLite database file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(),
    required=True,
    help="Output image path",
)
def plot_iqx_heatmap_cmd(db: str, output: str) -> None:
    """Plot IQx heatmap."""
    from traceiq.plotting import plot_iqx_heatmap

    config = TrackerConfig(
        storage_backend="sqlite",
        storage_path=db,
    )

    with InfluenceTracker(config=config, use_mock_embedder=True) as tracker:
        events = tracker.get_events()
        scores = tracker.get_scores()
        tracker.graph.build_from_events(events, scores)

        plot_iqx_heatmap(tracker.graph, output_path=output)

    console.print(f"[green]Saved IQx heatmap to:[/green] {output}")


if __name__ == "__main__":
    cli()
