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
@click.version_option(version="0.1.0", prog_name="traceiq")
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

    with InfluenceTracker(config=config, use_mock_embedder=mock_embedder) as tracker:
        interactions = []
        with open(input_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    interactions.append(json.loads(line))

        if not interactions:
            console.print("[yellow]No interactions found in file[/yellow]")
            return

        console.print(f"Ingesting {len(interactions)} interactions...")

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


if __name__ == "__main__":
    cli()
