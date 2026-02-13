"""Tests for plotting module."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from traceiq.models import InteractionEvent, ScoreResult

from traceiq.graph import InfluenceGraph

# Check if matplotlib is available
try:
    import matplotlib

    matplotlib.use("Agg")  # Use non-interactive backend
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


@pytest.mark.skipif(not HAS_MATPLOTLIB, reason="matplotlib not installed")
class TestPlotting:
    """Tests for plotting functions."""

    def test_plot_drift_over_time(
        self,
        sample_events: list[InteractionEvent],
        sample_scores: list[ScoreResult],
        tmp_path: Path,
    ) -> None:
        """Test drift over time plot generation."""
        from traceiq.plotting import plot_drift_over_time

        output_path = tmp_path / "drift.png"
        plot_drift_over_time(sample_events, sample_scores, output_path=output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_plot_influence_heatmap(
        self,
        sample_events: list[InteractionEvent],
        sample_scores: list[ScoreResult],
        tmp_path: Path,
    ) -> None:
        """Test influence heatmap generation."""
        from traceiq.plotting import plot_influence_heatmap

        graph = InfluenceGraph()
        graph.build_from_events(sample_events, sample_scores)

        output_path = tmp_path / "heatmap.png"
        plot_influence_heatmap(graph, output_path=output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_plot_top_influencers(
        self,
        sample_events: list[InteractionEvent],
        sample_scores: list[ScoreResult],
        tmp_path: Path,
    ) -> None:
        """Test top influencers bar chart generation."""
        from traceiq.plotting import plot_top_influencers

        graph = InfluenceGraph()
        graph.build_from_events(sample_events, sample_scores)

        output_path = tmp_path / "influencers.png"
        plot_top_influencers(graph, n=5, output_path=output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_plot_top_susceptible(
        self,
        sample_events: list[InteractionEvent],
        sample_scores: list[ScoreResult],
        tmp_path: Path,
    ) -> None:
        """Test top susceptible agents chart generation."""
        from traceiq.plotting import plot_top_susceptible

        graph = InfluenceGraph()
        graph.build_from_events(sample_events, sample_scores)

        output_path = tmp_path / "susceptible.png"
        plot_top_susceptible(graph, n=5, output_path=output_path)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_plot_influence_network(
        self,
        sample_events: list[InteractionEvent],
        sample_scores: list[ScoreResult],
        tmp_path: Path,
    ) -> None:
        """Test influence network graph generation."""
        from traceiq.plotting import plot_influence_network

        graph = InfluenceGraph()
        graph.build_from_events(sample_events, sample_scores)

        output_path = tmp_path / "network.png"
        plot_influence_network(graph, output_path=output_path, min_edge_weight=0.0)

        assert output_path.exists()
        assert output_path.stat().st_size > 0

    def test_plot_empty_data(self, tmp_path: Path) -> None:
        """Test plotting with empty data doesn't crash."""
        from traceiq.plotting import plot_influence_heatmap, plot_top_influencers

        graph = InfluenceGraph()

        # These should not raise, just print messages
        plot_influence_heatmap(graph, output_path=tmp_path / "empty_heatmap.png")
        plot_top_influencers(graph, output_path=tmp_path / "empty_influencers.png")


class TestPlottingImportError:
    """Test helpful error when matplotlib is missing."""

    def test_check_matplotlib_error_message(self) -> None:
        """Test that _check_matplotlib raises helpful error."""
        # We can't truly test missing matplotlib if it's installed,
        # but we can verify the error message format
        from traceiq.plotting import _check_matplotlib

        # If matplotlib is available, this should not raise
        if HAS_MATPLOTLIB:
            _check_matplotlib()  # Should not raise
        else:
            with pytest.raises(ImportError) as exc_info:
                _check_matplotlib()
            assert "matplotlib" in str(exc_info.value)
            assert "pip install" in str(exc_info.value)
