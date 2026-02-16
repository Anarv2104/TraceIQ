"""TraceIQ: Measure AI-to-AI influence in multi-agent systems."""

from importlib.metadata import version as _get_version

from traceiq.models import InteractionEvent, ScoreResult, SummaryReport, TrackerConfig
from traceiq.tracker import InfluenceTracker

__version__ = _get_version("traceiq")

__all__ = [
    "InfluenceTracker",
    "InteractionEvent",
    "ScoreResult",
    "SummaryReport",
    "TrackerConfig",
    "__version__",
]
