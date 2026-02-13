"""TraceIQ: Measure AI-to-AI influence in multi-agent systems."""

from traceiq.models import InteractionEvent, ScoreResult, SummaryReport, TrackerConfig
from traceiq.tracker import InfluenceTracker

__version__ = "0.2.0"

__all__ = [
    "InfluenceTracker",
    "InteractionEvent",
    "ScoreResult",
    "SummaryReport",
    "TrackerConfig",
    "__version__",
]
