"""TraceIQ: Measure AI-to-AI influence in multi-agent systems."""

from importlib.metadata import version as _get_version

from traceiq.capabilities import CapabilityRegistry
from traceiq.metrics import (
    compute_accumulated_influence,
    compute_attack_surface,
    compute_drift_l2,
    compute_IQx,
    compute_propagation_risk,
    compute_RWI,
    compute_z_score,
)
from traceiq.models import (
    AgentCapabilities,
    InteractionEvent,
    PropagationRiskResult,
    ScoreResult,
    SummaryReport,
    TrackerConfig,
)
from traceiq.tracker import InfluenceTracker

__version__ = _get_version("traceiq")

__all__ = [
    # Core classes
    "InfluenceTracker",
    "CapabilityRegistry",
    # Models
    "InteractionEvent",
    "ScoreResult",
    "SummaryReport",
    "TrackerConfig",
    "AgentCapabilities",
    "PropagationRiskResult",
    # Metrics functions
    "compute_drift_l2",
    "compute_IQx",
    "compute_accumulated_influence",
    "compute_propagation_risk",
    "compute_attack_surface",
    "compute_RWI",
    "compute_z_score",
    # Version
    "__version__",
]
