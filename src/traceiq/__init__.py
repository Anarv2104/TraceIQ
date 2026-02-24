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
from traceiq.policy import PolicyEngine
from traceiq.risk import (
    RiskResult,
    RiskThresholds,
    assign_risk_level,
    calibrate_thresholds,
    compute_risk_score,
)
from traceiq.schema import TraceIQEvent, compute_state_quality
from traceiq.tracker import InfluenceTracker
from traceiq.validity import ValidityResult, check_validity

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
    # Extended schema (v0.4.0)
    "TraceIQEvent",
    "compute_state_quality",
    # Validity (v0.4.0)
    "ValidityResult",
    "check_validity",
    # Risk scoring (v0.4.0)
    "RiskResult",
    "RiskThresholds",
    "compute_risk_score",
    "calibrate_thresholds",
    "assign_risk_level",
    # Policy (v0.4.0)
    "PolicyEngine",
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
