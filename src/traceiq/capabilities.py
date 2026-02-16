"""Agent capability registry for attack surface computation.

This module provides a registry for tracking agent capabilities and computing
attack surface scores. The attack surface represents the potential risk
associated with an agent based on its permissions/capabilities.

Default capability weights are provided based on common security risk assessments:
- execute_code: 1.0 (high risk - can run arbitrary code)
- admin: 1.5 (highest risk - full system access)
- network_access: 0.8 (can exfiltrate data or communicate externally)
- file_write: 0.7 (can modify system state)
- memory_access: 0.5 (can read/write memory)
- file_read: 0.3 (limited risk - read-only)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from traceiq.metrics import compute_attack_surface
from traceiq.models import AgentCapabilities

# Default weights based on security risk assessment
DEFAULT_CAPABILITY_WEIGHTS: dict[str, float] = {
    "execute_code": 1.0,
    "network_access": 0.8,
    "file_write": 0.7,
    "file_read": 0.3,
    "admin": 1.5,
    "memory_access": 0.5,
    "database_write": 0.6,
    "database_read": 0.3,
    "api_access": 0.4,
    "subprocess": 0.9,
}


class CapabilityRegistry:
    """Registry for agent capabilities and attack surface computation."""

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        """Initialize the capability registry.

        Args:
            weights: Custom capability weights. If None, uses DEFAULT_CAPABILITY_WEIGHTS.
        """
        self._weights = dict(DEFAULT_CAPABILITY_WEIGHTS)
        if weights:
            self._weights.update(weights)

        # Maps agent_id -> list of capabilities
        self._agents: dict[str, list[str]] = {}

    @property
    def weights(self) -> dict[str, float]:
        """Get current capability weights."""
        return dict(self._weights)

    def load_from_file(self, path: str | Path) -> None:
        """Load registry state from a JSON file.

        Expected format:
        {
            "weights": {"capability": weight, ...},
            "agents": {"agent_id": ["cap1", "cap2"], ...}
        }

        Args:
            path: Path to JSON file
        """
        path = Path(path)
        with open(path) as f:
            data = json.load(f)

        if "weights" in data:
            self._weights.update(data["weights"])

        if "agents" in data:
            for agent_id, caps in data["agents"].items():
                self._agents[agent_id] = list(caps)

    def save_to_file(self, path: str | Path) -> None:
        """Save registry state to a JSON file.

        Args:
            path: Path to JSON file
        """
        path = Path(path)
        data = {
            "weights": self._weights,
            "agents": self._agents,
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    def register_agent(self, agent_id: str, capabilities: list[str]) -> None:
        """Register an agent with its capabilities.

        Args:
            agent_id: Unique agent identifier
            capabilities: List of capability names
        """
        self._agents[agent_id] = list(capabilities)

    def unregister_agent(self, agent_id: str) -> None:
        """Remove an agent from the registry.

        Args:
            agent_id: Agent to remove
        """
        self._agents.pop(agent_id, None)

    def get_capabilities(self, agent_id: str) -> list[str]:
        """Get capabilities for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            List of capability names, empty if agent not registered
        """
        return list(self._agents.get(agent_id, []))

    def compute_attack_surface(self, agent_id: str) -> float:
        """Compute attack surface score for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            Sum of capability weights for this agent
        """
        caps = self.get_capabilities(agent_id)
        return compute_attack_surface(caps, self._weights)

    def set_capability_weight(self, capability: str, weight: float) -> None:
        """Set or update weight for a capability.

        Args:
            capability: Capability name
            weight: Weight value (should be >= 0)
        """
        if weight < 0:
            raise ValueError("Capability weight must be non-negative")
        self._weights[capability] = weight

    def get_agent_capabilities_model(self, agent_id: str) -> AgentCapabilities:
        """Get AgentCapabilities model for an agent.

        Args:
            agent_id: Agent identifier

        Returns:
            AgentCapabilities model with computed attack surface
        """
        caps = self.get_capabilities(agent_id)
        attack_surface = self.compute_attack_surface(agent_id)
        return AgentCapabilities(
            agent_id=agent_id,
            capabilities=caps,
            attack_surface=attack_surface,
        )

    def get_all_agents(self) -> list[str]:
        """Get list of all registered agent IDs.

        Returns:
            List of agent identifiers
        """
        return list(self._agents.keys())

    def get_all_capabilities_models(self) -> list[AgentCapabilities]:
        """Get AgentCapabilities models for all registered agents.

        Returns:
            List of AgentCapabilities models
        """
        return [self.get_agent_capabilities_model(aid) for aid in self._agents]

    def to_dict(self) -> dict[str, Any]:
        """Serialize registry to dictionary.

        Returns:
            Dict with weights and agents
        """
        return {
            "weights": self._weights,
            "agents": self._agents,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> CapabilityRegistry:
        """Create registry from dictionary.

        Args:
            data: Dict with weights and agents

        Returns:
            New CapabilityRegistry instance
        """
        registry = cls(weights=data.get("weights"))
        if "agents" in data:
            for agent_id, caps in data["agents"].items():
                registry.register_agent(agent_id, caps)
        return registry

    def __len__(self) -> int:
        """Return number of registered agents."""
        return len(self._agents)

    def __contains__(self, agent_id: str) -> bool:
        """Check if agent is registered."""
        return agent_id in self._agents
