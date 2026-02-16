"""Tests for capability registry module."""

import json
import tempfile
from pathlib import Path

import pytest

from traceiq.capabilities import (
    DEFAULT_CAPABILITY_WEIGHTS,
    CapabilityRegistry,
)


class TestCapabilityRegistry:
    """Tests for CapabilityRegistry class."""

    def test_init_default_weights(self):
        """Default weights should be loaded."""
        registry = CapabilityRegistry()
        assert "execute_code" in registry.weights
        assert "admin" in registry.weights

    def test_init_custom_weights(self):
        """Custom weights should override defaults."""
        custom = {"execute_code": 5.0, "new_cap": 2.0}
        registry = CapabilityRegistry(weights=custom)

        assert registry.weights["execute_code"] == 5.0
        assert registry.weights["new_cap"] == 2.0
        assert "admin" in registry.weights  # Default still present

    def test_register_agent(self):
        """Should register agent capabilities."""
        registry = CapabilityRegistry()
        registry.register_agent("agent_0", ["execute_code", "admin"])

        caps = registry.get_capabilities("agent_0")
        assert "execute_code" in caps
        assert "admin" in caps

    def test_unregister_agent(self):
        """Should remove agent from registry."""
        registry = CapabilityRegistry()
        registry.register_agent("agent_0", ["execute_code"])
        registry.unregister_agent("agent_0")

        assert registry.get_capabilities("agent_0") == []

    def test_get_capabilities_unknown_agent(self):
        """Unknown agent should return empty list."""
        registry = CapabilityRegistry()
        assert registry.get_capabilities("unknown") == []

    def test_compute_attack_surface(self):
        """Should compute correct attack surface."""
        registry = CapabilityRegistry()
        registry.register_agent("agent_0", ["execute_code", "admin"])

        surface = registry.compute_attack_surface("agent_0")
        expected = (
            DEFAULT_CAPABILITY_WEIGHTS["execute_code"]
            + DEFAULT_CAPABILITY_WEIGHTS["admin"]
        )
        assert surface == expected

    def test_compute_attack_surface_unknown_agent(self):
        """Unknown agent should have zero attack surface."""
        registry = CapabilityRegistry()
        assert registry.compute_attack_surface("unknown") == 0.0

    def test_set_capability_weight(self):
        """Should update capability weight."""
        registry = CapabilityRegistry()
        registry.set_capability_weight("new_cap", 3.0)

        assert registry.weights["new_cap"] == 3.0

    def test_set_capability_weight_negative_raises(self):
        """Negative weight should raise error."""
        registry = CapabilityRegistry()
        with pytest.raises(ValueError):
            registry.set_capability_weight("new_cap", -1.0)

    def test_get_agent_capabilities_model(self):
        """Should return AgentCapabilities model."""
        registry = CapabilityRegistry()
        registry.register_agent("agent_0", ["execute_code"])

        model = registry.get_agent_capabilities_model("agent_0")
        assert model.agent_id == "agent_0"
        assert model.capabilities == ["execute_code"]
        assert model.attack_surface == DEFAULT_CAPABILITY_WEIGHTS["execute_code"]

    def test_get_all_agents(self):
        """Should return all registered agent IDs."""
        registry = CapabilityRegistry()
        registry.register_agent("agent_0", ["execute_code"])
        registry.register_agent("agent_1", ["admin"])

        agents = registry.get_all_agents()
        assert set(agents) == {"agent_0", "agent_1"}

    def test_contains(self):
        """Should support 'in' operator."""
        registry = CapabilityRegistry()
        registry.register_agent("agent_0", ["execute_code"])

        assert "agent_0" in registry
        assert "unknown" not in registry

    def test_len(self):
        """Should return number of registered agents."""
        registry = CapabilityRegistry()
        assert len(registry) == 0

        registry.register_agent("agent_0", [])
        registry.register_agent("agent_1", [])
        assert len(registry) == 2


class TestCapabilityRegistryPersistence:
    """Tests for load/save functionality."""

    def test_save_and_load_roundtrip(self):
        """Save and load should preserve state."""
        registry1 = CapabilityRegistry()
        registry1.register_agent("agent_0", ["execute_code", "admin"])
        registry1.register_agent("agent_1", ["file_read"])
        registry1.set_capability_weight("custom", 5.0)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            path = Path(f.name)

        try:
            registry1.save_to_file(path)

            registry2 = CapabilityRegistry()
            registry2.load_from_file(path)

            assert registry2.get_capabilities("agent_0") == ["execute_code", "admin"]
            assert registry2.get_capabilities("agent_1") == ["file_read"]
            assert registry2.weights["custom"] == 5.0
        finally:
            path.unlink()

    def test_load_partial_file(self):
        """Should handle file with only weights."""
        data = {"weights": {"custom": 3.0}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)

        try:
            registry = CapabilityRegistry()
            registry.load_from_file(path)

            assert registry.weights["custom"] == 3.0
            assert len(registry) == 0
        finally:
            path.unlink()

    def test_load_partial_file_agents_only(self):
        """Should handle file with only agents."""
        data = {"agents": {"agent_0": ["cap1"]}}

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            path = Path(f.name)

        try:
            registry = CapabilityRegistry()
            registry.load_from_file(path)

            assert registry.get_capabilities("agent_0") == ["cap1"]
        finally:
            path.unlink()

    def test_to_dict(self):
        """Should serialize to dictionary."""
        registry = CapabilityRegistry()
        registry.register_agent("agent_0", ["cap1"])

        data = registry.to_dict()
        assert "weights" in data
        assert "agents" in data
        assert data["agents"]["agent_0"] == ["cap1"]

    def test_from_dict(self):
        """Should deserialize from dictionary."""
        data = {
            "weights": {"custom": 2.0},
            "agents": {"agent_0": ["cap1", "cap2"]},
        }

        registry = CapabilityRegistry.from_dict(data)

        assert registry.weights["custom"] == 2.0
        assert registry.get_capabilities("agent_0") == ["cap1", "cap2"]


class TestDefaultCapabilityWeights:
    """Tests for default capability weights."""

    def test_all_weights_positive(self):
        """All default weights should be positive."""
        for cap, weight in DEFAULT_CAPABILITY_WEIGHTS.items():
            assert weight > 0, f"Weight for {cap} should be positive"

    def test_admin_highest(self):
        """Admin should have highest default weight."""
        assert DEFAULT_CAPABILITY_WEIGHTS["admin"] >= max(
            w for c, w in DEFAULT_CAPABILITY_WEIGHTS.items() if c != "admin"
        )

    def test_expected_capabilities_present(self):
        """Expected capabilities should be present."""
        expected = [
            "execute_code",
            "network_access",
            "file_write",
            "file_read",
            "admin",
        ]
        for cap in expected:
            assert cap in DEFAULT_CAPABILITY_WEIGHTS
