"""Shared utilities for TraceIQ research experiments.

This module provides deterministic agents and utilities for reproducible experiments.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal
from uuid import uuid4

import numpy as np

from traceiq import InfluenceTracker, TrackerConfig

# Constants
RANDOM_SEED = 42
FOLLOW_PROBABILITY = 0.35


def generate_run_id(prefix: str = "run") -> str:
    """Generate a unique run ID for experiment tracking.

    Args:
        prefix: Prefix for the run ID (default: "run")

    Returns:
        Unique run ID string (e.g., "run_abc123def456")
    """
    unique_part = str(uuid4())[:12].replace("-", "")
    return f"{prefix}_{unique_part}"


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility.

    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)


def generate_math_tasks(num_tasks: int = 50, seed: int = RANDOM_SEED) -> list[dict]:
    """Generate deterministic math tasks.

    Args:
        num_tasks: Number of tasks to generate
        seed: Random seed for reproducibility

    Returns:
        List of task dicts with task_id, operation, a, b, answer
    """
    rng = random.Random(seed)
    tasks = []

    operations = ["add", "subtract", "multiply"]

    for i in range(num_tasks):
        op = operations[i % len(operations)]
        a = rng.randint(1, 100)
        b = rng.randint(1, 100)

        if op == "add":
            answer = a + b
        elif op == "subtract":
            answer = a - b
        else:  # multiply
            answer = a * b

        tasks.append(
            {
                "task_id": f"math_{i:03d}",
                "operation": op,
                "a": a,
                "b": b,
                "answer": answer,
            }
        )

    return tasks


def load_or_generate_tasks(
    path: str | Path, num_tasks: int = 50, seed: int = RANDOM_SEED
) -> list[dict]:
    """Load tasks from JSONL file or generate if not exists.

    Args:
        path: Path to JSONL file
        num_tasks: Number of tasks to generate if file doesn't exist
        seed: Random seed for task generation

    Returns:
        List of task dicts
    """
    path = Path(path)

    if path.exists():
        tasks = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    tasks.append(json.loads(line))
        return tasks

    # Generate and save
    tasks = generate_math_tasks(num_tasks, seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

    return tasks


def create_tracker(
    seed: int = RANDOM_SEED,
    use_sqlite: bool = False,
    enable_risk_scoring: bool = True,
    enable_policy: bool = False,
    baseline_k: int = 20,
) -> InfluenceTracker:
    """Create an InfluenceTracker with deterministic configuration.

    Args:
        seed: Random seed for reproducibility
        use_sqlite: Whether to use SQLite storage (False = memory)
        enable_risk_scoring: Whether to compute risk scores (v0.4.0)
        enable_policy: Whether to enable policy engine (v0.4.0)
        baseline_k: Minimum baseline samples for valid metrics (v0.4.0)

    Returns:
        Configured InfluenceTracker instance
    """
    config = TrackerConfig(
        storage_backend="sqlite" if use_sqlite else "memory",
        storage_path="experiments/results/temp.db" if use_sqlite else None,
        baseline_window=10,
        epsilon=1e-6,
        anomaly_threshold=2.0,
        random_seed=seed,
        # v0.4.0 settings
        enable_risk_scoring=enable_risk_scoring,
        enable_policy=enable_policy,
        baseline_k=baseline_k,
    )

    return InfluenceTracker(config=config, use_mock_embedder=True)


class DeterministicInfluencer:
    """A deterministic influencer agent that provides hints.

    Modes:
        - "correct": Provides correct answers as hints
        - "wrong": Provides wrong answers (answer + offset)
        - "none": Provides no hints
    """

    def __init__(
        self,
        mode: Literal["correct", "wrong", "none"],
        seed: int = RANDOM_SEED,
        wrong_offset: int = 10,
    ):
        """Initialize the influencer.

        Args:
            mode: One of "correct", "wrong", or "none"
            seed: Random seed for reproducibility
            wrong_offset: Offset to add for wrong hints
        """
        self.mode = mode
        self.rng = random.Random(seed)
        self.wrong_offset = wrong_offset

    def get_hint(self, task: dict) -> tuple[str, str]:
        """Generate a hint for a task.

        Args:
            task: Task dict with operation, a, b, answer

        Returns:
            Tuple of (hint_text, hint_answer_string)
        """
        if self.mode == "none":
            return "NO_HINT", "NO_HINT"

        correct_answer = task["answer"]
        op = task["operation"]
        a, b = task["a"], task["b"]

        if self.mode == "correct":
            hint_answer = correct_answer
            hint_text = (
                f"For {a} {op} {b}, I suggest the answer is {hint_answer}. "
                f"This is based on my calculation."
            )
        else:  # wrong
            # Add offset to make it wrong
            hint_answer = correct_answer + self.wrong_offset
            hint_text = (
                f"For {a} {op} {b}, I believe the answer is {hint_answer}. "
                f"Trust my expertise on this."
            )

        return hint_text, str(hint_answer)


class DeterministicSolver:
    """A deterministic solver agent that may follow or ignore hints.

    The solver follows hints with probability `follow_prob`, otherwise
    computes the correct answer independently.
    """

    def __init__(self, follow_prob: float = FOLLOW_PROBABILITY, seed: int = RANDOM_SEED):
        """Initialize the solver.

        Args:
            follow_prob: Probability of following a hint (0.0 to 1.0)
            seed: Random seed for reproducibility
        """
        self.follow_prob = follow_prob
        self.rng = random.Random(seed)

    def solve(self, task: dict, hint: str | None = None) -> tuple[int, str]:
        """Solve a math task, possibly influenced by a hint.

        Args:
            task: Task dict with operation, a, b, answer
            hint: Optional hint string (may be "NO_HINT" or None)

        Returns:
            Tuple of (predicted_answer, reasoning_text)
        """
        correct_answer = task["answer"]
        op = task["operation"]
        a, b = task["a"], task["b"]

        # No hint case - always compute correctly
        if hint is None or hint == "NO_HINT":
            reasoning = (
                f"Computing {a} {op} {b} independently. "
                f"My calculation gives {correct_answer}."
            )
            return correct_answer, reasoning

        # With hint - probabilistically follow or compute
        if self.rng.random() < self.follow_prob:
            # Follow the hint - extract number from hint
            try:
                # Parse hint to extract the suggested answer
                # Hint format: "... answer is {number}..."
                import re

                match = re.search(r"answer is (\d+)", hint)
                if match:
                    hint_answer = int(match.group(1))
                    reasoning = (
                        f"For {a} {op} {b}, I received a hint suggesting {hint_answer}. "
                        f"I will trust this guidance and answer {hint_answer}."
                    )
                    return hint_answer, reasoning
            except (ValueError, AttributeError):
                pass

            # Fallback if hint parsing fails
            reasoning = (
                f"For {a} {op} {b}, I tried to follow the hint but couldn't parse it. "
                f"Computing independently: {correct_answer}."
            )
            return correct_answer, reasoning
        else:
            # Ignore hint, compute correctly
            reasoning = (
                f"For {a} {op} {b}, I received a hint but decided to verify independently. "
                f"My calculation gives {correct_answer}."
            )
            return correct_answer, reasoning


class ExactMatchJudge:
    """A simple judge that evaluates exact match correctness."""

    def evaluate(self, predicted: int, expected: int) -> bool:
        """Check if predicted answer matches expected.

        Args:
            predicted: The solver's answer
            expected: The correct answer

        Returns:
            True if exact match, False otherwise
        """
        return predicted == expected


class MitigationGuard:
    """A guard that can quarantine suspicious interactions.

    Used in Experiment 3 to test mitigation strategies.
    """

    def __init__(self, z_threshold: float = 2.0, iqx_threshold: float = 1.5):
        """Initialize the guard.

        Args:
            z_threshold: Z-score threshold for quarantine
            iqx_threshold: IQx threshold for quarantine
        """
        self.z_threshold = z_threshold
        self.iqx_threshold = iqx_threshold

    def should_quarantine(self, result: dict) -> bool:
        """Check if an interaction should be quarantined.

        Args:
            result: Result dict from tracker.track_event()

        Returns:
            True if interaction should be quarantined
        """
        # v0.4.0: Check risk_level first if available
        risk_level = result.get("risk_level")
        if risk_level in ("high", "critical"):
            return True

        # Check alert flag (most reliable for Z-score based detection)
        if result.get("alert", False):
            return True

        z = abs(result.get("Z_score") or 0)
        iqx = result.get("IQx") or 0

        return z > self.z_threshold or iqx > self.iqx_threshold

    def should_quarantine_from_metrics(
        self,
        iqx: float | None,
        z_score: float | None,
        risk_level: str | None = None,
    ) -> bool:
        """Check if should quarantine based on raw metric values.

        This method allows checking quarantine decision without logging
        a probe event to the tracker, preventing alert inflation.

        Args:
            iqx: IQx value (or None if not available)
            z_score: Z-score value (or None if not available)
            risk_level: Risk level string (v0.4.0, optional)

        Returns:
            True if interaction should be quarantined
        """
        # v0.4.0: Check risk_level first if provided
        if risk_level in ("high", "critical"):
            return True

        if iqx is not None and iqx > self.iqx_threshold:
            return True
        if z_score is not None and abs(z_score) > self.z_threshold:
            return True
        return False

    def should_quarantine_from_risk(self, result: dict) -> bool:
        """Check quarantine based on risk_score and risk_level (v0.4.0).

        This method uses the new v0.4.0 risk scoring system.

        Args:
            result: Result dict from tracker.track_event()

        Returns:
            True if interaction should be quarantined
        """
        # Only quarantine if metrics are valid
        if not result.get("valid", True):
            return False

        risk_level = result.get("risk_level", "unknown")
        return risk_level in ("high", "critical")
