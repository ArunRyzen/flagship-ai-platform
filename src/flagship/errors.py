"""Domain exceptions."""

from __future__ import annotations


class FlagshipError(Exception):
    """Base class."""


class GuardrailViolation(FlagshipError):  # noqa: N818 - clearer domain term than "Error"
    """Input violated a guardrail (prompt injection)."""


class PolicyError(FlagshipError):
    """The agent's decision policy (LLM) failed."""


class GateFailure(FlagshipError):  # noqa: N818 - "Failure" reads better than "Error"
    """An evaluation did not meet its threshold (used to fail CI)."""
