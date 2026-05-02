"""Redaction models and helpers for runtime observability payloads."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class RedactionTarget(str, Enum):  # noqa: UP042
    """Payload surfaces where redaction rules may apply."""

    INPUT = "input"
    OUTPUT = "output"
    ERROR_DETAILS = "error_details"
    LOGS = "logs"
    ARTIFACTS = "artifacts"
    ALL = "all"


class RedactionRule(BaseModel):
    """A single redaction rule using field names and/or structured paths."""

    field_names: set[str] = Field(default_factory=set)
    paths: set[str] = Field(default_factory=set)
    targets: set[RedactionTarget] = Field(default_factory=lambda: {RedactionTarget.ALL})
    replacement: str = "[REDACTED]"


class RedactionConfig(BaseModel):
    """Configurable redaction policy for runtime outputs and observability."""

    rules: list[RedactionRule] = Field(default_factory=list)
    tool_rules: dict[str, list[RedactionRule]] = Field(default_factory=dict)
    redact_logs: bool = True
    redact_artifacts: bool = True
    retain_unredacted_inputs: bool = False
    retain_unredacted_outputs: bool = False


DEFAULT_SENSITIVE_FIELD_NAMES: set[str] = {
    "password",
    "secret",
    "token",
    "api_key",
    "access_token",
    "refresh_token",
    "authorization",
}


__all__ = [
    "DEFAULT_SENSITIVE_FIELD_NAMES",
    "RedactionConfig",
    "RedactionRule",
    "RedactionTarget",
]
