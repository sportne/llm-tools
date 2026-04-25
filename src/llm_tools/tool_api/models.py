"""Canonical enum and model definitions for the tool API."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class SideEffectClass(str, Enum):  # noqa: UP042
    """Classes of side effects a tool may perform."""

    NONE = "none"
    LOCAL_READ = "local_read"
    LOCAL_WRITE = "local_write"
    EXTERNAL_READ = "external_read"
    EXTERNAL_WRITE = "external_write"


class RiskLevel(str, Enum):  # noqa: UP042
    """Coarse-grained risk classification for tool metadata."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ErrorCode(str, Enum):  # noqa: UP042
    """Normalized runtime error categories."""

    TOOL_NOT_FOUND = "tool_not_found"
    INPUT_VALIDATION_ERROR = "input_validation_error"
    OUTPUT_VALIDATION_ERROR = "output_validation_error"
    POLICY_DENIED = "policy_denied"
    TIMEOUT = "timeout"
    DEPENDENCY_MISSING = "dependency_missing"
    EXECUTION_FAILED = "execution_failed"
    RUNTIME_ERROR = "runtime_error"


class PolicyVerdict(str, Enum):  # noqa: UP042
    """High-level policy outcomes for a tool invocation."""

    ALLOW = "allow"
    DENY = "deny"
    REQUIRE_APPROVAL = "require_approval"


class ToolSpec(BaseModel):
    """Canonical metadata describing a tool."""

    name: str = Field(min_length=1)
    version: str = Field(default="0.1.0", min_length=1)
    description: str
    tags: list[str] = Field(default_factory=list)

    side_effects: SideEffectClass = SideEffectClass.NONE
    idempotent: bool = True
    deterministic: bool = True
    timeout_seconds: int | None = None

    risk_level: RiskLevel = RiskLevel.LOW
    requires_network: bool = False
    requires_filesystem: bool = False
    requires_subprocess: bool = False
    writes_internal_workspace_cache: bool = False
    required_secrets: list[str] = Field(default_factory=list)

    cost_hint: str | None = None
    retain_output_in_execution_record: bool = True


class SourceProvenanceRef(BaseModel):
    """Structured reference to a source material visible to a tool invocation."""

    source_kind: str = Field(min_length=1)
    source_id: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    whole_source_reproduction_allowed: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProtectionProvenanceSnapshot(BaseModel):
    """Collection of source references currently visible to protection checks."""

    sources: list[SourceProvenanceRef] = Field(default_factory=list)


class ToolContext(BaseModel):
    """Runtime context for a single tool invocation."""

    invocation_id: str = Field(min_length=1)
    workspace: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    source_provenance: list[SourceProvenanceRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ToolInvocationRequest(BaseModel):
    """Normalized request to invoke a tool."""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)
    tool_call_id: str | None = None


class RetryableToolExecutionError(RuntimeError):
    """Execution failure that should be surfaced as retryable."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.retryable = True


class ToolError(BaseModel):
    """Normalized error payload returned across the runtime boundary."""

    code: ErrorCode
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)


class ToolResult(BaseModel):
    """Normalized result envelope produced by the runtime."""

    ok: bool
    tool_name: str
    tool_version: str

    output: dict[str, Any] | None = None
    error: ToolError | None = None

    logs: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    source_provenance: list[SourceProvenanceRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PolicyDecision(BaseModel):
    """Result of evaluating execution policy for a tool invocation."""

    allowed: bool
    reason: str
    requires_approval: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExecutionRecord(BaseModel):
    """Structured observability record for a tool execution."""

    invocation_id: str
    tool_name: str
    tool_version: str

    started_at: str
    ended_at: str | None = None
    duration_ms: int | None = None

    request: ToolInvocationRequest
    validated_input: dict[str, Any] | None = None
    redacted_input: dict[str, Any] | None = None
    validated_output: dict[str, Any] | None = None
    redacted_output: dict[str, Any] | None = None

    ok: bool | None = None
    error_code: ErrorCode | None = None

    policy_decision: PolicyDecision | None = None
    logs: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    source_provenance: list[SourceProvenanceRef] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
