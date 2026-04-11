"""Canonical enum definitions for the tool API.

Step 1.1 establishes the stable enum surface used by later canonical models.
"""

from __future__ import annotations

from enum import Enum


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
