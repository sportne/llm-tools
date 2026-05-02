"""Assistant-facing tool capability models and summaries."""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from llm_tools.tool_api import SideEffectClass

AssistantToolStatus = Literal[
    "available",
    "disabled",
    "missing_workspace",
    "missing_credentials",
    "permission_blocked",
]


AssistantBlockedCapability = Literal["network", "filesystem", "subprocess"]


class AssistantToolCapabilityReasonCode(str, Enum):  # noqa: UP042
    """Structured reason codes for assistant tool availability."""

    WORKSPACE_REQUIRED = "workspace_required"
    MISSING_CREDENTIALS = "missing_credentials"
    NETWORK_PERMISSION_BLOCKED = "network_permission_blocked"
    FILESYSTEM_PERMISSION_BLOCKED = "filesystem_permission_blocked"
    SUBPROCESS_PERMISSION_BLOCKED = "subprocess_permission_blocked"
    APPROVAL_REQUIRED = "approval_required"


class AssistantToolCapabilityReason(BaseModel):
    """One structured reason explaining a tool's capability state."""

    code: AssistantToolCapabilityReasonCode
    message: str
    missing_secrets: list[str] = Field(default_factory=list)
    blocked_capability: AssistantBlockedCapability | None = None


class AssistantToolApprovalGate(BaseModel):
    """Structured approval-gate metadata for one tool."""

    required: bool = False
    side_effects: SideEffectClass
    reason_code: AssistantToolCapabilityReasonCode | None = None
    message: str | None = None


class AssistantToolGroupCapabilitySummary(BaseModel):
    """Roll-up counts for one assistant tool group."""

    group: str
    total_tools: int = 0
    enabled_tools: int = 0
    exposed_tools: int = 0
    available_tools: int = 0
    disabled_tools: int = 0
    missing_workspace_tools: int = 0
    missing_credentials_tools: int = 0
    permission_blocked_tools: int = 0
    approval_gated_tools: int = 0


class AssistantToolCapability(BaseModel):
    """One tool plus its assistant-facing availability state."""

    tool_name: str
    group: str
    enabled: bool = False
    exposed_to_model: bool = False
    status: AssistantToolStatus
    detail: str | None = None
    primary_reason: AssistantToolCapabilityReason | None = None
    reasons: list[AssistantToolCapabilityReason] = Field(default_factory=list)
    approval_required: bool = False
    approval_gate: AssistantToolApprovalGate
    side_effects: SideEffectClass
    requires_network: bool = False
    requires_filesystem: bool = False
    requires_subprocess: bool = False
    required_secrets: list[str] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _populate_approval_gate(cls, data: Any) -> Any:
        """Backfill the additive approval gate for legacy payloads."""
        if not isinstance(data, dict):
            return data
        if data.get("approval_gate") is not None:
            return data
        side_effects = data.get("side_effects")
        if side_effects is None:
            return data
        approval_required = bool(data.get("approval_required", False))
        payload = dict(data)
        payload["approval_gate"] = AssistantToolApprovalGate(
            required=approval_required,
            side_effects=side_effects,
            reason_code=(
                AssistantToolCapabilityReasonCode.APPROVAL_REQUIRED
                if approval_required
                else None
            ),
            message=(
                "This tool requires approval before execution."
                if approval_required
                else None
            ),
        ).model_dump(mode="python")
        return payload


__all__ = [
    "AssistantBlockedCapability",
    "AssistantToolApprovalGate",
    "AssistantToolCapability",
    "AssistantToolCapabilityReason",
    "AssistantToolCapabilityReasonCode",
    "AssistantToolGroupCapabilitySummary",
    "AssistantToolStatus",
]
