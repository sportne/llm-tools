"""Assistant-facing tool capability models and summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence

from llm_tools.apps.assistant_tool_capabilities_models import (
    AssistantBlockedCapability as AssistantBlockedCapability,
)
from llm_tools.apps.assistant_tool_capabilities_models import (
    AssistantToolApprovalGate as AssistantToolApprovalGate,
)
from llm_tools.apps.assistant_tool_capabilities_models import (
    AssistantToolCapability as AssistantToolCapability,
)
from llm_tools.apps.assistant_tool_capabilities_models import (
    AssistantToolCapabilityReason as AssistantToolCapabilityReason,
)
from llm_tools.apps.assistant_tool_capabilities_models import (
    AssistantToolCapabilityReasonCode as AssistantToolCapabilityReasonCode,
)
from llm_tools.apps.assistant_tool_capabilities_models import (
    AssistantToolGroupCapabilitySummary as AssistantToolGroupCapabilitySummary,
)
from llm_tools.apps.assistant_tool_capabilities_models import (
    AssistantToolStatus as AssistantToolStatus,
)
from llm_tools.tool_api import SideEffectClass, ToolSpec


def assistant_tool_group(spec: ToolSpec) -> str:
    """Return the assistant UI group for one tool spec."""
    tags = set(spec.tags)
    if "gitlab" in tags:
        return "GitLab"
    if "jira" in tags:
        return "Jira"
    if "confluence" in tags:
        return "Confluence"
    if "bitbucket" in tags:
        return "Bitbucket"
    if "git" in tags:
        return "Git"
    if "filesystem" in tags:
        return "Local Files"
    return "Other"


_PERMISSION_BLOCKED_REASON_CODES = {
    AssistantToolCapabilityReasonCode.NETWORK_PERMISSION_BLOCKED,
    AssistantToolCapabilityReasonCode.FILESYSTEM_PERMISSION_BLOCKED,
    AssistantToolCapabilityReasonCode.SUBPROCESS_PERMISSION_BLOCKED,
}


def _build_tool_blocking_reasons(
    *,
    spec: ToolSpec,
    root_path: str | None,
    env: Mapping[str, str],
    allow_network: bool,
    allow_filesystem: bool,
    allow_subprocess: bool,
) -> list[AssistantToolCapabilityReason]:
    """Return blocking capability reasons in legacy precedence order."""
    reasons: list[AssistantToolCapabilityReason] = []
    if spec.requires_filesystem and root_path is None:
        reasons.append(
            AssistantToolCapabilityReason(
                code=AssistantToolCapabilityReasonCode.WORKSPACE_REQUIRED,
                message="Select a workspace root first.",
            )
        )

    missing_secrets = sorted(
        secret for secret in spec.required_secrets if not env.get(secret)
    )
    if missing_secrets:
        reasons.append(
            AssistantToolCapabilityReason(
                code=AssistantToolCapabilityReasonCode.MISSING_CREDENTIALS,
                message="Missing credentials: " + ", ".join(missing_secrets),
                missing_secrets=missing_secrets,
            )
        )

    capability_checks: tuple[
        tuple[
            AssistantBlockedCapability,
            bool,
            bool,
            AssistantToolCapabilityReasonCode,
        ],
        ...,
    ] = (
        (
            "network",
            spec.requires_network,
            allow_network,
            AssistantToolCapabilityReasonCode.NETWORK_PERMISSION_BLOCKED,
        ),
        (
            "filesystem",
            spec.requires_filesystem,
            allow_filesystem,
            AssistantToolCapabilityReasonCode.FILESYSTEM_PERMISSION_BLOCKED,
        ),
        (
            "subprocess",
            spec.requires_subprocess,
            allow_subprocess,
            AssistantToolCapabilityReasonCode.SUBPROCESS_PERMISSION_BLOCKED,
        ),
    )
    for capability_name, is_required, is_allowed, reason_code in capability_checks:
        if is_required and not is_allowed:
            reasons.append(
                AssistantToolCapabilityReason(
                    code=reason_code,
                    message="Current session permissions do not allow this tool.",
                    blocked_capability=capability_name,
                )
            )

    return reasons


def _capability_status_from_reasons(
    *,
    enabled: bool,
    reasons: Sequence[AssistantToolCapabilityReason],
) -> AssistantToolStatus:
    """Return the legacy tool status from typed capability reasons."""
    if not enabled:
        return "disabled"
    for reason in reasons:
        if reason.code is AssistantToolCapabilityReasonCode.WORKSPACE_REQUIRED:
            return "missing_workspace"
        if reason.code is AssistantToolCapabilityReasonCode.MISSING_CREDENTIALS:
            return "missing_credentials"
        if reason.code in _PERMISSION_BLOCKED_REASON_CODES:
            return "permission_blocked"
    return "available"


def _legacy_detail_from_reasons(
    reasons: Sequence[AssistantToolCapabilityReason],
) -> str | None:
    """Return the backward-compatible summary string for capability blockers."""
    details: list[str] = []
    permission_message_added = False
    for reason in reasons:
        if reason.code in _PERMISSION_BLOCKED_REASON_CODES:
            if permission_message_added:
                continue
            permission_message_added = True
        details.append(reason.message)
    return " ".join(details) if details else None


def _build_tool_approval_gate(
    *,
    spec: ToolSpec,
    require_approval_for: set[SideEffectClass],
) -> AssistantToolApprovalGate:
    """Return structured approval-gate metadata for one tool."""
    required = spec.side_effects in require_approval_for
    return AssistantToolApprovalGate(
        required=required,
        side_effects=spec.side_effects,
        reason_code=(
            AssistantToolCapabilityReasonCode.APPROVAL_REQUIRED if required else None
        ),
        message=("This tool requires approval before execution." if required else None),
    )


def _build_tool_capability(
    *,
    tool_name: str,
    spec: ToolSpec,
    enabled: bool,
    group_name: str,
    root_path: str | None,
    env: Mapping[str, str],
    allow_network: bool,
    allow_filesystem: bool,
    allow_subprocess: bool,
    require_approval_for: set[SideEffectClass],
) -> AssistantToolCapability:
    """Return one assistant tool capability record."""
    reasons = (
        _build_tool_blocking_reasons(
            spec=spec,
            root_path=root_path,
            env=env,
            allow_network=allow_network,
            allow_filesystem=allow_filesystem,
            allow_subprocess=allow_subprocess,
        )
        if enabled
        else []
    )
    status = _capability_status_from_reasons(enabled=enabled, reasons=reasons)
    approval_gate = _build_tool_approval_gate(
        spec=spec,
        require_approval_for=require_approval_for,
    )
    return AssistantToolCapability(
        tool_name=tool_name,
        group=group_name,
        enabled=enabled,
        exposed_to_model=enabled and status == "available",
        status=status,
        detail=_legacy_detail_from_reasons(reasons),
        primary_reason=reasons[0] if reasons else None,
        reasons=reasons,
        approval_required=approval_gate.required,
        approval_gate=approval_gate,
        side_effects=spec.side_effects,
        requires_network=spec.requires_network,
        requires_filesystem=spec.requires_filesystem,
        requires_subprocess=spec.requires_subprocess,
        required_secrets=list(spec.required_secrets),
    )


def build_tool_capabilities(
    *,
    tool_specs: dict[str, ToolSpec],
    enabled_tools: set[str],
    root_path: str | None,
    env: dict[str, str],
    allow_network: bool,
    allow_filesystem: bool,
    allow_subprocess: bool,
    require_approval_for: set[SideEffectClass],
) -> dict[str, list[AssistantToolCapability]]:
    """Return grouped assistant-facing capability state for all tools."""
    grouped: dict[str, list[AssistantToolCapability]] = defaultdict(list)
    for tool_name, spec in sorted(tool_specs.items()):
        enabled = tool_name in enabled_tools
        group_name = assistant_tool_group(spec)
        grouped[group_name].append(
            _build_tool_capability(
                tool_name=tool_name,
                spec=spec,
                enabled=enabled,
                group_name=group_name,
                root_path=root_path,
                env=env,
                allow_network=allow_network,
                allow_filesystem=allow_filesystem,
                allow_subprocess=allow_subprocess,
                require_approval_for=require_approval_for,
            )
        )
    return dict(sorted(grouped.items()))


def build_tool_group_capability_summaries(
    capability_groups: Mapping[str, Sequence[AssistantToolCapability]],
) -> dict[str, AssistantToolGroupCapabilitySummary]:
    """Return group-level capability counts for frontend consumption."""
    summaries: dict[str, AssistantToolGroupCapabilitySummary] = {}
    for group_name, items in sorted(capability_groups.items()):
        summaries[group_name] = AssistantToolGroupCapabilitySummary(
            group=group_name,
            total_tools=len(items),
            enabled_tools=sum(item.enabled for item in items),
            exposed_tools=sum(item.exposed_to_model for item in items),
            available_tools=sum(item.status == "available" for item in items),
            disabled_tools=sum(item.status == "disabled" for item in items),
            missing_workspace_tools=sum(
                item.status == "missing_workspace" for item in items
            ),
            missing_credentials_tools=sum(
                item.status == "missing_credentials" for item in items
            ),
            permission_blocked_tools=sum(
                item.status == "permission_blocked" for item in items
            ),
            approval_gated_tools=sum(item.approval_gate.required for item in items),
        )
    return summaries


__all__ = [
    "AssistantToolApprovalGate",
    "AssistantToolCapability",
    "AssistantToolCapabilityReason",
    "AssistantToolCapabilityReasonCode",
    "AssistantToolGroupCapabilitySummary",
    "AssistantToolStatus",
    "assistant_tool_group",
    "build_tool_capabilities",
    "build_tool_group_capability_summaries",
]
