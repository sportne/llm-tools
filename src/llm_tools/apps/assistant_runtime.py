"""Runtime helpers for the assistant-focused app surfaces."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Mapping, Sequence
from enum import Enum
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from llm_tools.apps.assistant_config import StreamlitAssistantConfig
from llm_tools.apps.assistant_prompts import build_research_system_prompt
from llm_tools.apps.chat_config import ChatLLMConfig
from llm_tools.apps.chat_runtime import create_provider
from llm_tools.apps.protection_runtime import (
    build_protection_controller,
    build_protection_environment,
)
from llm_tools.harness_api.models import HarnessState
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.llm_providers import OpenAICompatibleProvider, ProviderModeStrategy
from llm_tools.tool_api import (
    ProtectionProvenanceSnapshot,
    SideEffectClass,
    ToolContext,
    ToolPolicy,
    ToolRegistry,
    ToolSpec,
)
from llm_tools.tool_api.redaction import RedactionConfig
from llm_tools.tools import (
    register_atlassian_tools,
    register_filesystem_tools,
    register_git_tools,
    register_gitlab_tools,
    register_text_tools,
)
from llm_tools.workflow_api import WorkflowExecutor
from llm_tools.workflow_api.executor import PreparedModelInteraction
from llm_tools.workflow_api.protection import (
    ProtectionController,
    collect_provenance_from_tool_results,
)

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


def build_assistant_registry() -> ToolRegistry:
    """Return the full assistant-visible built-in registry."""
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_git_tools(registry)
    register_text_tools(registry)
    register_atlassian_tools(registry)
    register_gitlab_tools(registry)
    return registry


def build_assistant_executor(
    *,
    policy: ToolPolicy | None = None,
    redaction_config: RedactionConfig | None = None,
) -> tuple[ToolRegistry, WorkflowExecutor]:
    """Return the assistant registry and a workflow executor for one turn."""
    registry = build_assistant_registry()
    effective_policy = policy or ToolPolicy(
        allowed_tools=set(),
        allowed_side_effects={SideEffectClass.NONE},
        require_approval_for={
            SideEffectClass.LOCAL_WRITE,
            SideEffectClass.EXTERNAL_WRITE,
        },
        allow_network=False,
        allow_filesystem=False,
        allow_subprocess=False,
        redaction=(
            redaction_config.model_copy(deep=True)
            if redaction_config is not None
            else RedactionConfig()
        ),
    )
    return registry, WorkflowExecutor(registry=registry, policy=effective_policy)


def build_assistant_available_tool_specs() -> dict[str, ToolSpec]:
    """Return assistant-visible tool specs keyed by tool name."""
    registry = build_assistant_registry()
    return {tool.spec.name: tool.spec for tool in registry.list_registered_tools()}


def resolve_assistant_default_enabled_tools(
    config: StreamlitAssistantConfig,
) -> set[str]:
    """Return the default tool set for a new assistant chat session."""
    configured = config.policy.enabled_tools
    if configured is None:
        return set()
    return set(configured).intersection(build_assistant_available_tool_specs())


def assistant_tool_group(spec: ToolSpec) -> str:
    """Return the assistant UI group for one tool spec."""
    tags = set(spec.tags)
    if "gitlab" in tags:
        return "GitLab"
    if "atlassian" in tags or "jira" in tags or "confluence" in tags:
        return "Atlassian"
    if "git" in tags:
        return "Git"
    if "filesystem" in tags:
        return "Local Files"
    if "text" in tags:
        return "Text"
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


def build_assistant_policy(
    *,
    enabled_tools: set[str],
    tool_specs: dict[str, ToolSpec],
    require_approval_for: set[SideEffectClass],
    allow_network: bool,
    allow_filesystem: bool,
    allow_subprocess: bool,
    redaction_config: RedactionConfig,
) -> ToolPolicy:
    """Build the assistant workflow policy for one session."""
    allowed_side_effects = {SideEffectClass.NONE}
    for tool_name in enabled_tools:
        spec = tool_specs.get(tool_name)
        if spec is not None:
            allowed_side_effects.add(spec.side_effects)
    return ToolPolicy(
        allowed_tools=set(enabled_tools),
        allowed_side_effects=allowed_side_effects,
        require_approval_for=set(require_approval_for),
        allow_network=allow_network,
        allow_filesystem=allow_filesystem,
        allow_subprocess=allow_subprocess,
        redaction=redaction_config.model_copy(deep=True),
    )


def build_assistant_context(
    *,
    root_path: Path | None,
    config: StreamlitAssistantConfig,
    app_name: str,
) -> ToolContext:
    """Build the tool context passed into assistant workflow execution."""
    effective_read_limit = (
        config.tool_limits.max_read_file_chars
        if config.tool_limits.max_read_file_chars is not None
        else max(1, config.session.max_context_tokens * 4)
    )
    effective_tool_limits = config.tool_limits.model_copy(
        update={"max_read_file_chars": effective_read_limit}
    )
    return ToolContext(
        invocation_id=f"{app_name}-{uuid4()}",
        workspace=str(root_path) if root_path is not None else None,
        env=dict(os.environ),
        metadata={
            "tool_limits": effective_tool_limits.model_dump(mode="json"),
            "assistant_mode": "streamlit_assistant",
        },
    )


class AssistantHarnessTurnProvider:
    """Live harness provider wrapper backed by the OpenAI-compatible client."""

    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        temperature: float,
        system_prompt: str,
        protection_controller: ProtectionController | None = None,
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._protection_controller = protection_controller

    def run(
        self,
        *,
        state: object,
        selected_task_ids: Sequence[str],
        context: ToolContext,
        adapter: ActionEnvelopeAdapter,
        prepared_interaction: PreparedModelInteraction,
    ) -> ParsedModelResponse:
        provenance = _collect_harness_provenance(state)
        messages = _build_research_messages(
            system_prompt=self._system_prompt,
            selected_task_ids=selected_task_ids,
            context=context,
        )
        if self._protection_controller is not None:
            prompt_decision = self._protection_controller.assess_prompt(
                messages=messages,
                provenance=provenance,
            )
            if (
                prompt_decision.action.value == "constrain"
                and prompt_decision.guard_text
            ):
                messages = [
                    {"role": "system", "content": prompt_decision.guard_text},
                    *messages,
                ]
            elif prompt_decision.action.value in {"challenge", "block"}:
                safe_message = (
                    prompt_decision.challenge_message
                    or "The request was withheld for this environment."
                )
                context.metadata["protection_review"] = {
                    "purge_requested": True,
                    "safe_message": safe_message,
                }
                return ParsedModelResponse(final_response=safe_message)
        parsed = self._provider.run(
            adapter=adapter,
            messages=messages,
            response_model=prepared_interaction.response_model,
            request_params={"temperature": self._temperature},
        )
        if self._protection_controller is None or parsed.final_response is None:
            return parsed
        response_decision = self._protection_controller.review_response(
            response_payload=parsed.final_response,
            provenance=provenance,
        )
        if (
            response_decision.action.value == "sanitize"
            and response_decision.sanitized_payload is not None
        ):
            context.metadata["protection_review"] = {
                "purge_requested": response_decision.should_purge,
                "safe_message": response_decision.sanitized_payload,
            }
            return parsed.model_copy(
                update={"final_response": response_decision.sanitized_payload}
            )
        if response_decision.action.value == "block":
            safe_message = (
                response_decision.safe_message
                or "The response was withheld for this environment."
            )
            context.metadata["protection_review"] = {
                "purge_requested": response_decision.should_purge,
                "safe_message": safe_message,
            }
            return parsed.model_copy(update={"final_response": safe_message})
        return parsed

    async def run_async(
        self,
        *,
        state: object,
        selected_task_ids: Sequence[str],
        context: ToolContext,
        adapter: ActionEnvelopeAdapter,
        prepared_interaction: PreparedModelInteraction,
    ) -> ParsedModelResponse:
        provenance = _collect_harness_provenance(state)
        messages = _build_research_messages(
            system_prompt=self._system_prompt,
            selected_task_ids=selected_task_ids,
            context=context,
        )
        if self._protection_controller is not None:
            prompt_decision = await self._protection_controller.assess_prompt_async(
                messages=messages,
                provenance=provenance,
            )
            if (
                prompt_decision.action.value == "constrain"
                and prompt_decision.guard_text
            ):
                messages = [
                    {"role": "system", "content": prompt_decision.guard_text},
                    *messages,
                ]
            elif prompt_decision.action.value in {"challenge", "block"}:
                safe_message = (
                    prompt_decision.challenge_message
                    or "The request was withheld for this environment."
                )
                context.metadata["protection_review"] = {
                    "purge_requested": True,
                    "safe_message": safe_message,
                }
                return ParsedModelResponse(final_response=safe_message)
        parsed = await self._provider.run_async(
            adapter=adapter,
            messages=messages,
            response_model=prepared_interaction.response_model,
            request_params={"temperature": self._temperature},
        )
        if self._protection_controller is None or parsed.final_response is None:
            return parsed
        response_decision = await self._protection_controller.review_response_async(
            response_payload=parsed.final_response,
            provenance=provenance,
        )
        if (
            response_decision.action.value == "sanitize"
            and response_decision.sanitized_payload is not None
        ):
            context.metadata["protection_review"] = {
                "purge_requested": response_decision.should_purge,
                "safe_message": response_decision.sanitized_payload,
            }
            return parsed.model_copy(
                update={"final_response": response_decision.sanitized_payload}
            )
        if response_decision.action.value == "block":
            safe_message = (
                response_decision.safe_message
                or "The response was withheld for this environment."
            )
            context.metadata["protection_review"] = {
                "purge_requested": response_decision.should_purge,
                "safe_message": safe_message,
            }
            return parsed.model_copy(update={"final_response": safe_message})
        return parsed


def build_live_harness_provider(
    *,
    config: StreamlitAssistantConfig,
    provider_config: ChatLLMConfig,
    model_name: str,
    api_key: str | None,
    mode_strategy: ProviderModeStrategy,
    tool_registry: ToolRegistry,
    enabled_tool_names: set[str],
    workspace_enabled: bool,
    workspace: str | None,
    allow_network: bool,
    allow_filesystem: bool,
    allow_subprocess: bool,
) -> AssistantHarnessTurnProvider:
    """Create the live provider wrapper used by research sessions."""
    provider = create_provider(
        provider_config,
        api_key=api_key,
        model_name=model_name,
        mode_strategy=mode_strategy,
    )
    protection_controller = build_protection_controller(
        config=config.protection,
        provider=provider,
        environment=build_protection_environment(
            app_name="streamlit_assistant_research",
            model_name=model_name,
            workspace=workspace,
            enabled_tools=enabled_tool_names,
            allow_network=allow_network,
            allow_filesystem=allow_filesystem,
            allow_subprocess=allow_subprocess,
        ),
    )
    return AssistantHarnessTurnProvider(
        provider=provider,
        temperature=config.llm.temperature,
        system_prompt=build_research_system_prompt(
            tool_registry=tool_registry,
            tool_limits=config.tool_limits,
            enabled_tool_names=enabled_tool_names,
            workspace_enabled=workspace_enabled,
        ),
        protection_controller=protection_controller,
    )


def _collect_harness_provenance(state: object) -> ProtectionProvenanceSnapshot:
    if not isinstance(state, HarnessState):
        return ProtectionProvenanceSnapshot()

    tool_results = []
    for turn in state.turns:
        workflow_result = turn.workflow_result
        if workflow_result is None:
            continue
        for outcome in workflow_result.outcomes:
            if outcome.tool_result is not None:
                tool_results.append(outcome.tool_result)
    return collect_provenance_from_tool_results(tool_results)


def _build_research_messages(
    *,
    system_prompt: str,
    selected_task_ids: Sequence[str],
    context: ToolContext,
) -> list[dict[str, str]]:
    projection = context.metadata.get("harness_turn_context", {})
    user_payload = {
        "selected_task_ids": list(selected_task_ids),
        "turn_context": projection,
    }
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(user_payload, indent=2, sort_keys=True, default=str),
        },
    ]


__all__ = [
    "AssistantHarnessTurnProvider",
    "AssistantToolApprovalGate",
    "AssistantToolCapability",
    "AssistantToolCapabilityReason",
    "AssistantToolCapabilityReasonCode",
    "AssistantToolGroupCapabilitySummary",
    "assistant_tool_group",
    "build_assistant_available_tool_specs",
    "build_assistant_context",
    "build_assistant_executor",
    "build_assistant_policy",
    "build_assistant_registry",
    "build_live_harness_provider",
    "build_tool_capabilities",
    "build_tool_group_capability_summaries",
    "resolve_assistant_default_enabled_tools",
]
