"""Runtime helpers for the assistant-focused app surfaces."""

from __future__ import annotations

import json
import os
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel, Field

from llm_tools.apps.assistant_config import StreamlitAssistantConfig
from llm_tools.apps.assistant_prompts import build_research_system_prompt
from llm_tools.apps.chat_config import ChatLLMConfig
from llm_tools.apps.chat_runtime import create_provider
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.llm_providers import OpenAICompatibleProvider, ProviderModeStrategy
from llm_tools.tool_api import (
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

AssistantToolStatus = Literal[
    "available",
    "disabled",
    "missing_workspace",
    "missing_credentials",
    "permission_blocked",
]


class AssistantToolCapability(BaseModel):
    """One tool plus its assistant-facing availability state."""

    tool_name: str
    group: str
    enabled: bool = False
    exposed_to_model: bool = False
    status: AssistantToolStatus
    detail: str | None = None
    approval_required: bool = False
    side_effects: SideEffectClass
    requires_network: bool = False
    requires_filesystem: bool = False
    requires_subprocess: bool = False
    required_secrets: list[str] = Field(default_factory=list)


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
        status: AssistantToolStatus = "disabled"
        reasons: list[str] = []
        if enabled:
            if spec.requires_filesystem and root_path is None:
                reasons.append("Select a workspace root first.")
                status = "missing_workspace"
            missing_secrets = [
                secret for secret in spec.required_secrets if not env.get(secret)
            ]
            if missing_secrets:
                reasons.append(
                    "Missing credentials: " + ", ".join(sorted(missing_secrets))
                )
                if status == "disabled":
                    status = "missing_credentials"
            if (
                spec.requires_network
                and not allow_network
                or spec.requires_filesystem
                and not allow_filesystem
                or spec.requires_subprocess
                and not allow_subprocess
            ):
                reasons.append("Current session permissions do not allow this tool.")
                if status == "disabled":
                    status = "permission_blocked"
            if not reasons:
                status = "available"
        group_name = assistant_tool_group(spec)
        grouped[group_name].append(
            AssistantToolCapability(
                tool_name=tool_name,
                group=group_name,
                enabled=enabled,
                exposed_to_model=enabled and status == "available",
                status=status,
                detail=" ".join(reasons) if reasons else None,
                approval_required=spec.side_effects in require_approval_for,
                side_effects=spec.side_effects,
                requires_network=spec.requires_network,
                requires_filesystem=spec.requires_filesystem,
                requires_subprocess=spec.requires_subprocess,
                required_secrets=list(spec.required_secrets),
            )
        )
    return dict(sorted(grouped.items()))


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
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._system_prompt = system_prompt

    def run(
        self,
        *,
        state: object,
        selected_task_ids: Sequence[str],
        context: ToolContext,
        adapter: ActionEnvelopeAdapter,
        prepared_interaction: PreparedModelInteraction,
    ) -> ParsedModelResponse:
        del state
        return self._provider.run(
            adapter=adapter,
            messages=_build_research_messages(
                system_prompt=self._system_prompt,
                selected_task_ids=selected_task_ids,
                context=context,
            ),
            response_model=prepared_interaction.response_model,
            request_params={"temperature": self._temperature},
        )

    async def run_async(
        self,
        *,
        state: object,
        selected_task_ids: Sequence[str],
        context: ToolContext,
        adapter: ActionEnvelopeAdapter,
        prepared_interaction: PreparedModelInteraction,
    ) -> ParsedModelResponse:
        del state
        return await self._provider.run_async(
            adapter=adapter,
            messages=_build_research_messages(
                system_prompt=self._system_prompt,
                selected_task_ids=selected_task_ids,
                context=context,
            ),
            response_model=prepared_interaction.response_model,
            request_params={"temperature": self._temperature},
        )


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
) -> AssistantHarnessTurnProvider:
    """Create the live provider wrapper used by research sessions."""
    provider = create_provider(
        provider_config,
        api_key=api_key,
        model_name=model_name,
        mode_strategy=mode_strategy,
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
    )


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
    "AssistantToolCapability",
    "assistant_tool_group",
    "build_assistant_available_tool_specs",
    "build_assistant_context",
    "build_assistant_executor",
    "build_assistant_policy",
    "build_assistant_registry",
    "build_live_harness_provider",
    "build_tool_capabilities",
    "resolve_assistant_default_enabled_tools",
]
