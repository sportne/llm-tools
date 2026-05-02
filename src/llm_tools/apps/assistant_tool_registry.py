"""Assistant registry assembly helpers."""

from __future__ import annotations

from llm_tools.apps.assistant_config import AssistantConfig
from llm_tools.tool_api import SideEffectClass, ToolPolicy, ToolRegistry, ToolSpec
from llm_tools.tool_api.redaction import RedactionConfig
from llm_tools.tools import (
    register_atlassian_tools,
    register_filesystem_tools,
    register_git_tools,
    register_gitlab_tools,
)
from llm_tools.workflow_api import WorkflowExecutor


def build_assistant_registry() -> ToolRegistry:
    """Return the full assistant-visible built-in registry."""
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_git_tools(registry)
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
    return {spec.name: spec for spec in registry.list_tools()}


def resolve_assistant_default_enabled_tools(
    config: AssistantConfig,
) -> set[str]:
    """Return the default tool set for a new assistant chat session."""
    configured = config.policy.enabled_tools
    if configured is None:
        return set()
    return set(configured).intersection(build_assistant_available_tool_specs())


__all__ = [
    "build_assistant_available_tool_specs",
    "build_assistant_executor",
    "build_assistant_registry",
    "resolve_assistant_default_enabled_tools",
]
