"""Runtime helpers for the assistant-focused app surfaces."""

from llm_tools.apps.assistant_config import StreamlitAssistantConfig
from llm_tools.apps.assistant_execution import (
    build_assistant_context,
    build_assistant_policy,
)
from llm_tools.apps.assistant_prompts import build_research_system_prompt
from llm_tools.apps.assistant_research_provider import AssistantHarnessTurnProvider
from llm_tools.apps.assistant_tool_capabilities import (
    AssistantToolApprovalGate,
    AssistantToolCapability,
    AssistantToolCapabilityReason,
    AssistantToolCapabilityReasonCode,
    AssistantToolGroupCapabilitySummary,
    AssistantToolStatus,
    assistant_tool_group,
    build_tool_capabilities,
    build_tool_group_capability_summaries,
)
from llm_tools.apps.assistant_tool_registry import (
    build_assistant_available_tool_specs,
    build_assistant_executor,
    build_assistant_registry,
    resolve_assistant_default_enabled_tools,
)
from llm_tools.apps.chat_config import ChatLLMConfig
from llm_tools.apps.chat_runtime import create_provider
from llm_tools.apps.protection_runtime import (
    build_protection_controller,
    build_protection_environment,
)
from llm_tools.llm_providers import ProviderModeStrategy
from llm_tools.tool_api import ToolRegistry


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


__all__ = [
    "AssistantHarnessTurnProvider",
    "AssistantToolApprovalGate",
    "AssistantToolCapability",
    "AssistantToolCapabilityReason",
    "AssistantToolCapabilityReasonCode",
    "AssistantToolGroupCapabilitySummary",
    "AssistantToolStatus",
    "assistant_tool_group",
    "build_assistant_available_tool_specs",
    "build_assistant_context",
    "build_assistant_executor",
    "build_assistant_policy",
    "build_assistant_registry",
    "build_live_harness_provider",
    "build_protection_controller",
    "build_protection_environment",
    "build_research_system_prompt",
    "build_tool_capabilities",
    "build_tool_group_capability_summaries",
    "create_provider",
    "resolve_assistant_default_enabled_tools",
]
