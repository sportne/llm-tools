"""Shared runtime helpers for assistant and harness app surfaces."""

from __future__ import annotations

from os import getenv

from llm_tools.apps.chat_config import ChatLLMConfig
from llm_tools.llm_providers import OpenAICompatibleProvider, ProviderModeStrategy
from llm_tools.tool_api import SideEffectClass, ToolPolicy, ToolRegistry
from llm_tools.tool_api.redaction import RedactionConfig
from llm_tools.tools import (
    register_atlassian_tools,
    register_filesystem_tools,
    register_git_tools,
    register_text_tools,
)
from llm_tools.workflow_api import WorkflowExecutor


def create_provider(
    config: ChatLLMConfig,
    *,
    api_key: str | None,
    model_name: str,
    mode_strategy: ProviderModeStrategy = ProviderModeStrategy.AUTO,
) -> OpenAICompatibleProvider:
    """Create a provider client for the configured OpenAI-compatible backend."""
    request_params = {"timeout": config.timeout_seconds}
    if config.provider.value == "openai":
        return OpenAICompatibleProvider.for_openai(
            model=model_name,
            api_key=api_key or getenv(config.api_key_env_var or "OPENAI_API_KEY"),
            mode_strategy=mode_strategy,
            default_request_params=request_params,
        )
    if config.provider.value == "ollama":
        base_url = config.api_base_url or "http://127.0.0.1:11434/v1"
        return OpenAICompatibleProvider.for_ollama(
            model=model_name,
            base_url=base_url,
            api_key=api_key or "ollama",
            mode_strategy=mode_strategy,
            default_request_params=request_params,
        )
    if not config.api_base_url:
        raise ValueError("Custom OpenAI-compatible providers require api_base_url.")
    return OpenAICompatibleProvider(
        model=model_name,
        base_url=config.api_base_url,
        api_key=api_key or getenv(config.api_key_env_var or "OPENAI_API_KEY"),
        mode_strategy=mode_strategy,
        default_request_params=request_params,
    )


def build_chat_registry() -> ToolRegistry:
    """Return the shared read-oriented registry used by app helpers."""
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_git_tools(registry)
    register_text_tools(registry)
    register_atlassian_tools(registry)
    return registry


def build_chat_executor(
    *,
    policy: ToolPolicy | None = None,
    redaction_config: RedactionConfig | None = None,
) -> tuple[ToolRegistry, WorkflowExecutor]:
    """Return the shared registry and policy-aware executor for one turn."""
    registry = build_chat_registry()
    effective_policy = policy or ToolPolicy(
        allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=False,
        redaction=(
            redaction_config.model_copy(deep=True)
            if redaction_config is not None
            else RedactionConfig()
        ),
    )
    return registry, WorkflowExecutor(registry=registry, policy=effective_policy)


__all__ = ["build_chat_executor", "build_chat_registry", "create_provider"]
