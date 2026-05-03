"""Shared runtime helpers for assistant and harness app surfaces."""

from __future__ import annotations

from os import getenv
from typing import Any

from llm_tools.apps.chat_config import (
    ProviderAuthScheme,
    ProviderConnectionConfig,
    ProviderProtocol,
)
from llm_tools.llm_providers import (
    AskSageNativeProvider,
    OllamaNativeProvider,
    OpenAICompatibleProvider,
    ResponseModeStrategy,
)
from llm_tools.tool_api import SideEffectClass, ToolPolicy, ToolRegistry
from llm_tools.tool_api.redaction import RedactionConfig
from llm_tools.tools import (
    register_atlassian_tools,
    register_filesystem_tools,
    register_git_tools,
)
from llm_tools.workflow_api import ModelTurnProvider, WorkflowExecutor


def create_provider(
    *,
    provider_protocol: ProviderProtocol,
    provider_connection: ProviderConnectionConfig,
    api_key: str | None,
    selected_model: str,
    response_mode_strategy: ResponseModeStrategy | str,
    timeout_seconds: float,
    provider_request_settings: dict[str, Any] | None = None,
    allow_env_api_key: bool = True,
) -> ModelTurnProvider:
    """Create a provider client from execution-ready provider fields."""
    if not selected_model.strip():
        raise ValueError("Choose a model before running a model turn.")
    if not provider_connection.api_base_url:
        raise ValueError("Enter an API base URL before running a model turn.")
    request_params = {"timeout": timeout_seconds}
    if provider_protocol is ProviderProtocol.OLLAMA_NATIVE:
        if provider_connection.auth_scheme is not ProviderAuthScheme.NONE:
            raise ValueError("Native Ollama provider protocol uses auth scheme 'none'.")
        return OllamaNativeProvider(
            model=selected_model,
            host=provider_connection.api_base_url,
            response_mode_strategy=response_mode_strategy,
            default_request_params=request_params,
        )
    if provider_protocol is ProviderProtocol.ASK_SAGE_NATIVE:
        if provider_connection.auth_scheme is not ProviderAuthScheme.X_ACCESS_TOKENS:
            raise ValueError(
                "Native Ask Sage provider protocol uses auth scheme 'x_access_tokens'."
            )
        if not api_key:
            raise ValueError("Enter provider credentials before running a model turn.")
        return AskSageNativeProvider(
            model=selected_model,
            access_token=api_key,
            base_url=provider_connection.api_base_url,
            response_mode_strategy=response_mode_strategy,
            request_settings=provider_request_settings,
            default_request_params=request_params,
        )
    if provider_protocol is not ProviderProtocol.OPENAI_API:  # pragma: no cover
        raise ValueError(f"Unsupported provider protocol: {provider_protocol.value}")
    if provider_connection.auth_scheme is ProviderAuthScheme.X_ACCESS_TOKENS:
        raise ValueError(
            "OpenAI API provider protocol does not support x_access_tokens auth."
        )
    effective_api_key = api_key
    if effective_api_key is None and allow_env_api_key:
        effective_api_key = getenv("OPENAI_API_KEY")
    if (
        effective_api_key is None
        and provider_connection.auth_scheme is ProviderAuthScheme.NONE
    ):
        effective_api_key = "unused"
    return OpenAICompatibleProvider(
        model=selected_model,
        base_url=provider_connection.api_base_url,
        api_key=effective_api_key,
        response_mode_strategy=response_mode_strategy,
        default_request_params=request_params,
    )


def build_chat_registry() -> ToolRegistry:
    """Return the shared read-oriented registry used by app helpers."""
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_git_tools(registry)
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
