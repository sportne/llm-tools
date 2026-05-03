"""Project-shipped defaults for the assistant app."""

from __future__ import annotations

from llm_tools.apps.assistant_app.models import (
    AssistantProjectDefaults,
    ProviderConnectionPreset,
)
from llm_tools.apps.chat_config import (
    ProviderAuthScheme,
    ProviderConnectionConfig,
    ProviderProtocol,
)

PROJECT_DEFAULTS = AssistantProjectDefaults(
    provider_connection_presets=(
        ProviderConnectionPreset(
            id="local-ollama-native",
            label="Local Ollama Native",
            provider_protocol=ProviderProtocol.OLLAMA_NATIVE,
            provider_connection=ProviderConnectionConfig(
                api_base_url="http://127.0.0.1:11434",
                auth_scheme=ProviderAuthScheme.NONE,
            ),
        ),
        ProviderConnectionPreset(
            id="ask-sage-native",
            label="Ask Sage Native",
            provider_protocol=ProviderProtocol.ASK_SAGE_NATIVE,
            provider_connection=ProviderConnectionConfig(
                api_base_url="https://api.asksage.ai/server",
                auth_scheme=ProviderAuthScheme.X_ACCESS_TOKENS,
            ),
        ),
        ProviderConnectionPreset(
            id="local-ollama-openai-api",
            label="Local Ollama",
            provider_protocol=ProviderProtocol.OPENAI_API,
            provider_connection=ProviderConnectionConfig(
                api_base_url="http://127.0.0.1:11434/v1",
                auth_scheme=ProviderAuthScheme.NONE,
            ),
        ),
        ProviderConnectionPreset(
            id="openai",
            label="OpenAI",
            provider_protocol=ProviderProtocol.OPENAI_API,
            provider_connection=ProviderConnectionConfig(
                api_base_url="https://api.openai.com/v1",
            ),
        ),
        ProviderConnectionPreset(
            id="xai",
            label="xAI",
            provider_protocol=ProviderProtocol.OPENAI_API,
            provider_connection=ProviderConnectionConfig(
                api_base_url="https://api.x.ai/v1",
            ),
        ),
        ProviderConnectionPreset(
            id="gemini-openai-api",
            label="Gemini",
            provider_protocol=ProviderProtocol.OPENAI_API,
            provider_connection=ProviderConnectionConfig(
                api_base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            ),
        ),
        ProviderConnectionPreset(
            id="claude-openai-api",
            label="Claude",
            provider_protocol=ProviderProtocol.OPENAI_API,
            provider_connection=ProviderConnectionConfig(
                api_base_url="https://api.anthropic.com/v1",
            ),
        ),
    )
)


__all__ = ["PROJECT_DEFAULTS"]
