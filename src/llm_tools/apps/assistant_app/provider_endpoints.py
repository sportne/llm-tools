"""Common provider connection presets for the assistant app."""

from __future__ import annotations

from dataclasses import dataclass

from llm_tools.apps.chat_config import ProviderAuthScheme, ProviderProtocol


@dataclass(frozen=True, slots=True)
class ProviderEndpoint:
    """One non-secret provider connection preset entry."""

    name: str
    url: str
    provider_protocol: ProviderProtocol = ProviderProtocol.OPENAI_API
    auth_scheme: ProviderAuthScheme = ProviderAuthScheme.BEARER


COMMON_OPENAI_COMPATIBLE_ENDPOINTS: tuple[ProviderEndpoint, ...] = (
    ProviderEndpoint(
        name="Local Ollama",
        url="http://127.0.0.1:11434/v1",
        auth_scheme=ProviderAuthScheme.NONE,
    ),
    ProviderEndpoint(name="OpenAI", url="https://api.openai.com/v1"),
    ProviderEndpoint(name="xAI", url="https://api.x.ai/v1"),
    ProviderEndpoint(
        name="Gemini",
        url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    ProviderEndpoint(name="Claude", url="https://api.anthropic.com/v1/"),
)

COMMON_NATIVE_PROVIDER_ENDPOINTS: tuple[ProviderEndpoint, ...] = (
    ProviderEndpoint(
        name="Local Ollama Native",
        url="http://127.0.0.1:11434",
        provider_protocol=ProviderProtocol.OLLAMA_NATIVE,
        auth_scheme=ProviderAuthScheme.NONE,
    ),
    ProviderEndpoint(
        name="Ask Sage Native",
        url="https://api.asksage.ai/server",
        provider_protocol=ProviderProtocol.ASK_SAGE_NATIVE,
        auth_scheme=ProviderAuthScheme.X_ACCESS_TOKENS,
    ),
)

COMMON_PROVIDER_ENDPOINTS = (
    *COMMON_NATIVE_PROVIDER_ENDPOINTS,
    *COMMON_OPENAI_COMPATIBLE_ENDPOINTS,
)


__all__ = [
    "COMMON_NATIVE_PROVIDER_ENDPOINTS",
    "COMMON_OPENAI_COMPATIBLE_ENDPOINTS",
    "COMMON_PROVIDER_ENDPOINTS",
    "ProviderEndpoint",
]
