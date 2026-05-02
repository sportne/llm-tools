"""Common provider connection presets for the assistant app."""

from __future__ import annotations

from dataclasses import dataclass

from llm_tools.apps.chat_config import ProviderProtocol


@dataclass(frozen=True, slots=True)
class ProviderEndpoint:
    """One non-secret provider connection preset entry."""

    name: str
    url: str
    provider_protocol: ProviderProtocol = ProviderProtocol.OPENAI_API
    requires_bearer_token: bool = True


COMMON_OPENAI_COMPATIBLE_ENDPOINTS: tuple[ProviderEndpoint, ...] = (
    ProviderEndpoint(
        name="Local Ollama",
        url="http://127.0.0.1:11434/v1",
        requires_bearer_token=False,
    ),
    ProviderEndpoint(name="OpenAI", url="https://api.openai.com/v1"),
    ProviderEndpoint(name="xAI", url="https://api.x.ai/v1"),
    ProviderEndpoint(
        name="Gemini",
        url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    ProviderEndpoint(name="Claude", url="https://api.anthropic.com/v1/"),
)


__all__ = ["COMMON_OPENAI_COMPATIBLE_ENDPOINTS", "ProviderEndpoint"]
