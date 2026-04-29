"""Common OpenAI-compatible provider endpoint URLs for the assistant app."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ProviderEndpoint:
    """One copyable provider endpoint entry."""

    name: str
    url: str


COMMON_OPENAI_COMPATIBLE_ENDPOINTS: tuple[ProviderEndpoint, ...] = (
    ProviderEndpoint(name="OpenAI", url="https://api.openai.com/v1"),
    ProviderEndpoint(name="xAI", url="https://api.x.ai/v1"),
    ProviderEndpoint(
        name="Gemini",
        url="https://generativelanguage.googleapis.com/v1beta/openai/",
    ),
    ProviderEndpoint(name="Claude", url="https://api.anthropic.com/v1/"),
)


__all__ = ["COMMON_OPENAI_COMPATIBLE_ENDPOINTS", "ProviderEndpoint"]
