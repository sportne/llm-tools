"""Provider connection preset helpers for the assistant app."""

from __future__ import annotations

from llm_tools.apps.assistant_app.models import ProviderConnectionPreset
from llm_tools.apps.assistant_app.project_defaults import PROJECT_DEFAULTS
from llm_tools.apps.chat_config import ProviderProtocol

ProviderEndpoint = ProviderConnectionPreset

COMMON_PROVIDER_ENDPOINTS: tuple[ProviderConnectionPreset, ...] = (
    PROJECT_DEFAULTS.provider_connection_presets
)
COMMON_OPENAI_COMPATIBLE_ENDPOINTS: tuple[ProviderConnectionPreset, ...] = tuple(
    preset
    for preset in COMMON_PROVIDER_ENDPOINTS
    if preset.provider_protocol is ProviderProtocol.OPENAI_API
)
COMMON_NATIVE_PROVIDER_ENDPOINTS: tuple[ProviderConnectionPreset, ...] = tuple(
    preset
    for preset in COMMON_PROVIDER_ENDPOINTS
    if preset.provider_protocol is not ProviderProtocol.OPENAI_API
)


__all__ = [
    "COMMON_NATIVE_PROVIDER_ENDPOINTS",
    "COMMON_OPENAI_COMPATIBLE_ENDPOINTS",
    "COMMON_PROVIDER_ENDPOINTS",
    "ProviderEndpoint",
]
