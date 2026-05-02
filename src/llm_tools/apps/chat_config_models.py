"""Shared provider and policy config models for app surfaces."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, field_validator

from llm_tools.llm_providers import ResponseModeStrategy
from llm_tools.tool_api import SideEffectClass
from llm_tools.tool_api.redaction import RedactionConfig


class ProviderProtocol(str, Enum):  # noqa: UP042
    """Model-service API protocols available in app surfaces."""

    OPENAI_API = "openai_api"


class ProviderConnectionConfig(BaseModel):
    """Non-secret endpoint and auth settings for one provider connection."""

    api_base_url: str | None = None
    requires_bearer_token: bool = True

    @field_validator("api_base_url")
    @classmethod
    def validate_api_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().rstrip("/")
        return cleaned or None


class ChatCredentialPromptMetadata(BaseModel):
    """UI-safe credential prompt policy without any persisted secret value."""

    prompt_for_api_key_if_missing: bool = True
    expects_api_key: bool = False
    secret_kind: str = "api_key"  # noqa: S105
    mask_input: bool = True
    allow_empty_secret: bool = True
    persist_secret: bool = False


class ChatLLMConfig(BaseModel):
    """Standalone chat runtime configuration."""

    provider_protocol: ProviderProtocol = ProviderProtocol.OPENAI_API
    provider_connection: ProviderConnectionConfig = Field(
        default_factory=ProviderConnectionConfig
    )
    response_mode_strategy: ResponseModeStrategy = ResponseModeStrategy.AUTO
    selected_model: str | None = None
    temperature: float = 0.1
    timeout_seconds: float = 60.0
    prompt_for_api_key_if_missing: bool = True

    @field_validator("selected_model")
    @classmethod
    def validate_selected_model(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        return cleaned

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if value < 0.0 or value > 1.0:
            raise ValueError("temperature must be between 0.0 and 1.0")
        return value

    @field_validator("timeout_seconds")
    @classmethod
    def validate_timeout_seconds(cls, value: float) -> float:
        if value <= 0:
            raise ValueError("timeout_seconds must be greater than 0")
        return value

    def credential_prompt_metadata(self) -> ChatCredentialPromptMetadata:
        """Return UI-safe credential-prompt metadata derived from config."""
        expects_api_key = self.provider_connection.requires_bearer_token
        return ChatCredentialPromptMetadata(
            prompt_for_api_key_if_missing=self.prompt_for_api_key_if_missing,
            expects_api_key=expects_api_key,
        )


class ChatUIConfig(BaseModel):
    """Minimal UI toggles for the interactive app surfaces."""

    show_token_usage: bool = True
    show_footer_help: bool = True
    inspector_open_by_default: bool = False


class ChatPolicyConfig(BaseModel):
    """Startup policy defaults for session-scoped app controls."""

    enabled_tools: list[str] | None = None
    require_approval_for: set[SideEffectClass] = Field(default_factory=set)
    redaction: RedactionConfig = Field(default_factory=RedactionConfig)

    @field_validator("enabled_tools")
    @classmethod
    def validate_enabled_tools(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None
        cleaned = [entry.strip() for entry in value]
        if any(not entry for entry in cleaned):
            raise ValueError("enabled_tools must not contain empty values")
        return cleaned


__all__ = [
    "ChatCredentialPromptMetadata",
    "ChatLLMConfig",
    "ChatPolicyConfig",
    "ChatUIConfig",
    "ProviderConnectionConfig",
    "ProviderProtocol",
]
