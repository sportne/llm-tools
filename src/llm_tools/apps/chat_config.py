"""Shared config models and loading helpers for repository chat app layers."""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationError as PydanticValidationError

from llm_tools.tool_api import SideEffectClass
from llm_tools.tool_api.redaction import RedactionConfig
from llm_tools.tools.filesystem import SourceFilters, ToolLimits
from llm_tools.workflow_api import ChatSessionConfig


class ProviderPreset(str, Enum):  # noqa: UP042
    """OpenAI-compatible provider presets available in repository chat apps."""

    OPENAI = "openai"
    OLLAMA = "ollama"
    CUSTOM_OPENAI_COMPATIBLE = "custom_openai_compatible"


class ChatCredentialPromptMetadata(BaseModel):
    """UI-safe credential prompt policy without any persisted secret value."""

    api_key_env_var: str = "OPENAI_API_KEY"
    prompt_for_api_key_if_missing: bool = True
    expects_api_key: bool = False
    secret_kind: str = "api_key"  # noqa: S105
    mask_input: bool = True
    allow_empty_secret: bool = True
    persist_secret: bool = False


class ChatLLMConfig(BaseModel):
    """Standalone chat runtime configuration."""

    provider: ProviderPreset = ProviderPreset.OLLAMA
    model_name: str = "gemma4:26b"
    temperature: float = 0.1
    api_base_url: str | None = "http://127.0.0.1:11434/v1"
    timeout_seconds: float = 60.0
    api_key_env_var: str | None = None
    prompt_for_api_key_if_missing: bool = True

    @field_validator("model_name")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("chat LLM string settings must not be empty")
        return cleaned

    @field_validator("api_base_url")
    @classmethod
    def validate_optional_api_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("api_base_url must not be empty when provided")
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
        env_var = self.api_key_env_var
        expects_api_key = self.provider is not ProviderPreset.OLLAMA
        if env_var is None and self.provider is ProviderPreset.OPENAI:
            env_var = "OPENAI_API_KEY"
        if env_var is None and self.provider is ProviderPreset.CUSTOM_OPENAI_COMPATIBLE:
            env_var = "OPENAI_API_KEY"
        return ChatCredentialPromptMetadata(
            api_key_env_var=env_var or "API key",
            prompt_for_api_key_if_missing=self.prompt_for_api_key_if_missing,
            expects_api_key=expects_api_key,
        )


class ChatUIConfig(BaseModel):
    """Minimal UI toggles for the interactive chat apps."""

    show_token_usage: bool = True
    show_footer_help: bool = True
    inspector_open_by_default: bool = False


class ChatPolicyConfig(BaseModel):
    """Startup policy defaults for the session-scoped chat controls."""

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


class TextualChatConfig(BaseModel):
    """Shared repository-chat configuration used by app frontends."""

    llm: ChatLLMConfig = Field(default_factory=ChatLLMConfig)
    source_filters: SourceFilters = Field(default_factory=SourceFilters)
    session: ChatSessionConfig = Field(default_factory=ChatSessionConfig)
    tool_limits: ToolLimits = Field(default_factory=ToolLimits)
    policy: ChatPolicyConfig = Field(default_factory=ChatPolicyConfig)
    ui: ChatUIConfig = Field(default_factory=ChatUIConfig)


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ValueError(f"Configuration file not found: {path}")
    if not path.is_file():
        raise ValueError(f"Configuration path is not a file: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ValueError(f"Invalid YAML at {path}: {exc}") from exc
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"Expected mapping at root of YAML file: {path}")
    return raw


def load_textual_chat_config(path: Path) -> TextualChatConfig:
    """Load and validate the shared repository chat configuration file."""
    raw = _load_yaml(path)
    for section_name in (
        "llm",
        "source_filters",
        "session",
        "tool_limits",
        "policy",
        "ui",
    ):
        section_value = raw.get(section_name)
        if section_value is not None and not isinstance(section_value, dict):
            raise ValueError(f"chat config '{section_name}' must be a mapping")
    try:
        return TextualChatConfig.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValueError(f"Invalid chat config at {path}: {exc}") from exc
