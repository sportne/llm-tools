"""Shared persisted models for Streamlit-based app surfaces."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

from llm_tools.apps.assistant_config import AssistantResearchConfig
from llm_tools.apps.chat_config import ProviderPreset
from llm_tools.llm_providers import ProviderModeStrategy
from llm_tools.tool_api import SideEffectClass
from llm_tools.tools.filesystem import ToolLimits
from llm_tools.workflow_api import (
    ChatFinalResponse,
    ChatSessionConfig,
    ChatSessionState,
    ChatTokenUsage,
    ProtectionConfig,
)


class StreamlitTranscriptEntry(BaseModel):
    """One rendered transcript entry persisted with a Streamlit session."""

    role: Literal["user", "assistant", "system", "error"]
    text: str
    final_response: ChatFinalResponse | None = None
    assistant_completion_state: Literal["complete", "interrupted"] = "complete"
    show_in_transcript: bool = True


class StreamlitInspectorEntry(BaseModel):
    """One persisted inspector/debug payload."""

    label: str
    payload: object


class StreamlitInspectorState(BaseModel):
    """Persisted inspector/debug state for one Streamlit session."""

    provider_messages: list[StreamlitInspectorEntry] = Field(default_factory=list)
    parsed_responses: list[StreamlitInspectorEntry] = Field(default_factory=list)
    tool_executions: list[StreamlitInspectorEntry] = Field(default_factory=list)


class StreamlitRuntimeConfig(BaseModel):
    """Mutable runtime controls owned by one Streamlit session."""

    provider: ProviderPreset = ProviderPreset.OLLAMA
    provider_mode_strategy: ProviderModeStrategy = ProviderModeStrategy.AUTO
    model_name: str = "gemma4:26b"
    api_base_url: str | None = "http://127.0.0.1:11434/v1"
    temperature: float = 0.1
    timeout_seconds: float = 60.0
    root_path: str | None = None
    default_workspace_root: str | None = None
    enabled_tools: list[str] = Field(default_factory=list)
    require_approval_for: set[SideEffectClass] = Field(default_factory=set)
    allow_network: bool = True
    allow_filesystem: bool = True
    allow_subprocess: bool = True
    inspector_open: bool = False
    show_token_usage: bool = True
    show_footer_help: bool = True
    session_config: ChatSessionConfig = Field(default_factory=ChatSessionConfig)
    tool_limits: ToolLimits = Field(default_factory=ToolLimits)
    research: AssistantResearchConfig = Field(default_factory=AssistantResearchConfig)
    protection: ProtectionConfig = Field(default_factory=ProtectionConfig)

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("model_name must not be empty")
        return cleaned

    @field_validator("api_base_url")
    @classmethod
    def validate_api_base_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        return cleaned

    @field_validator("root_path")
    @classmethod
    def validate_root_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        return cleaned

    @field_validator("default_workspace_root")
    @classmethod
    def validate_default_workspace_root(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        return cleaned

    @field_validator("enabled_tools")
    @classmethod
    def validate_enabled_tools(cls, value: list[str]) -> list[str]:
        cleaned = [entry.strip() for entry in value]
        if any(not entry for entry in cleaned):
            raise ValueError("enabled_tools must not contain empty values")
        return cleaned


class StreamlitSessionSummary(BaseModel):
    """Lightweight persisted summary for the session rail."""

    session_id: str
    title: str = "New chat"
    created_at: str
    updated_at: str
    root_path: str | None = None
    provider: ProviderPreset = ProviderPreset.OLLAMA
    model_name: str = "gemma4:26b"
    message_count: int = 0


class StreamlitPersistedSessionRecord(BaseModel):
    """Full persisted session payload stored on disk."""

    summary: StreamlitSessionSummary
    runtime: StreamlitRuntimeConfig
    transcript: list[StreamlitTranscriptEntry] = Field(default_factory=list)
    workflow_session_state: ChatSessionState = Field(default_factory=ChatSessionState)
    token_usage: ChatTokenUsage | None = None
    inspector_state: StreamlitInspectorState = Field(
        default_factory=StreamlitInspectorState
    )
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class StreamlitSessionIndex(BaseModel):
    """Top-level persisted session index for a Streamlit app."""

    active_session_id: str | None = None
    session_order: list[str] = Field(default_factory=list)
    summaries: list[StreamlitSessionSummary] = Field(default_factory=list)


class StreamlitPreferences(BaseModel):
    """Persisted app-wide UI preferences and recents."""

    theme_mode: Literal["dark", "light"] = "dark"
    appearance_mode_explicit: bool = False
    settings_panel_open: bool = True
    recent_roots: list[str] = Field(default_factory=list)
    recent_models: dict[str, list[str]] = Field(default_factory=dict)
    recent_base_urls: dict[str, list[str]] = Field(default_factory=dict)

    @field_validator("recent_roots")
    @classmethod
    def validate_recent_roots(cls, value: list[str]) -> list[str]:
        return [entry.strip() for entry in value if entry.strip()]

    @field_validator("recent_models", "recent_base_urls")
    @classmethod
    def validate_mapping_lists(
        cls, value: dict[str, list[str]]
    ) -> dict[str, list[str]]:
        normalized: dict[str, list[str]] = {}
        for key, entries in value.items():
            cleaned_key = key.strip()
            if not cleaned_key:
                continue
            cleaned_entries = [entry.strip() for entry in entries if entry.strip()]
            if cleaned_entries:
                normalized[cleaned_key] = cleaned_entries
        return normalized


__all__ = [
    "StreamlitInspectorEntry",
    "StreamlitInspectorState",
    "StreamlitPersistedSessionRecord",
    "StreamlitPreferences",
    "StreamlitRuntimeConfig",
    "StreamlitSessionIndex",
    "StreamlitSessionSummary",
    "StreamlitTranscriptEntry",
]
