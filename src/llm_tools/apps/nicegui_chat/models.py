"""Typed state models for the NiceGUI chat app."""

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


class NiceGUITranscriptEntry(BaseModel):
    """One persisted transcript entry rendered in the NiceGUI chat canvas."""

    role: Literal["user", "assistant", "system", "error"]
    text: str
    final_response: ChatFinalResponse | None = None
    assistant_completion_state: Literal["complete", "interrupted"] = "complete"
    show_in_transcript: bool = True
    created_at: str | None = None


class NiceGUIInspectorEntry(BaseModel):
    """One persisted inspector/debug payload."""

    label: str
    payload: object
    created_at: str | None = None


class NiceGUIInspectorState(BaseModel):
    """Persisted inspector/debug state for one NiceGUI session."""

    provider_messages: list[NiceGUIInspectorEntry] = Field(default_factory=list)
    parsed_responses: list[NiceGUIInspectorEntry] = Field(default_factory=list)
    tool_executions: list[NiceGUIInspectorEntry] = Field(default_factory=list)


class NiceGUIWorkbenchItem(BaseModel):
    """Persisted workbench shell item for inspector or future artifacts."""

    item_id: str
    kind: Literal["inspector", "artifact", "result", "file", "tool"] = "inspector"
    title: str
    payload: object = Field(default_factory=dict)
    version: int = Field(default=1, ge=1)
    active: bool = False
    started_at: str | None = None
    finished_at: str | None = None
    duration_seconds: float | None = Field(default=None, ge=0.0)
    created_at: str
    updated_at: str


class NiceGUIRuntimeConfig(BaseModel):
    """Mutable runtime controls owned by one NiceGUI session."""

    provider: ProviderPreset = ProviderPreset.OLLAMA
    provider_mode_strategy: ProviderModeStrategy = ProviderModeStrategy.AUTO
    model_name: str = "gemma4:26b"
    api_base_url: str | None = "http://127.0.0.1:11434/v1"
    api_key_env_var: str | None = None
    temperature: float = 0.1
    timeout_seconds: float = 60.0
    root_path: str | None = None
    default_workspace_root: str | None = None
    enabled_tools: list[str] = Field(default_factory=list)
    tool_urls: dict[str, str] = Field(default_factory=dict)
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
        return cleaned or None

    @field_validator("api_key_env_var")
    @classmethod
    def validate_api_key_env_var(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("root_path", "default_workspace_root")
    @classmethod
    def validate_optional_paths(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None

    @field_validator("enabled_tools")
    @classmethod
    def validate_enabled_tools(cls, value: list[str]) -> list[str]:
        cleaned = [entry.strip() for entry in value]
        if any(not entry for entry in cleaned):
            raise ValueError("enabled_tools must not contain empty values")
        return cleaned

    @field_validator("tool_urls")
    @classmethod
    def validate_tool_urls(cls, value: dict[str, str]) -> dict[str, str]:
        return {
            key.strip(): url.strip()
            for key, url in value.items()
            if key.strip() and url.strip()
        }


class NiceGUISessionSummary(BaseModel):
    """Lightweight persisted summary for the chat rail."""

    session_id: str
    title: str = "New chat"
    created_at: str
    updated_at: str
    root_path: str | None = None
    provider: ProviderPreset = ProviderPreset.OLLAMA
    model_name: str = "gemma4:26b"
    message_count: int = 0
    temporary: bool = False
    project_id: str | None = None


class NiceGUISessionRecord(BaseModel):
    """Full NiceGUI session record."""

    summary: NiceGUISessionSummary
    runtime: NiceGUIRuntimeConfig
    transcript: list[NiceGUITranscriptEntry] = Field(default_factory=list)
    workflow_session_state: ChatSessionState = Field(default_factory=ChatSessionState)
    token_usage: ChatTokenUsage | None = None
    inspector_state: NiceGUIInspectorState = Field(
        default_factory=NiceGUIInspectorState
    )
    workbench_items: list[NiceGUIWorkbenchItem] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)


class NiceGUIPreferences(BaseModel):
    """Persisted app-wide UI preferences and recents."""

    theme_mode: Literal["dark", "light"] = "light"
    active_session_id: str | None = None
    sidebar_collapsed: bool = False
    workbench_open: bool = False
    settings_open: bool = False
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
    "NiceGUIInspectorEntry",
    "NiceGUIInspectorState",
    "NiceGUIPreferences",
    "NiceGUIRuntimeConfig",
    "NiceGUISessionRecord",
    "NiceGUISessionSummary",
    "NiceGUITranscriptEntry",
    "NiceGUIWorkbenchItem",
]
