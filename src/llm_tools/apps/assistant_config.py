"""Config models and loading helpers for the Streamlit assistant app."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]
from pydantic import BaseModel, Field, field_validator
from pydantic import ValidationError as PydanticValidationError

from llm_tools.apps.chat_config import ChatLLMConfig, ChatPolicyConfig, ChatUIConfig
from llm_tools.tools.filesystem import ToolLimits
from llm_tools.workflow_api import ChatSessionConfig


class AssistantWorkspaceConfig(BaseModel):
    """Optional local workspace defaults for assistant sessions."""

    default_root: str | None = None

    @field_validator("default_root")
    @classmethod
    def validate_default_root(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        return cleaned


class AssistantResearchConfig(BaseModel):
    """Assistant research-session defaults backed by the harness surface."""

    enabled: bool = True
    store_dir: str | None = None
    max_recent_sessions: int = Field(default=8, ge=1)
    default_max_turns: int = Field(default=6, ge=1)
    default_max_tool_invocations: int | None = Field(default=24, ge=1)
    default_max_elapsed_seconds: int | None = Field(default=None, ge=1)
    include_replay_by_default: bool = False

    @field_validator("store_dir")
    @classmethod
    def validate_store_dir(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None
        return cleaned


class StreamlitAssistantConfig(BaseModel):
    """Shared configuration for the Streamlit assistant app."""

    llm: ChatLLMConfig = Field(default_factory=ChatLLMConfig)
    session: ChatSessionConfig = Field(default_factory=ChatSessionConfig)
    tool_limits: ToolLimits = Field(default_factory=ToolLimits)
    policy: ChatPolicyConfig = Field(default_factory=ChatPolicyConfig)
    ui: ChatUIConfig = Field(default_factory=ChatUIConfig)
    workspace: AssistantWorkspaceConfig = Field(
        default_factory=AssistantWorkspaceConfig
    )
    research: AssistantResearchConfig = Field(default_factory=AssistantResearchConfig)


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


def load_streamlit_assistant_config(path: Path) -> StreamlitAssistantConfig:
    """Load and validate the Streamlit assistant configuration file."""
    raw = _load_yaml(path)
    for section_name in (
        "llm",
        "session",
        "tool_limits",
        "policy",
        "ui",
        "workspace",
        "research",
    ):
        section_value = raw.get(section_name)
        if section_value is not None and not isinstance(section_value, dict):
            raise ValueError(f"assistant config '{section_name}' must be a mapping")
    try:
        return StreamlitAssistantConfig.model_validate(raw)
    except PydanticValidationError as exc:
        raise ValueError(f"Invalid assistant config at {path}: {exc}") from exc


__all__ = [
    "AssistantResearchConfig",
    "AssistantWorkspaceConfig",
    "StreamlitAssistantConfig",
    "load_streamlit_assistant_config",
]
