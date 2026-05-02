"""Config models and loading helpers for the LLM Tools Assistant app."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator

from llm_tools.apps.chat_config import ChatLLMConfig, ChatPolicyConfig, ChatUIConfig
from llm_tools.tools.filesystem import ToolLimits
from llm_tools.workflow_api import ChatSessionConfig
from llm_tools.workflow_api.protection import ProtectionConfig


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


class AssistantConfig(BaseModel):
    """Shared configuration for the LLM Tools Assistant app."""

    llm: ChatLLMConfig = Field(default_factory=ChatLLMConfig)
    session: ChatSessionConfig = Field(default_factory=ChatSessionConfig)
    tool_limits: ToolLimits = Field(default_factory=ToolLimits)
    policy: ChatPolicyConfig = Field(default_factory=ChatPolicyConfig)
    protection: ProtectionConfig = Field(default_factory=ProtectionConfig)
    ui: ChatUIConfig = Field(default_factory=ChatUIConfig)
    workspace: AssistantWorkspaceConfig = Field(
        default_factory=AssistantWorkspaceConfig
    )
    research: AssistantResearchConfig = Field(default_factory=AssistantResearchConfig)


__all__ = [
    "AssistantConfig",
    "AssistantResearchConfig",
    "AssistantWorkspaceConfig",
]
