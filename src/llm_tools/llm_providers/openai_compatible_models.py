"""OpenAI-compatible provider client implemented with Instructor."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ResponseModeStrategy(str, Enum):  # noqa: UP042
    """Instructor mode strategy for provider calls."""

    AUTO = "auto"
    TOOLS = "tools"
    JSON = "json"
    PROMPT_TOOLS = "prompt_tools"


class ProviderPreflightResult(BaseModel):
    """Typed result for one provider connectivity and mode probe."""

    ok: bool
    connection_succeeded: bool = False
    model_accepted: bool = False
    selected_mode_supported: bool = False
    model_listing_supported: bool = False
    available_models: list[str] = Field(default_factory=list)
    resolved_mode: ResponseModeStrategy | None = None
    actionable_message: str
    error_message: str | None = None


class _ProviderPreflightResponse(BaseModel):
    status: str


__all__ = [
    "ResponseModeStrategy",
    "ProviderPreflightResult",
    "_ProviderPreflightResponse",
]
