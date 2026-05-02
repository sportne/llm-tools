"""Git log tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.git._shared import (
    MAX_GIT_LOG_LIMIT,
)


class RunGitLogInput(BaseModel):
    path: str = "."
    limit: int = Field(default=10, ge=1, le=MAX_GIT_LOG_LIMIT)


class RunGitLogOutput(BaseModel):
    resolved_root: str
    log_text: str
    truncated: bool = False


__all__ = [
    "RunGitLogInput",
    "RunGitLogOutput",
]
