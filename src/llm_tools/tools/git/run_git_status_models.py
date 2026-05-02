"""Git status tool."""

from __future__ import annotations

from pydantic import BaseModel


class RunGitStatusOutput(BaseModel):
    resolved_root: str
    status_text: str
    truncated: bool = False


__all__ = [
    "RunGitStatusOutput",
]
