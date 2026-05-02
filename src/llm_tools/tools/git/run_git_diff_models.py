"""Git diff tool."""

from __future__ import annotations

from pydantic import BaseModel


class RunGitDiffInput(BaseModel):
    path: str = "."
    ref: str | None = None
    staged: bool = False


class RunGitDiffOutput(BaseModel):
    resolved_root: str
    diff_text: str
    truncated: bool = False


__all__ = [
    "RunGitDiffInput",
    "RunGitDiffOutput",
]
