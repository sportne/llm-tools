"""GitLab code-search tool."""

from __future__ import annotations

from pydantic import BaseModel, Field


class SearchGitLabCodeInput(BaseModel):
    project: str
    query: str
    ref: str | None = None
    limit: int = Field(default=20, ge=1, le=100)


class GitLabCodeSearchMatch(BaseModel):
    project: str
    path: str
    name: str
    ref: str | None = None
    start_line: int | None = None
    snippet: str | None = None


class SearchGitLabCodeOutput(BaseModel):
    project: str
    query: str
    ref: str | None = None
    matches: list[GitLabCodeSearchMatch] = Field(default_factory=list)
    truncated: bool = False


__all__ = [
    "GitLabCodeSearchMatch",
    "SearchGitLabCodeInput",
    "SearchGitLabCodeOutput",
]
