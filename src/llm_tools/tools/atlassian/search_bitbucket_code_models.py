"""Bitbucket code-search tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.atlassian._shared import (
    _REMOTE_COLLECTION_LIMIT,
)


class SearchBitbucketCodeInput(BaseModel):
    project_key: str
    query: str
    limit: int = Field(default=20, ge=1, le=_REMOTE_COLLECTION_LIMIT)


class BitbucketCodeMatch(BaseModel):
    repository_slug: str | None = None
    path: str
    line_number: int | None = None
    snippet: str | None = None


class SearchBitbucketCodeOutput(BaseModel):
    project_key: str
    query: str
    matches: list[BitbucketCodeMatch] = Field(default_factory=list)
    truncated: bool = False


__all__ = [
    "BitbucketCodeMatch",
    "SearchBitbucketCodeInput",
    "SearchBitbucketCodeOutput",
]
