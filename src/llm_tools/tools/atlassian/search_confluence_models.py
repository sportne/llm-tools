"""Confluence search tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.atlassian._shared import (
    _REMOTE_COLLECTION_LIMIT,
)


class SearchConfluenceInput(BaseModel):
    cql: str
    limit: int = Field(default=20, ge=1, le=_REMOTE_COLLECTION_LIMIT)


class ConfluenceSearchMatch(BaseModel):
    content_id: str
    title: str | None = None
    content_type: str | None = None
    space_key: str | None = None
    excerpt: str | None = None
    web_url: str | None = None


class SearchConfluenceOutput(BaseModel):
    cql: str
    matches: list[ConfluenceSearchMatch] = Field(default_factory=list)
    truncated: bool = False


__all__ = [
    "ConfluenceSearchMatch",
    "SearchConfluenceInput",
    "SearchConfluenceOutput",
]
