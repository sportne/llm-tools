"""Models for repository-style text search tools."""

from __future__ import annotations

from pydantic import BaseModel, Field


class TextSearchMatch(BaseModel):
    """One matching line returned by the text-search tool."""

    path: str
    line_number: int
    line_text: str
    is_hidden: bool


class TextSearchResult(BaseModel):
    """Structured result for deterministic text search."""

    requested_path: str
    resolved_path: str
    query: str
    matches: list[TextSearchMatch] = Field(default_factory=list)
    truncated: bool = False


class SearchTextInput(BaseModel):
    path: str = "."
    query: str
    include_hidden: bool = False


class SearchTextOutput(TextSearchResult):
    pass


__all__ = [
    "SearchTextInput",
    "TextSearchMatch",
    "TextSearchResult",
    "SearchTextOutput",
]
