"""Jira search tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.atlassian._shared import (
    _REMOTE_COLLECTION_LIMIT,
)


class JiraIssueSummary(BaseModel):
    key: str
    summary: str | None = None
    status: str | None = None
    issue_type: str | None = None
    assignee: str | None = None


class SearchJiraInput(BaseModel):
    jql: str
    limit: int = Field(default=20, ge=1, le=_REMOTE_COLLECTION_LIMIT)


class SearchJiraOutput(BaseModel):
    issues: list[JiraIssueSummary] = Field(default_factory=list)
    truncated: bool = False


__all__ = [
    "JiraIssueSummary",
    "SearchJiraInput",
    "SearchJiraOutput",
]
