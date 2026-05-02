"""Jira issue read tool."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, model_validator


class ReadJiraIssueInput(BaseModel):
    issue_key: str
    requested_fields: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_requested_fields(self) -> ReadJiraIssueInput:
        if len(set(self.requested_fields)) != len(self.requested_fields):
            raise ValueError("requested_fields must be unique.")
        if any(field.strip() == "" for field in self.requested_fields):
            raise ValueError("requested_fields must not contain empty entries.")
        return self


class ReadJiraIssueOutput(BaseModel):
    key: str
    summary: str | None = None
    description: Any = None
    status: str | None = None
    issue_type: str | None = None
    assignee: str | None = None
    requested_fields: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "ReadJiraIssueInput",
    "ReadJiraIssueOutput",
]
