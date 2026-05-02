"""Bitbucket file-read tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.filesystem.models import FileReadResult


class ReadBitbucketFileInput(BaseModel):
    project_key: str
    repository_slug: str
    path: str
    ref: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class ReadBitbucketFileOutput(FileReadResult):
    project_key: str
    repository_slug: str
    ref: str


__all__ = [
    "ReadBitbucketFileInput",
    "ReadBitbucketFileOutput",
]
