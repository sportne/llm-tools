"""GitLab file-read tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.filesystem.models import FileReadResult


class ReadGitLabFileInput(BaseModel):
    project: str
    file_path: str
    ref: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class ReadGitLabFileOutput(FileReadResult):
    project: str
    ref: str


__all__ = [
    "ReadGitLabFileInput",
    "ReadGitLabFileOutput",
]
