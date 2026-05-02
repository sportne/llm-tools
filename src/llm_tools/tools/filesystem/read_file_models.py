"""Read-file filesystem tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.filesystem.models import FileReadResult


class ReadFileInput(BaseModel):
    path: str
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class ReadFileOutput(FileReadResult):
    pass


__all__ = [
    "ReadFileInput",
    "ReadFileOutput",
]
