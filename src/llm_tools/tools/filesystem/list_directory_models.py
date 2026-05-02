"""List-directory filesystem tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.filesystem.models import DirectoryListingResult


class ListDirectoryInput(BaseModel):
    path: str = "."
    recursive: bool = False
    max_depth: int | None = Field(default=None, gt=0)
    include_hidden: bool = False


class ListDirectoryOutput(DirectoryListingResult):
    pass


__all__ = [
    "ListDirectoryInput",
    "ListDirectoryOutput",
]
