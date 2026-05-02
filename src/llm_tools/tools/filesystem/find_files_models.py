"""Find-files filesystem tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.filesystem.models import FileSearchResult


class FindFilesInput(BaseModel):
    path: str = Field(
        default=".",
        description=(
            "Workspace-relative directory to search under. Use '.' for the "
            "workspace root."
        ),
    )
    pattern: str = Field(
        description=(
            "Workspace-relative glob pattern matched against full relative file "
            "paths under path. Use recursive globs such as '**/*.py' to find "
            "Python files in nested directories. A pattern like '*.py' only "
            "matches files directly at the search root."
        )
    )
    include_hidden: bool = Field(
        default=False,
        description=(
            "Include hidden files and directories such as '.github' or '.env' "
            "when true. Hidden paths are excluded by default."
        ),
    )


class FindFilesOutput(FileSearchResult):
    pass


__all__ = [
    "FindFilesInput",
    "FindFilesOutput",
]
