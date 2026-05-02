"""Get-file-info filesystem tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.filesystem.models import FileInfoResult, GetFileInfoInputShape


class GetFileInfoOutput(BaseModel):
    results: list[FileInfoResult] = Field(default_factory=list)


class GetFileInfoInput(GetFileInfoInputShape):
    pass


__all__ = [
    "GetFileInfoOutput",
    "GetFileInfoInput",
]
