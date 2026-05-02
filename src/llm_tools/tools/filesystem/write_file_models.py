"""Write-file filesystem tool."""

from __future__ import annotations

from pydantic import BaseModel


class WriteFileInput(BaseModel):
    path: str
    content: str
    encoding: str = "utf-8"
    overwrite: bool = False
    create_parents: bool = False


class WriteFileOutput(BaseModel):
    path: str
    resolved_path: str
    bytes_written: int
    created: bool


__all__ = [
    "WriteFileInput",
    "WriteFileOutput",
]
