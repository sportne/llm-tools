"""Confluence page-read tool."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.tools.filesystem.models import FileReadResult


class ReadConfluencePageInput(BaseModel):
    page_id: str
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)


class ReadConfluencePageOutput(FileReadResult):
    page_id: str
    title: str | None = None
    space_key: str | None = None
    web_url: str | None = None
    representation: str | None = None


__all__ = [
    "ReadConfluencePageInput",
    "ReadConfluencePageOutput",
]
