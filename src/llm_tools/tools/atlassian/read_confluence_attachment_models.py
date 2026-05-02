"""Confluence attachment-read tool."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from llm_tools.tools.filesystem.models import FileReadResult


class ReadConfluenceAttachmentInput(BaseModel):
    page_id: str
    attachment_id: str | None = None
    attachment_filename: str | None = None
    start_char: int | None = Field(default=None, ge=0)
    end_char: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_attachment_selector(self) -> ReadConfluenceAttachmentInput:
        if self.attachment_id is not None and self.attachment_filename is not None:
            raise ValueError(
                "provide at most one of attachment_id or attachment_filename"
            )
        if self.attachment_id is None and self.attachment_filename is None:
            raise ValueError("provide one of attachment_id or attachment_filename")
        return self


class ReadConfluenceAttachmentOutput(FileReadResult):
    page_id: str
    title: str | None = None
    space_key: str | None = None
    web_url: str | None = None
    attachment_id: str | None = None
    attachment_filename: str | None = None


__all__ = [
    "ReadConfluenceAttachmentInput",
    "ReadConfluenceAttachmentOutput",
]
