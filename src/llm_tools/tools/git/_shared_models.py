"""Shared helpers for git tool implementations."""

from __future__ import annotations

from pydantic import BaseModel


class GitCommandInput(BaseModel):
    path: str = "."


__all__ = [
    "GitCommandInput",
]
