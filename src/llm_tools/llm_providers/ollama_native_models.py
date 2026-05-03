"""Models for the native Ollama provider client."""

from __future__ import annotations

from pydantic import BaseModel


class ProviderPreflightToolInput(BaseModel):
    """Arguments for the native tool-call preflight probe."""

    status: str


__all__ = ["ProviderPreflightToolInput"]
