"""Approval and retry models for harness execution."""

from __future__ import annotations

from pydantic import BaseModel, Field


class HarnessRetryPolicy(BaseModel):
    """Retry budgets for driver, tool, and persistence recovery."""

    max_provider_retries: int = Field(default=0, ge=0)
    max_retryable_tool_retries: int = Field(default=0, ge=0)
    max_persistence_retries: int = Field(default=1, ge=0)


__all__ = [
    "HarnessRetryPolicy",
]
