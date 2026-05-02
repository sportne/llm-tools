"""Harness executor loop implementation."""

from __future__ import annotations

from pydantic import BaseModel

from llm_tools.harness_api.resume import (
    ResumedHarnessSession,
)
from llm_tools.harness_api.store import StoredHarnessState


class HarnessExecutionResult(BaseModel):
    """Final stored snapshot and resume view after executor work stops."""

    snapshot: StoredHarnessState
    resumed: ResumedHarnessSession


__all__ = [
    "HarnessExecutionResult",
]
