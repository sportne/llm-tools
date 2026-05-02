"""Harness session service models and orchestration."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.harness_api.executor import (
    ApprovalResolution,
)
from llm_tools.harness_api.models import (
    BudgetPolicy,
    HarnessStopReason,
)
from llm_tools.harness_api.replay import (
    HarnessReplayResult,
    HarnessSessionSummary,
)
from llm_tools.harness_api.resume import ResumedHarnessSession
from llm_tools.harness_api.store import (
    StoredHarnessState,
)


class HarnessSessionCreateRequest(BaseModel):
    """Create a new persisted root-task harness session."""

    title: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    budget_policy: BudgetPolicy
    session_id: str | None = None
    root_task_id: str = "task-1"
    started_at: str | None = None


class HarnessSessionRunRequest(BaseModel):
    """Run a persisted harness session to its next durable stop."""

    session_id: str = Field(min_length=1)
    expected_revision: str | None = None
    allow_interrupted_turn_replay: bool = False


class HarnessSessionResumeRequest(BaseModel):
    """Resume a persisted harness session, optionally resolving approval."""

    session_id: str = Field(min_length=1)
    approval_resolution: ApprovalResolution | None = None
    allow_interrupted_turn_replay: bool = False


class HarnessSessionStopRequest(BaseModel):
    """Stop a persisted harness session without further execution."""

    session_id: str = Field(min_length=1)
    stop_reason: HarnessStopReason = HarnessStopReason.CANCELED


class HarnessSessionInspectRequest(BaseModel):
    """Inspect one persisted harness session."""

    session_id: str = Field(min_length=1)
    include_replay: bool = False


class HarnessSessionListRequest(BaseModel):
    """List recent persisted harness sessions."""

    limit: int | None = Field(default=None, ge=1)
    include_replay: bool = False


class HarnessSessionInspection(BaseModel):
    """Typed inspection payload for one stored harness session."""

    snapshot: StoredHarnessState
    resumed: ResumedHarnessSession
    summary: HarnessSessionSummary
    replay: HarnessReplayResult | None = None


class HarnessSessionListItem(BaseModel):
    """One recent persisted harness session."""

    snapshot: StoredHarnessState
    summary: HarnessSessionSummary
    replay: HarnessReplayResult | None = None


class HarnessSessionListResult(BaseModel):
    """Recent stored harness sessions in newest-first order."""

    sessions: list[HarnessSessionListItem] = Field(default_factory=list)


__all__ = [
    "HarnessSessionCreateRequest",
    "HarnessSessionInspectRequest",
    "HarnessSessionInspection",
    "HarnessSessionListItem",
    "HarnessSessionListRequest",
    "HarnessSessionListResult",
    "HarnessSessionResumeRequest",
    "HarnessSessionRunRequest",
    "HarnessSessionStopRequest",
]
