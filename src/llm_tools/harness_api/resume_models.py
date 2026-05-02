"""Typed rehydration and resume inspection for persisted harness sessions."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from llm_tools.harness_api.models import (
    HarnessTurn,
    PendingApprovalRecord,
    TaskRecord,
)
from llm_tools.harness_api.store import (
    StoredHarnessState,
)


class ResumeDisposition(str, Enum):  # noqa: UP042
    """Canonical classification of a persisted session at resume time."""

    RUNNABLE = "runnable"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    APPROVAL_EXPIRED = "approval_expired"
    INTERRUPTED = "interrupted"
    TERMINAL = "terminal"
    INCOMPATIBLE_SCHEMA = "incompatible_schema"
    CORRUPT = "corrupt"


class ResumeIssue(BaseModel):
    """Structured validation issue surfaced while rehydrating persisted state."""

    code: str = Field(min_length=1)
    message: str = Field(min_length=1)


class ResumedHarnessSession(BaseModel):
    """Typed result of resume-time state classification."""

    snapshot: StoredHarnessState
    disposition: ResumeDisposition
    active_tasks: list[TaskRecord] = Field(default_factory=list)
    incomplete_turn: HarnessTurn | None = None
    pending_approval: PendingApprovalRecord | None = None
    issues: list[ResumeIssue] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_resume_shape(self) -> ResumedHarnessSession:
        """Require approval metadata for approval-related resume dispositions."""
        if self.disposition in {
            ResumeDisposition.WAITING_FOR_APPROVAL,
            ResumeDisposition.APPROVAL_EXPIRED,
        } and (self.incomplete_turn is None or self.pending_approval is None):
            raise ValueError(
                "Approval resume dispositions require incomplete_turn and "
                "pending_approval."
            )
        return self


__all__ = [
    "ResumeDisposition",
    "ResumeIssue",
    "ResumedHarnessSession",
]
