"""Harness observability, summary, and replay contracts."""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from llm_tools.harness_api.models import (
    HarnessStopReason,
    TurnDecisionAction,
)
from llm_tools.tool_api import ErrorCode
from llm_tools.workflow_api import (
    WorkflowInvocationStatus,
)


class HarnessReplayMode(str, Enum):  # noqa: UP042
    """Supported harness replay fidelity modes."""

    TRACE = "trace"


class HarnessPolicySnapshot(BaseModel):
    """Redacted policy decision snapshot for one workflow invocation."""

    tool_name: str = Field(min_length=1)
    tool_version: str | None = None
    allowed: bool
    requires_approval: bool = False
    reason: str = Field(min_length=1)
    metadata: dict[str, object] = Field(default_factory=dict)
    source: str = Field(min_length=1)


class HarnessInvocationTrace(BaseModel):
    """Trace record for one workflow invocation outcome."""

    invocation_index: int = Field(ge=1)
    invocation_id: str | None = None
    status: WorkflowInvocationStatus
    tool_name: str = Field(min_length=1)
    tool_version: str | None = None
    redacted_arguments: dict[str, object] = Field(default_factory=dict)
    policy_snapshot: HarnessPolicySnapshot | None = None
    ok: bool | None = None
    error_code: ErrorCode | None = None
    approval_id: str | None = None
    redaction: dict[str, object] = Field(default_factory=dict)
    logs: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)


class HarnessTurnTrace(BaseModel):
    """Structured harness trace for one persisted turn."""

    turn_index: int = Field(ge=1)
    started_at: str = Field(min_length=1)
    ended_at: str | None = None
    selected_task_ids: list[str] = Field(default_factory=list)
    planner_selected_task_ids: list[str] = Field(default_factory=list)
    replanning_triggers: list[str] = Field(default_factory=list)
    context_projection: dict[str, Any] | None = None
    workflow_outcome_statuses: list[WorkflowInvocationStatus] = Field(
        default_factory=list
    )
    invocation_traces: list[HarnessInvocationTrace] = Field(default_factory=list)
    pending_approval_id: str | None = None
    verification_status_by_task_id: dict[str, str] = Field(default_factory=dict)
    no_progress_signals: list[str] = Field(default_factory=list)
    decision_action: TurnDecisionAction | None = None
    decision_stop_reason: HarnessStopReason | None = None
    decision_summary: str | None = None

    @model_validator(mode="after")
    def validate_turn_shape(self) -> HarnessTurnTrace:
        """Require unique selected task ids and stop reasons only on stop decisions."""
        if len(set(self.selected_task_ids)) != len(self.selected_task_ids):
            raise ValueError("HarnessTurnTrace selected_task_ids must be unique.")
        if self.decision_action is not TurnDecisionAction.STOP and (
            self.decision_stop_reason is not None
        ):
            raise ValueError(
                "HarnessTurnTrace decision_stop_reason is only allowed for stop "
                "decisions."
            )
        return self


class HarnessSessionTrace(BaseModel):
    """Aggregated trace history for one persisted harness session."""

    session_id: str = Field(min_length=1)
    turns: list[HarnessTurnTrace] = Field(default_factory=list)
    final_stop_reason: HarnessStopReason | None = None

    @model_validator(mode="after")
    def validate_turn_indices(self) -> HarnessSessionTrace:
        """Require unique turn indices in ascending order."""
        indices = [turn.turn_index for turn in self.turns]
        if indices != sorted(set(indices)):
            raise ValueError(
                "HarnessSessionTrace turns must have unique ascending indices."
            )
        return self


class HarnessSessionSummary(BaseModel):
    """Operator-facing summary derived from canonical state and traces."""

    session_id: str = Field(min_length=1)
    current_turn_index: int = Field(ge=0)
    total_turns: int = Field(ge=0)
    stop_reason: HarnessStopReason | None = None
    completed_task_ids: list[str] = Field(default_factory=list)
    active_task_ids: list[str] = Field(default_factory=list)
    pending_approval_ids: list[str] = Field(default_factory=list)
    verification_status_counts: dict[str, int] = Field(default_factory=dict)
    latest_decision_summary: str | None = None


class HarnessReplayStep(BaseModel):
    """One replayable step reconstructed from persisted harness artifacts."""

    turn_index: int = Field(ge=1)
    selected_task_ids: list[str] = Field(default_factory=list)
    workflow_outcome_statuses: list[WorkflowInvocationStatus] = Field(
        default_factory=list
    )
    decision_action: TurnDecisionAction | None = None
    decision_stop_reason: HarnessStopReason | None = None
    decision_summary: str | None = None


class HarnessReplayResult(BaseModel):
    """Deterministic replay projection for one persisted harness session."""

    session_id: str = Field(min_length=1)
    mode: HarnessReplayMode = HarnessReplayMode.TRACE
    steps: list[HarnessReplayStep] = Field(default_factory=list)
    final_stop_reason: HarnessStopReason | None = None
    limitations: list[str] = Field(default_factory=list)


class StoredHarnessArtifacts(BaseModel):
    """Derived observability artifacts attached to one stored snapshot."""

    trace: HarnessSessionTrace | None = None
    summary: HarnessSessionSummary | None = None


__all__ = [
    "HarnessInvocationTrace",
    "HarnessPolicySnapshot",
    "HarnessReplayMode",
    "HarnessReplayResult",
    "HarnessReplayStep",
    "HarnessSessionSummary",
    "HarnessSessionTrace",
    "HarnessTurnTrace",
    "StoredHarnessArtifacts",
]
