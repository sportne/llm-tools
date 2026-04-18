"""Canonical typed models for durable harness session state."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from llm_tools.workflow_api.models import WorkflowTurnResult


class TaskLifecycleStatus(str, Enum):  # noqa: UP042
    """Lifecycle status for one tracked harness task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"


class VerificationStatus(str, Enum):  # noqa: UP042
    """Verification status recorded against a task."""

    NOT_RUN = "not_run"
    PASSED = "passed"
    FAILED = "failed"
    INCONCLUSIVE = "inconclusive"


class HarnessStopReason(str, Enum):  # noqa: UP042
    """Canonical reasons why harness execution stops."""

    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    VERIFICATION_FAILED = "verification_failed"
    NO_PROGRESS = "no_progress"
    CANCELED = "canceled"
    ERROR = "error"


class TurnDecisionAction(str, Enum):  # noqa: UP042
    """Actions available after one harness turn completes."""

    CONTINUE = "continue"
    SELECT_TASKS = "select_tasks"
    STOP = "stop"


class BudgetPolicy(BaseModel):
    """Configured execution limits for one harness session."""

    max_turns: int | None = Field(default=None, ge=1)
    max_tool_invocations: int | None = Field(default=None, ge=1)
    max_elapsed_seconds: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def validate_configured_limits(self) -> BudgetPolicy:
        """Require at least one positive budget limit to be configured."""
        if (
            self.max_turns is None
            and self.max_tool_invocations is None
            and self.max_elapsed_seconds is None
        ):
            raise ValueError("BudgetPolicy must configure at least one limit.")
        return self


class VerificationOutcome(BaseModel):
    """Persisted verification state for a task."""

    status: VerificationStatus = VerificationStatus.NOT_RUN
    checked_at: str | None = None
    summary: str | None = None
    evidence_refs: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_verification_state(self) -> VerificationOutcome:
        """Require checked timestamps and unique evidence once verification runs."""
        if self.status is not VerificationStatus.NOT_RUN and (
            self.checked_at is None or self.checked_at.strip() == ""
        ):
            raise ValueError("checked_at is required once verification has run.")
        if len(set(self.evidence_refs)) != len(self.evidence_refs):
            raise ValueError("Verification evidence_refs must be unique.")
        return self


class TaskRecord(BaseModel):
    """Durable task record tracked by the harness."""

    task_id: str
    title: str
    status: TaskLifecycleStatus = TaskLifecycleStatus.PENDING
    depends_on_task_ids: list[str] = Field(default_factory=list)
    verification: VerificationOutcome = Field(default_factory=VerificationOutcome)
    created_at: str | None = None
    updated_at: str | None = None

    @model_validator(mode="after")
    def validate_task_record(self) -> TaskRecord:
        """Require canonical task identity and dependency hygiene."""
        if self.task_id.strip() == "":
            raise ValueError("task_id must not be empty.")
        if self.title.strip() == "":
            raise ValueError("title must not be empty.")
        if self.task_id in self.depends_on_task_ids:
            raise ValueError("TaskRecord must not depend on itself.")
        if len(set(self.depends_on_task_ids)) != len(self.depends_on_task_ids):
            raise ValueError("TaskRecord dependency ids must be unique.")
        return self


class TurnDecision(BaseModel):
    """Harness-level decision emitted after a completed turn."""

    action: TurnDecisionAction
    selected_task_ids: list[str] = Field(default_factory=list)
    stop_reason: HarnessStopReason | None = None
    summary: str | None = None

    @model_validator(mode="after")
    def validate_turn_decision(self) -> TurnDecision:
        """Require canonical stop semantics and unique task selection."""
        if len(set(self.selected_task_ids)) != len(self.selected_task_ids):
            raise ValueError("selected_task_ids must be unique.")
        if self.action is TurnDecisionAction.STOP:
            if self.stop_reason is None:
                raise ValueError("stop_reason is required for stop decisions.")
            return self
        if self.stop_reason is not None:
            raise ValueError("stop_reason is only allowed for stop decisions.")
        return self


class HarnessTurn(BaseModel):
    """Persisted record for one harness turn."""

    turn_index: int = Field(ge=1)
    started_at: str
    workflow_result: WorkflowTurnResult | None = None
    decision: TurnDecision | None = None
    ended_at: str | None = None

    @model_validator(mode="after")
    def validate_turn_timestamps(self) -> HarnessTurn:
        """Require ended_at once a turn decision has been recorded."""
        if self.decision is not None and (
            self.ended_at is None or self.ended_at.strip() == ""
        ):
            raise ValueError("ended_at is required once decision exists.")
        return self


class HarnessSession(BaseModel):
    """Session-level durable metadata for harness execution."""

    session_id: str
    root_task_id: str
    budget_policy: BudgetPolicy
    started_at: str
    current_turn_index: int = Field(default=0, ge=0)
    ended_at: str | None = None
    stop_reason: HarnessStopReason | None = None

    @model_validator(mode="after")
    def validate_session_state(self) -> HarnessSession:
        """Require canonical session resume and stop metadata."""
        if self.root_task_id.strip() == "":
            raise ValueError("root_task_id must not be empty.")
        if self.ended_at is not None and self.stop_reason is None:
            raise ValueError("ended_at requires stop_reason.")
        return self


class HarnessState(BaseModel):
    """Top-level persisted harness state envelope."""

    schema_version: str = Field(min_length=1)
    session: HarnessSession
    tasks: list[TaskRecord] = Field(default_factory=list)
    turns: list[HarnessTurn] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_state_integrity(self) -> HarnessState:
        """Require cross-record references and turn ordering to be consistent."""
        task_ids = [task.task_id for task in self.tasks]
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("HarnessState task ids must be unique.")

        known_task_ids = set(task_ids)
        if self.session.root_task_id not in known_task_ids:
            raise ValueError("session.root_task_id must resolve to a known task.")

        for task in self.tasks:
            for dependency_id in task.depends_on_task_ids:
                if dependency_id not in known_task_ids:
                    raise ValueError(
                        "TaskRecord depends_on_task_ids must resolve to known tasks."
                    )

        turn_indices = [turn.turn_index for turn in self.turns]
        if turn_indices != list(range(1, len(self.turns) + 1)):
            raise ValueError("HarnessState turns must have contiguous indices from 1.")

        for turn in self.turns:
            if turn.decision is None:
                continue
            for task_id in turn.decision.selected_task_ids:
                if task_id not in known_task_ids:
                    raise ValueError(
                        "TurnDecision selected_task_ids must resolve to known tasks."
                    )

        if self.session.current_turn_index != len(self.turns):
            raise ValueError("session.current_turn_index must equal len(turns).")

        return self
