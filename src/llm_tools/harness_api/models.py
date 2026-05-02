"""Canonical typed models for durable harness session state."""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum

from pydantic import BaseModel, Field, field_validator, model_validator

from llm_tools.harness_api.approval_context import (
    rehydrate_pending_approval_context as rehydrate_pending_approval_context,
)
from llm_tools.harness_api.approval_context import (
    sanitize_pending_approval_context as sanitize_pending_approval_context,
)
from llm_tools.harness_api.verification import (
    NoProgressSignal,
    VerificationEvidenceRecord,
    VerificationExpectation,
    VerificationStatus,
)
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolContext
from llm_tools.workflow_api.models import ApprovalRequest, WorkflowTurnResult


class TaskLifecycleStatus(str, Enum):  # noqa: UP042
    """Lifecycle status for one tracked harness task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELED = "canceled"
    SUPERSEDED = "superseded"


class TaskOrigin(str, Enum):  # noqa: UP042
    """How a harness task entered the durable task graph."""

    USER_REQUESTED = "user_requested"
    DERIVED = "derived"


class HarnessStopReason(str, Enum):  # noqa: UP042
    """Canonical reasons why harness execution stops."""

    COMPLETED = "completed"
    BUDGET_EXHAUSTED = "budget_exhausted"
    VERIFICATION_FAILED = "verification_failed"
    NO_PROGRESS = "no_progress"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_EXPIRED = "approval_expired"
    APPROVAL_CANCELED = "approval_canceled"
    CANCELED = "canceled"
    ERROR = "error"


class TurnDecisionAction(str, Enum):  # noqa: UP042
    """Actions available after one harness turn completes."""

    CONTINUE = "continue"
    SELECT_TASKS = "select_tasks"
    STOP = "stop"


class TurnApprovalAuditRecord(BaseModel):
    """Minimal persisted approval metadata retained for turn-history audit."""

    approval_id: str = Field(min_length=1)
    invocation_index: int = Field(ge=1)
    tool_name: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    policy_reason: str
    policy_metadata: dict[str, object] = Field(default_factory=dict)

    @classmethod
    def from_approval_request(
        cls,
        approval_request: ApprovalRequest,
    ) -> TurnApprovalAuditRecord:
        return cls(
            approval_id=approval_request.approval_id,
            invocation_index=approval_request.invocation_index,
            tool_name=approval_request.tool_name,
            tool_version=approval_request.tool_version,
            policy_reason=approval_request.policy_reason,
            policy_metadata=dict(approval_request.policy_metadata),
        )


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
    intent: str
    origin: TaskOrigin
    status: TaskLifecycleStatus = TaskLifecycleStatus.PENDING
    parent_task_id: str | None = None
    depends_on_task_ids: list[str] = Field(default_factory=list)
    superseded_by_task_id: str | None = None
    verification_expectations: list[VerificationExpectation] = Field(
        default_factory=list
    )
    verification: VerificationOutcome = Field(default_factory=VerificationOutcome)
    artifact_refs: list[str] = Field(default_factory=list)
    started_at: str | None = None
    finished_at: str | None = None
    status_summary: str | None = None
    retry_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_task_record(self) -> TaskRecord:
        """Require canonical task identity and dependency hygiene."""
        _validate_task_identity(self)
        _validate_task_relationships(self)
        _validate_task_verification(self)
        _validate_task_artifacts(self)
        _validate_task_supersession(self)
        _validate_task_timestamps(self)
        return self


def _validate_task_identity(task: TaskRecord) -> None:
    if task.task_id.strip() == "":
        raise ValueError("task_id must not be empty.")
    if task.title.strip() == "":
        raise ValueError("title must not be empty.")
    if task.intent.strip() == "":
        raise ValueError("intent must not be empty.")


def _validate_task_relationships(task: TaskRecord) -> None:
    if task.task_id in task.depends_on_task_ids:
        raise ValueError("TaskRecord must not depend on itself.")
    if len(set(task.depends_on_task_ids)) != len(task.depends_on_task_ids):
        raise ValueError("TaskRecord dependency ids must be unique.")
    if task.origin is TaskOrigin.USER_REQUESTED and task.parent_task_id is not None:
        raise ValueError("user_requested tasks must not set parent_task_id.")
    if task.origin is TaskOrigin.DERIVED and (
        task.parent_task_id is None or task.parent_task_id.strip() == ""
    ):
        raise ValueError("derived tasks must set parent_task_id.")
    if task.parent_task_id is not None and task.parent_task_id.strip() == "":
        raise ValueError("parent_task_id must not be empty when provided.")


def _validate_task_verification(task: TaskRecord) -> None:
    expectation_ids = [
        expectation.expectation_id for expectation in task.verification_expectations
    ]
    if len(set(expectation_ids)) != len(expectation_ids):
        raise ValueError("TaskRecord verification expectation ids must be unique.")
    if (
        task.status is TaskLifecycleStatus.COMPLETED
        and any(
            expectation.required_for_completion
            for expectation in task.verification_expectations
        )
        and task.verification.status is not VerificationStatus.PASSED
    ):
        raise ValueError(
            "Completed tasks with required verification expectations must "
            "have verification.status=passed."
        )


def _validate_task_artifacts(task: TaskRecord) -> None:
    if len(set(task.artifact_refs)) != len(task.artifact_refs):
        raise ValueError("TaskRecord artifact_refs must be unique.")


def _validate_task_supersession(task: TaskRecord) -> None:
    if task.superseded_by_task_id is not None and (
        task.superseded_by_task_id.strip() == ""
    ):
        raise ValueError("superseded_by_task_id must not be empty when provided.")
    if task.superseded_by_task_id == task.task_id:
        raise ValueError("TaskRecord must not supersede itself.")
    if task.status is TaskLifecycleStatus.SUPERSEDED:
        if task.superseded_by_task_id is None:
            raise ValueError("SUPERSEDED tasks must set superseded_by_task_id.")
        return
    if task.superseded_by_task_id is not None:
        raise ValueError("superseded_by_task_id is only allowed for SUPERSEDED tasks.")


def _validate_task_timestamps(task: TaskRecord) -> None:
    if task.started_at is not None and task.started_at.strip() == "":
        raise ValueError("started_at must not be empty when provided.")
    if task.finished_at is not None and task.finished_at.strip() == "":
        raise ValueError("finished_at must not be empty when provided.")
    if task.status_summary is not None and task.status_summary.strip() == "":
        raise ValueError("status_summary must not be empty when provided.")


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


class PendingApprovalRecord(BaseModel):
    """Durable approval state required to resume an interrupted workflow turn."""

    approval_request: ApprovalRequest
    parsed_response: ParsedModelResponse
    base_context: ToolContext
    pending_index: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_pending_approval(self) -> PendingApprovalRecord:
        """Require the approval record to point at a known invocation."""
        if self.pending_index != self.approval_request.invocation_index:
            raise ValueError(
                "pending_index must match approval_request.invocation_index."
            )
        if self.pending_index > len(self.parsed_response.invocations):
            raise ValueError(
                "pending_index must reference an invocation in parsed_response."
            )
        return self


class HarnessTurn(BaseModel):
    """Persisted record for one harness turn."""

    turn_index: int = Field(ge=1)
    started_at: str
    selected_task_ids: list[str] = Field(default_factory=list)
    workflow_result: WorkflowTurnResult | None = None
    pending_approval_request: TurnApprovalAuditRecord | None = None
    verification_status_by_task_id: dict[str, str] = Field(default_factory=dict)
    decision: TurnDecision | None = None
    no_progress_signals: list[NoProgressSignal] = Field(default_factory=list)
    ended_at: str | None = None

    @field_validator("pending_approval_request", mode="before")
    @classmethod
    def coerce_pending_approval_request(
        cls,
        value: object,
    ) -> object:
        if value is None or isinstance(value, TurnApprovalAuditRecord):
            return value
        if isinstance(value, ApprovalRequest):
            return TurnApprovalAuditRecord.from_approval_request(value)
        if isinstance(value, dict):
            allowed_keys = {
                "approval_id",
                "invocation_index",
                "tool_name",
                "tool_version",
                "policy_reason",
                "policy_metadata",
            }
            return {key: value[key] for key in allowed_keys if key in value}
        return value

    @model_validator(mode="after")
    def validate_turn_timestamps(self) -> HarnessTurn:
        """Require ended_at once a turn decision has been recorded."""
        if self.started_at.strip() == "":
            raise ValueError("started_at must not be empty.")
        if len(set(self.selected_task_ids)) != len(self.selected_task_ids):
            raise ValueError("selected_task_ids must be unique.")
        if self.decision is not None and (
            self.ended_at is None or self.ended_at.strip() == ""
        ):
            raise ValueError("ended_at is required once decision exists.")
        if (
            self.decision is not None
            and self.decision.action is TurnDecisionAction.STOP
            and self.decision.stop_reason is HarnessStopReason.NO_PROGRESS
            and len(self.no_progress_signals) == 0
        ):
            raise ValueError(
                "HarnessTurn stopping for no_progress must include at least one "
                "no_progress_signal."
            )
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
    retry_count: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def validate_session_state(self) -> HarnessSession:
        """Require canonical session resume and stop metadata."""
        if self.session_id.strip() == "":
            raise ValueError("session_id must not be empty.")
        if self.root_task_id.strip() == "":
            raise ValueError("root_task_id must not be empty.")
        if self.started_at.strip() == "":
            raise ValueError("started_at must not be empty.")
        if self.ended_at is not None and self.stop_reason is None:
            raise ValueError("ended_at requires stop_reason.")
        return self


class HarnessState(BaseModel):
    """Top-level persisted harness state envelope."""

    schema_version: str = Field(min_length=1)
    session: HarnessSession
    tasks: list[TaskRecord] = Field(default_factory=list)
    verification_evidence: list[VerificationEvidenceRecord] = Field(
        default_factory=list
    )
    turns: list[HarnessTurn] = Field(default_factory=list)
    pending_approvals: list[PendingApprovalRecord] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_state_integrity(self) -> HarnessState:
        """Require cross-record references and turn ordering to be consistent."""
        task_map = _validate_harness_tasks(self)
        _validate_harness_verification_evidence(self, known_task_ids=set(task_map))
        _validate_harness_turns(self, known_task_ids=set(task_map))
        _validate_harness_session_shape(self)
        _validate_harness_pending_approvals(self)
        return self


def _validate_harness_tasks(state: HarnessState) -> dict[str, TaskRecord]:
    task_ids = [task.task_id for task in state.tasks]
    if len(set(task_ids)) != len(task_ids):
        raise ValueError("HarnessState task ids must be unique.")

    task_map = {task.task_id: task for task in state.tasks}
    known_task_ids = set(task_map)
    if state.session.root_task_id not in task_map:
        raise ValueError("session.root_task_id must resolve to a known task.")

    for task in state.tasks:
        _validate_task_references(task, known_task_ids)

    root_candidates = [
        task
        for task in state.tasks
        if task.origin is TaskOrigin.USER_REQUESTED and task.parent_task_id is None
    ]
    if len(root_candidates) != 1:
        raise ValueError(
            "HarnessState must contain exactly one root user_requested task."
        )
    if root_candidates[0].task_id != state.session.root_task_id:
        raise ValueError(
            "session.root_task_id must resolve to the root user_requested task."
        )

    _ensure_acyclic_dependency_graph(
        state.tasks,
        lambda task: task.depends_on_task_ids,
        "TaskRecord dependency graph must be acyclic.",
    )
    _ensure_acyclic_parent_graph(state.tasks)
    _ensure_acyclic_supersession_graph(state.tasks)
    return task_map


def _validate_task_references(task: TaskRecord, known_task_ids: set[str]) -> None:
    for dependency_id in task.depends_on_task_ids:
        if dependency_id not in known_task_ids:
            raise ValueError(
                "TaskRecord depends_on_task_ids must resolve to known tasks."
            )
    if task.parent_task_id is not None and task.parent_task_id not in known_task_ids:
        raise ValueError("TaskRecord parent_task_id must resolve to a known task.")
    if (
        task.superseded_by_task_id is not None
        and task.superseded_by_task_id not in known_task_ids
    ):
        raise ValueError(
            "TaskRecord superseded_by_task_id must resolve to a known task."
        )


def _validate_harness_turns(state: HarnessState, *, known_task_ids: set[str]) -> None:
    turn_indices = [turn.turn_index for turn in state.turns]
    if turn_indices != list(range(1, len(state.turns) + 1)):
        raise ValueError("HarnessState turns must have contiguous indices from 1.")

    for turn in state.turns:
        for task_id in turn.selected_task_ids:
            if task_id not in known_task_ids:
                raise ValueError(
                    "HarnessTurn selected_task_ids must resolve to known tasks."
                )
        for signal in turn.no_progress_signals:
            if signal.task_id is not None and signal.task_id not in known_task_ids:
                raise ValueError(
                    "No-progress signal task ids must resolve to known tasks."
                )
        if turn.decision is None:
            continue
        for task_id in turn.decision.selected_task_ids:
            if task_id not in known_task_ids:
                raise ValueError(
                    "TurnDecision selected_task_ids must resolve to known tasks."
                )

    incomplete_turn_indices = [
        turn.turn_index for turn in state.turns if turn.decision is None
    ]
    if len(incomplete_turn_indices) > 1:
        raise ValueError("HarnessState may contain at most one incomplete turn.")
    if incomplete_turn_indices and incomplete_turn_indices[0] != len(state.turns):
        raise ValueError("HarnessState incomplete turn is only allowed at the tail.")
    if state.session.ended_at is not None and incomplete_turn_indices:
        raise ValueError("Ended sessions must not retain incomplete turns.")


def _validate_harness_session_shape(state: HarnessState) -> None:
    if state.session.current_turn_index != len(state.turns):
        raise ValueError("session.current_turn_index must equal len(turns).")


def _validate_harness_verification_evidence(
    state: HarnessState,
    *,
    known_task_ids: set[str],
) -> None:
    evidence_by_id: dict[str, VerificationEvidenceRecord] = {}
    for evidence in state.verification_evidence:
        if evidence.evidence_id in evidence_by_id:
            raise ValueError("HarnessState verification evidence ids must be unique.")
        if evidence.task_id is not None and evidence.task_id not in known_task_ids:
            raise ValueError(
                "Verification evidence task ids must resolve to known tasks."
            )
        evidence_by_id[evidence.evidence_id] = evidence

    for task in state.tasks:
        for evidence_ref in task.verification.evidence_refs:
            referenced_evidence = evidence_by_id.get(evidence_ref)
            if referenced_evidence is None:
                raise ValueError(
                    "VerificationOutcome evidence_refs must resolve to known "
                    "verification evidence."
                )
            owner_task_id = referenced_evidence.task_id
            if owner_task_id is not None and owner_task_id != task.task_id:
                raise ValueError(
                    "VerificationOutcome evidence_refs must not reference "
                    "evidence owned by a different task."
                )


def _validate_harness_pending_approvals(state: HarnessState) -> None:
    approval_ids = [
        approval.approval_request.approval_id for approval in state.pending_approvals
    ]
    if len(set(approval_ids)) != len(approval_ids):
        raise ValueError("HarnessState pending approval ids must be unique.")
    if len(state.pending_approvals) > 1:
        raise ValueError(
            "HarnessState currently supports at most one pending approval."
        )
    if state.session.ended_at is not None and state.pending_approvals:
        raise ValueError("Ended sessions must not retain pending approvals.")


def _ensure_acyclic_dependency_graph(
    tasks: list[TaskRecord],
    edge_getter: Callable[[TaskRecord], list[str]],
    message: str,
) -> None:
    task_map = {task.task_id: task for task in tasks}
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(task_id: str) -> None:
        if task_id in visited:
            return
        if task_id in visiting:
            raise ValueError(message)

        visiting.add(task_id)
        for dependency_id in edge_getter(task_map[task_id]):
            visit(dependency_id)
        visiting.remove(task_id)
        visited.add(task_id)

    for task in tasks:
        visit(task.task_id)


def _ensure_acyclic_parent_graph(tasks: list[TaskRecord]) -> None:
    task_map = {task.task_id: task for task in tasks}

    for task in tasks:
        lineage: set[str] = {task.task_id}
        parent_id = task.parent_task_id
        while parent_id is not None:
            if parent_id in lineage:
                raise ValueError("TaskRecord parent_task_id graph must be acyclic.")
            lineage.add(parent_id)
            parent_id = task_map[parent_id].parent_task_id


def _ensure_acyclic_supersession_graph(tasks: list[TaskRecord]) -> None:
    task_map = {task.task_id: task for task in tasks}

    for task in tasks:
        lineage: set[str] = {task.task_id}
        replacement_id = task.superseded_by_task_id
        while replacement_id is not None:
            if replacement_id in lineage:
                raise ValueError(
                    "TaskRecord superseded_by_task_id graph must be acyclic."
                )
            lineage.add(replacement_id)
            replacement_id = task_map[replacement_id].superseded_by_task_id
