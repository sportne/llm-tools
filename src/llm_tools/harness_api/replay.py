"""Harness observability, summary, and replay contracts."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field, model_validator

from llm_tools.harness_api.models import (
    HarnessState,
    HarnessStopReason,
    HarnessTurn,
    TaskLifecycleStatus,
    TurnDecisionAction,
)
from llm_tools.tool_api import ErrorCode, ToolContext
from llm_tools.workflow_api import (
    ApprovalRequest,
    WorkflowInvocationOutcome,
    WorkflowInvocationStatus,
)

if TYPE_CHECKING:
    from llm_tools.harness_api.store import StoredHarnessState


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


def build_session_trace(state: HarnessState) -> HarnessSessionTrace | None:
    """Rebuild a trusted trace view directly from canonical harness state."""
    if not state.turns:
        return None

    return HarnessSessionTrace(
        session_id=state.session.session_id,
        turns=[
            build_turn_trace(
                turn=turn,
                context=None,
                tasks_state=None,
            )
            for turn in state.turns
        ],
        final_stop_reason=state.session.stop_reason,
    )


def build_canonical_artifacts(state: HarnessState) -> StoredHarnessArtifacts:
    """Rebuild trusted stored artifacts directly from canonical state."""
    trace = build_session_trace(state)
    summary = build_session_summary(state, trace=trace)
    return StoredHarnessArtifacts(trace=trace, summary=summary)


def build_session_summary(
    state: HarnessState,
    *,
    trace: HarnessSessionTrace | None = None,
) -> HarnessSessionSummary:
    """Build a durable operator-facing summary from canonical harness state."""
    del trace
    verification_counts: dict[str, int] = {}
    for task in state.tasks:
        key = task.verification.status.value
        verification_counts[key] = verification_counts.get(key, 0) + 1

    latest_decision_summary = None
    for turn in reversed(state.turns):
        if turn.decision is not None and turn.decision.summary is not None:
            latest_decision_summary = turn.decision.summary
            break

    return HarnessSessionSummary(
        session_id=state.session.session_id,
        current_turn_index=state.session.current_turn_index,
        total_turns=len(state.turns),
        stop_reason=state.session.stop_reason,
        completed_task_ids=[
            task.task_id
            for task in state.tasks
            if task.status is TaskLifecycleStatus.COMPLETED
        ],
        active_task_ids=[
            task.task_id
            for task in state.tasks
            if task.status
            in {
                TaskLifecycleStatus.PENDING,
                TaskLifecycleStatus.IN_PROGRESS,
                TaskLifecycleStatus.BLOCKED,
            }
        ],
        pending_approval_ids=[
            approval.approval_request.approval_id
            for approval in state.pending_approvals
        ],
        verification_status_counts=verification_counts,
        latest_decision_summary=latest_decision_summary,
    )


def append_turn_trace(
    trace: HarnessSessionTrace | None,
    *,
    session_id: str,
    turn_trace: HarnessTurnTrace,
    final_stop_reason: HarnessStopReason | None,
) -> HarnessSessionTrace:
    """Append or replace one turn trace in the aggregated session trace."""
    turns = [] if trace is None else list(trace.turns)
    replaced = False
    for index, existing in enumerate(turns):
        if existing.turn_index == turn_trace.turn_index:
            replacement = turn_trace
            if (
                existing.pending_approval_id is not None
                and replacement.pending_approval_id is None
            ):
                resumed_invocation_traces = [
                    invocation
                    if invocation.approval_id is not None
                    else invocation.model_copy(
                        update={"approval_id": existing.pending_approval_id}
                    )
                    for invocation in replacement.invocation_traces
                ]
                replacement = replacement.model_copy(
                    update={
                        "pending_approval_id": existing.pending_approval_id,
                        "invocation_traces": [
                            *existing.invocation_traces,
                            *resumed_invocation_traces,
                        ],
                        "workflow_outcome_statuses": [
                            *existing.workflow_outcome_statuses,
                            *replacement.workflow_outcome_statuses,
                        ],
                    }
                )
            turns[index] = replacement
            replaced = True
            break
    if not replaced:
        turns.append(turn_trace)
    turns.sort(key=lambda item: item.turn_index)
    return HarnessSessionTrace(
        session_id=session_id,
        turns=turns,
        final_stop_reason=final_stop_reason,
    )


def build_turn_trace(
    *,
    turn: HarnessTurn,
    context: ToolContext | None,
    tasks_state: HarnessState | None,
) -> HarnessTurnTrace:
    """Build one structured trace record for a persisted harness turn."""
    context_projection = None
    planner_selected_task_ids: list[str] = []
    replanning_triggers: list[str] = []
    if context is not None:
        payload = context.metadata.get("harness_turn_context")
        if isinstance(payload, dict):
            context_projection = payload
        planning_payload = context.metadata.get("harness_planning")
        if isinstance(planning_payload, dict):
            selected_payload = planning_payload.get("selected_task_ids")
            if isinstance(selected_payload, list):
                planner_selected_task_ids = [
                    str(task_id) for task_id in selected_payload if str(task_id).strip()
                ]
            trigger_payload = planning_payload.get("replanning_triggers")
            if isinstance(trigger_payload, list):
                replanning_triggers = [
                    str(trigger) for trigger in trigger_payload if str(trigger).strip()
                ]

    invocation_traces = _canonical_invocation_traces(turn)
    workflow_statuses = _canonical_workflow_outcome_statuses(turn)
    pending_approval_id = _canonical_pending_approval_id(turn)

    verification_status_by_task_id = dict(turn.verification_status_by_task_id)
    if (
        not verification_status_by_task_id
        and context is not None
        and tasks_state is not None
    ):
        selected_task_ids = set(turn.selected_task_ids)
        verification_status_by_task_id = {
            task.task_id: task.verification.status.value
            for task in tasks_state.tasks
            if task.task_id in selected_task_ids
        }
    decision = turn.decision
    return HarnessTurnTrace(
        turn_index=turn.turn_index,
        started_at=turn.started_at,
        ended_at=turn.ended_at,
        selected_task_ids=list(turn.selected_task_ids),
        planner_selected_task_ids=planner_selected_task_ids,
        replanning_triggers=replanning_triggers,
        context_projection=context_projection,
        workflow_outcome_statuses=workflow_statuses,
        invocation_traces=invocation_traces,
        pending_approval_id=pending_approval_id,
        verification_status_by_task_id=verification_status_by_task_id,
        no_progress_signals=[
            signal.summary or signal.kind.value for signal in turn.no_progress_signals
        ],
        decision_action=None if decision is None else decision.action,
        decision_stop_reason=None if decision is None else decision.stop_reason,
        decision_summary=None if decision is None else decision.summary,
    )


def build_stored_artifacts(
    *,
    state: HarnessState,
    prior_artifacts: StoredHarnessArtifacts | None = None,
    turn_trace: HarnessTurnTrace | None = None,
) -> StoredHarnessArtifacts:
    """Return updated stored artifacts for one persisted snapshot."""
    trace = None if prior_artifacts is None else prior_artifacts.trace
    if turn_trace is not None:
        trace = append_turn_trace(
            trace,
            session_id=state.session.session_id,
            turn_trace=turn_trace,
            final_stop_reason=state.session.stop_reason,
        )
    elif trace is not None:
        trace = trace.model_copy(
            update={"final_stop_reason": state.session.stop_reason}
        )
    summary = build_session_summary(state, trace=trace)
    return StoredHarnessArtifacts(trace=trace, summary=summary)


def replay_session(snapshot: StoredHarnessState) -> HarnessReplayResult:
    """Reconstruct a deterministic replay view from canonical harness turns."""
    steps = [
        HarnessReplayStep(
            turn_index=turn.turn_index,
            selected_task_ids=list(turn.selected_task_ids),
            workflow_outcome_statuses=_canonical_workflow_outcome_statuses(turn),
            decision_action=None if turn.decision is None else turn.decision.action,
            decision_stop_reason=(
                None if turn.decision is None else turn.decision.stop_reason
            ),
            decision_summary=None if turn.decision is None else turn.decision.summary,
        )
        for turn in snapshot.state.turns
    ]
    return HarnessReplayResult(
        session_id=snapshot.session_id,
        mode=HarnessReplayMode.TRACE,
        steps=steps,
        final_stop_reason=snapshot.state.session.stop_reason,
        limitations=[],
    )


def _canonical_pending_approval_id(turn: HarnessTurn) -> str | None:
    if turn.pending_approval_request is not None:
        return turn.pending_approval_request.approval_id
    workflow_result = turn.workflow_result
    if workflow_result is None or not workflow_result.outcomes:
        return None
    approval_request = workflow_result.outcomes[-1].approval_request
    if approval_request is None:
        return None
    return approval_request.approval_id


def _canonical_workflow_outcome_statuses(
    turn: HarnessTurn,
) -> list[WorkflowInvocationStatus]:
    statuses = []
    if _should_prepend_pending_approval(turn):
        statuses.append(WorkflowInvocationStatus.APPROVAL_REQUESTED)
    workflow_result = turn.workflow_result
    if workflow_result is None:
        return statuses
    return [*statuses, *[outcome.status for outcome in workflow_result.outcomes]]


def _canonical_invocation_traces(turn: HarnessTurn) -> list[HarnessInvocationTrace]:
    traces: list[HarnessInvocationTrace] = []
    if _should_prepend_pending_approval(turn):
        approval_request = turn.pending_approval_request
        if approval_request is not None:
            traces.append(_build_pending_approval_trace(approval_request))
    workflow_result = turn.workflow_result
    if workflow_result is None:
        return traces
    return [
        *traces,
        *[_build_invocation_trace(outcome) for outcome in workflow_result.outcomes],
    ]


def _should_prepend_pending_approval(turn: HarnessTurn) -> bool:
    approval_request = turn.pending_approval_request
    if approval_request is None:
        return False
    workflow_result = turn.workflow_result
    if workflow_result is None:
        return True
    return not any(
        outcome.status is WorkflowInvocationStatus.APPROVAL_REQUESTED
        and outcome.approval_request is not None
        and outcome.approval_request.approval_id == approval_request.approval_id
        for outcome in workflow_result.outcomes
    )


def _build_pending_approval_trace(
    approval_request: ApprovalRequest,
) -> HarnessInvocationTrace:
    return HarnessInvocationTrace(
        invocation_index=approval_request.invocation_index,
        invocation_id=None,
        status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
        tool_name=approval_request.tool_name,
        tool_version=approval_request.tool_version,
        redacted_arguments={},
        policy_snapshot=HarnessPolicySnapshot(
            tool_name=approval_request.tool_name,
            tool_version=approval_request.tool_version,
            allowed=True,
            requires_approval=True,
            reason=approval_request.policy_reason,
            metadata=dict(approval_request.policy_metadata),
            source="approval_request",
        ),
        approval_id=approval_request.approval_id,
    )


def _build_invocation_trace(
    outcome: WorkflowInvocationOutcome,
) -> HarnessInvocationTrace:
    execution_record = _execution_record_from_outcome(outcome)
    approval_request = outcome.approval_request
    policy_snapshot = _build_policy_snapshot(
        outcome=outcome,
        execution_record=execution_record,
        approval_request=approval_request,
    )
    return HarnessInvocationTrace(
        invocation_index=outcome.invocation_index,
        invocation_id=_string_or_none(execution_record.get("invocation_id")),
        status=outcome.status,
        tool_name=_trace_tool_name(outcome=outcome, approval_request=approval_request),
        tool_version=_trace_tool_version(
            outcome=outcome,
            approval_request=approval_request,
        ),
        redacted_arguments=_redacted_arguments(execution_record=execution_record),
        policy_snapshot=policy_snapshot,
        ok=_bool_or_none(execution_record.get("ok")),
        error_code=_error_code_or_none(execution_record.get("error_code")),
        approval_id=None if approval_request is None else approval_request.approval_id,
        redaction=_record_object(execution_record.get("metadata")).get("redaction", {}),
        logs=_record_string_list(execution_record.get("logs")),
        artifacts=_record_string_list(execution_record.get("artifacts")),
    )


def _build_policy_snapshot(
    *,
    outcome: WorkflowInvocationOutcome,
    execution_record: dict[str, Any],
    approval_request: ApprovalRequest | None,
) -> HarnessPolicySnapshot | None:
    if approval_request is not None:
        return HarnessPolicySnapshot(
            tool_name=approval_request.tool_name,
            tool_version=approval_request.tool_version,
            allowed=True,
            requires_approval=True,
            reason=approval_request.policy_reason,
            metadata=dict(approval_request.policy_metadata),
            source="approval_request",
        )

    policy_payload = _record_object(execution_record.get("policy_decision"))
    if not policy_payload:
        return None
    reason = _string_or_none(policy_payload.get("reason")) or "policy evaluated"
    return HarnessPolicySnapshot(
        tool_name=_trace_tool_name(outcome=outcome, approval_request=None),
        tool_version=_trace_tool_version(outcome=outcome, approval_request=None),
        allowed=bool(policy_payload.get("allowed", False)),
        requires_approval=bool(policy_payload.get("requires_approval", False)),
        reason=reason,
        metadata=_record_object(policy_payload.get("metadata")),
        source="execution_record",
    )


def _execution_record_from_outcome(
    outcome: WorkflowInvocationOutcome,
) -> dict[str, Any]:
    if outcome.tool_result is None:
        return {}
    return _record_object(outcome.tool_result.metadata.get("execution_record"))


def _redacted_arguments(
    *,
    execution_record: dict[str, Any],
) -> dict[str, object]:
    redacted_input = execution_record.get("redacted_input")
    return redacted_input if isinstance(redacted_input, dict) else {}


def _trace_tool_name(
    *,
    outcome: WorkflowInvocationOutcome,
    approval_request: ApprovalRequest | None,
) -> str:
    if approval_request is not None:
        return approval_request.tool_name
    if outcome.tool_result is not None:
        return outcome.tool_result.tool_name
    return outcome.request.tool_name


def _trace_tool_version(
    *,
    outcome: WorkflowInvocationOutcome,
    approval_request: ApprovalRequest | None,
) -> str | None:
    if approval_request is not None:
        return approval_request.tool_version
    if outcome.tool_result is not None:
        return outcome.tool_result.tool_version
    return None


def _record_object(value: object) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _record_string_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [str(entry) for entry in value]


def _string_or_none(value: object) -> str | None:
    if isinstance(value, str) and value.strip():
        return value
    return None


def _bool_or_none(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _error_code_or_none(value: object) -> ErrorCode | None:
    if not isinstance(value, str):
        return None
    try:
        return ErrorCode(value)
    except ValueError:
        return None
