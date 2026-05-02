"""Typed rehydration and resume inspection for persisted harness sessions."""

from __future__ import annotations

from datetime import UTC, datetime

from pydantic import ValidationError

from llm_tools.harness_api.models import (
    HarnessState,
    HarnessTurn,
    PendingApprovalRecord,
    TaskLifecycleStatus,
)
from llm_tools.harness_api.resume_models import (
    ResumedHarnessSession as ResumedHarnessSession,
)
from llm_tools.harness_api.resume_models import (
    ResumeDisposition as ResumeDisposition,
)
from llm_tools.harness_api.resume_models import (
    ResumeIssue as ResumeIssue,
)
from llm_tools.harness_api.store import (
    SUPPORTED_HARNESS_STATE_SCHEMA_VERSIONS,
    HarnessStateStore,
    StoredHarnessState,
)
from llm_tools.workflow_api.models import WorkflowInvocationStatus

_NON_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskLifecycleStatus.PENDING,
        TaskLifecycleStatus.IN_PROGRESS,
        TaskLifecycleStatus.BLOCKED,
    }
)


def load_resumed_session(
    store: HarnessStateStore,
    session_id: str,
    *,
    now: datetime | None = None,
) -> ResumedHarnessSession | None:
    """Load one persisted session from the store and classify its resume state."""
    snapshot = store.load_session(session_id)
    if snapshot is None:
        return None
    return resume_session(snapshot, now=now)


def resume_session(
    snapshot: StoredHarnessState,
    *,
    now: datetime | None = None,
) -> ResumedHarnessSession:
    """Classify persisted state for execution resume or operator intervention."""
    state = snapshot.state
    active_tasks = [
        task for task in state.tasks if task.status in _NON_TERMINAL_TASK_STATUSES
    ]

    if state.schema_version not in SUPPORTED_HARNESS_STATE_SCHEMA_VERSIONS:
        return ResumedHarnessSession(
            snapshot=snapshot,
            disposition=ResumeDisposition.INCOMPATIBLE_SCHEMA,
            active_tasks=active_tasks,
            issues=[
                ResumeIssue(
                    code="unsupported_schema_version",
                    message=(
                        "Unsupported harness state schema_version: "
                        f"'{state.schema_version}'."
                    ),
                )
            ],
        )

    try:
        state = HarnessState.model_validate(state.model_dump(mode="json"))
    except ValidationError as exc:
        return ResumedHarnessSession(
            snapshot=snapshot,
            disposition=ResumeDisposition.CORRUPT,
            active_tasks=active_tasks,
            issues=[
                ResumeIssue(code="invalid_state", message=_validation_message(exc))
            ],
        )

    snapshot = snapshot.model_copy(update={"state": state})
    active_tasks = [
        task for task in state.tasks if task.status in _NON_TERMINAL_TASK_STATUSES
    ]

    if state.session.ended_at is not None:
        return ResumedHarnessSession(
            snapshot=snapshot,
            disposition=ResumeDisposition.TERMINAL,
            active_tasks=active_tasks,
        )

    incomplete_turn = next(
        (turn for turn in state.turns if turn.decision is None), None
    )
    if incomplete_turn is None:
        if state.pending_approvals:
            return ResumedHarnessSession(
                snapshot=snapshot,
                disposition=ResumeDisposition.CORRUPT,
                active_tasks=active_tasks,
                issues=[
                    ResumeIssue(
                        code="dangling_pending_approval",
                        message=("Pending approvals require an incomplete tail turn."),
                    )
                ],
            )
        return ResumedHarnessSession(
            snapshot=snapshot,
            disposition=ResumeDisposition.RUNNABLE,
            active_tasks=active_tasks,
        )

    if not state.pending_approvals:
        workflow_result = incomplete_turn.workflow_result
        if (
            workflow_result is not None
            and workflow_result.outcomes
            and workflow_result.outcomes[-1].status
            is WorkflowInvocationStatus.APPROVAL_REQUESTED
        ):
            return ResumedHarnessSession(
                snapshot=snapshot,
                disposition=ResumeDisposition.CORRUPT,
                active_tasks=active_tasks,
                incomplete_turn=incomplete_turn,
                issues=[
                    ResumeIssue(
                        code="missing_pending_approval",
                        message=(
                            "Incomplete turns ending in approval_requested must "
                            "persist a matching pending approval."
                        ),
                    )
                ],
            )
        return ResumedHarnessSession(
            snapshot=snapshot,
            disposition=ResumeDisposition.INTERRUPTED,
            active_tasks=active_tasks,
            incomplete_turn=incomplete_turn,
            issues=[
                ResumeIssue(
                    code="interrupted_turn",
                    message=(
                        "Incomplete non-approval turn detected. Operator review is "
                        "required before replaying the interrupted turn."
                    ),
                )
            ],
        )

    if len(state.pending_approvals) != 1:
        return ResumedHarnessSession(
            snapshot=snapshot,
            disposition=ResumeDisposition.CORRUPT,
            active_tasks=active_tasks,
            incomplete_turn=incomplete_turn,
            issues=[
                ResumeIssue(
                    code="missing_pending_approval",
                    message=(
                        "Incomplete turns must persist exactly one matching "
                        "pending approval."
                    ),
                )
            ],
        )

    pending_approval = state.pending_approvals[0]
    mismatch = _pending_approval_mismatch(
        incomplete_turn=incomplete_turn,
        pending_approval=pending_approval,
    )
    if mismatch is not None:
        return ResumedHarnessSession(
            snapshot=snapshot,
            disposition=ResumeDisposition.CORRUPT,
            active_tasks=active_tasks,
            incomplete_turn=incomplete_turn,
            pending_approval=pending_approval,
            issues=[ResumeIssue(code="approval_turn_mismatch", message=mismatch)],
        )

    current_time = now or datetime.now(UTC)
    expires_at = _parse_timestamp(pending_approval.approval_request.expires_at)
    disposition = (
        ResumeDisposition.APPROVAL_EXPIRED
        if expires_at <= current_time
        else ResumeDisposition.WAITING_FOR_APPROVAL
    )
    return ResumedHarnessSession(
        snapshot=snapshot,
        disposition=disposition,
        active_tasks=active_tasks,
        incomplete_turn=incomplete_turn,
        pending_approval=pending_approval,
    )


def _pending_approval_mismatch(
    *,
    incomplete_turn: HarnessTurn,
    pending_approval: PendingApprovalRecord,
) -> str | None:
    workflow_result = incomplete_turn.workflow_result
    if workflow_result is None:
        return "Incomplete turns with pending approval must persist workflow_result."

    if workflow_result.parsed_response != pending_approval.parsed_response:
        return (
            "Pending approval parsed_response must match the incomplete turn "
            "workflow_result."
        )

    if not workflow_result.outcomes:
        return "Pending approval turns must persist at least one workflow outcome."

    approval_outcome = workflow_result.outcomes[-1]
    if approval_outcome.status is not WorkflowInvocationStatus.APPROVAL_REQUESTED:
        return "Incomplete turns must end with an approval_requested workflow outcome."

    approval_request = approval_outcome.approval_request
    if approval_request is None:
        return "approval_requested workflow outcomes must include approval_request."

    if approval_request.approval_id != pending_approval.approval_request.approval_id:
        return "Pending approval id must match the incomplete turn approval_request id."

    if approval_request.invocation_index != pending_approval.pending_index:
        return "Pending approval pending_index must match the blocked invocation index."

    return None


def _parse_timestamp(value: str) -> datetime:
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    parsed = datetime.fromisoformat(normalized)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _validation_message(exc: ValidationError) -> str:
    return "; ".join(error["msg"] for error in exc.errors())
