"""Resume-time validation and classification tests for persisted harness state."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from llm_tools.harness_api import (
    CURRENT_HARNESS_STATE_SCHEMA_VERSION,
    BudgetPolicy,
    HarnessSession,
    HarnessState,
    HarnessStopReason,
    HarnessTurn,
    InMemoryHarnessStateStore,
    PendingApprovalRecord,
    ResumeDisposition,
    StoredHarnessState,
    TaskLifecycleStatus,
    TaskOrigin,
    TaskRecord,
    load_resumed_session,
    resume_session,
)
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolContext, ToolInvocationRequest, ToolResult
from llm_tools.workflow_api.models import (
    ApprovalRequest,
    WorkflowInvocationOutcome,
    WorkflowInvocationStatus,
    WorkflowTurnResult,
)


def _root_task(status: TaskLifecycleStatus = TaskLifecycleStatus.PENDING) -> TaskRecord:
    return TaskRecord(
        task_id="task-1",
        title="Root task",
        intent="Complete the request.",
        origin=TaskOrigin.USER_REQUESTED,
        status=status,
    )


def _runnable_snapshot() -> StoredHarnessState:
    state = HarnessState(
        schema_version=CURRENT_HARNESS_STATE_SCHEMA_VERSION,
        session=HarnessSession(
            session_id="session-1",
            root_task_id="task-1",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at="2026-01-01T00:00:00Z",
        ),
        tasks=[_root_task()],
    )
    return StoredHarnessState(session_id="session-1", revision="1", state=state)


def _approval_snapshot(*, expires_at: str) -> StoredHarnessState:
    parsed_response = ParsedModelResponse(
        invocations=[
            ToolInvocationRequest(
                tool_name="write_file",
                arguments={"path": "note.txt", "content": "hello"},
            )
        ]
    )
    approval_request = ApprovalRequest(
        approval_id="approval-1",
        invocation_index=1,
        request=parsed_response.invocations[0],
        tool_name="write_file",
        tool_version="0.1.0",
        policy_reason="requires approval",
        requested_at="2026-01-01T00:00:00Z",
        expires_at=expires_at,
    )
    pending_approval = PendingApprovalRecord(
        approval_request=approval_request,
        parsed_response=parsed_response,
        base_context=ToolContext(invocation_id="turn-1"),
        pending_index=1,
    )
    turn = HarnessTurn(
        turn_index=1,
        started_at="2026-01-01T00:00:00Z",
        workflow_result=WorkflowTurnResult(
            parsed_response=parsed_response,
            outcomes=[
                WorkflowInvocationOutcome(
                    invocation_index=1,
                    request=parsed_response.invocations[0],
                    status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                    approval_request=approval_request,
                )
            ],
        ),
    )
    state = HarnessState(
        schema_version=CURRENT_HARNESS_STATE_SCHEMA_VERSION,
        session=HarnessSession(
            session_id="session-1",
            root_task_id="task-1",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at="2026-01-01T00:00:00Z",
            current_turn_index=1,
        ),
        tasks=[_root_task(status=TaskLifecycleStatus.IN_PROGRESS)],
        turns=[turn],
        pending_approvals=[pending_approval],
    )
    return StoredHarnessState(session_id="session-1", revision="1", state=state)


def test_resume_classifies_runnable_sessions() -> None:
    resumed = resume_session(_runnable_snapshot())

    assert resumed.disposition is ResumeDisposition.RUNNABLE
    assert [task.task_id for task in resumed.active_tasks] == ["task-1"]
    assert resumed.pending_approval is None


def test_resume_classifies_waiting_for_approval_sessions() -> None:
    future_time = datetime.now(UTC) + timedelta(minutes=5)
    snapshot = _approval_snapshot(
        expires_at=future_time.isoformat().replace("+00:00", "Z")
    )

    resumed = resume_session(snapshot, now=datetime.now(UTC))

    assert resumed.disposition is ResumeDisposition.WAITING_FOR_APPROVAL
    assert resumed.pending_approval is not None
    assert resumed.incomplete_turn is not None


def test_resume_surfaces_expired_approvals() -> None:
    expired_time = datetime.now(UTC) - timedelta(minutes=5)
    snapshot = _approval_snapshot(
        expires_at=expired_time.isoformat().replace("+00:00", "Z")
    )

    resumed = resume_session(snapshot, now=datetime.now(UTC))

    assert resumed.disposition is ResumeDisposition.APPROVAL_EXPIRED


def test_resume_rejects_incompatible_schema_versions() -> None:
    snapshot = _runnable_snapshot()
    snapshot = snapshot.model_copy(
        update={"state": snapshot.state.model_copy(update={"schema_version": "99"})}
    )

    resumed = resume_session(snapshot)

    assert resumed.disposition is ResumeDisposition.INCOMPATIBLE_SCHEMA
    assert resumed.issues[0].code == "unsupported_schema_version"


def test_resume_flags_partial_incomplete_turns_without_pending_approval() -> None:
    snapshot = _approval_snapshot(expires_at="2026-01-01T00:05:00Z")
    broken_state = snapshot.state.model_copy(update={"pending_approvals": []})

    resumed = resume_session(
        snapshot.model_copy(update={"state": broken_state}),
        now=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
    )

    assert resumed.disposition is ResumeDisposition.CORRUPT
    assert resumed.issues[0].code == "missing_pending_approval"


def test_resume_classifies_incomplete_non_approval_turns_as_interrupted() -> None:
    snapshot = _runnable_snapshot().model_copy(
        update={
            "state": _runnable_snapshot().state.model_copy(
                update={
                    "session": _runnable_snapshot().state.session.model_copy(
                        update={"current_turn_index": 1}
                    ),
                    "turns": [
                        HarnessTurn(
                            turn_index=1,
                            started_at="2026-01-01T00:00:00Z",
                            selected_task_ids=["task-1"],
                        )
                    ],
                },
                deep=True,
            )
        },
        deep=True,
    )

    resumed = resume_session(snapshot)

    assert resumed.disposition is ResumeDisposition.INTERRUPTED
    assert resumed.incomplete_turn is not None
    assert resumed.issues[0].code == "interrupted_turn"


def test_resume_can_load_from_store() -> None:
    store = InMemoryHarnessStateStore()
    snapshot = store.save_session(_runnable_snapshot().state)

    resumed = load_resumed_session(store, snapshot.session_id)

    assert resumed is not None
    assert resumed.snapshot.session_id == "session-1"
    assert resumed.disposition is ResumeDisposition.RUNNABLE


def test_load_resumed_session_returns_none_for_missing_snapshot() -> None:
    store = InMemoryHarnessStateStore()

    assert load_resumed_session(store, "missing-session") is None


def test_resumed_harness_session_requires_approval_payload_for_waiting_states() -> None:
    with pytest.raises(ValueError, match="Approval resume dispositions require"):
        from llm_tools.harness_api.resume import ResumedHarnessSession

        ResumedHarnessSession(
            snapshot=_runnable_snapshot(),
            disposition=ResumeDisposition.WAITING_FOR_APPROVAL,
        )


def test_resume_classifies_terminal_sessions() -> None:
    snapshot = _runnable_snapshot().model_copy(
        update={
            "state": _runnable_snapshot().state.model_copy(
                update={
                    "session": _runnable_snapshot().state.session.model_copy(
                        update={
                            "ended_at": "2026-01-01T00:01:00Z",
                            "stop_reason": HarnessStopReason.ERROR,
                        }
                    )
                }
            )
        }
    )

    resumed = resume_session(snapshot)

    assert resumed.disposition is ResumeDisposition.TERMINAL


def test_resume_classifies_corrupt_state_when_revalidation_fails() -> None:
    snapshot = _runnable_snapshot().model_copy(
        update={
            "state": _runnable_snapshot().state.model_copy(
                update={
                    "session": _runnable_snapshot().state.session.model_copy(
                        update={"root_task_id": "missing"}
                    )
                }
            )
        }
    )

    resumed = resume_session(snapshot)

    assert resumed.disposition is ResumeDisposition.CORRUPT
    assert resumed.issues[0].code == "invalid_state"


def test_resume_flags_dangling_pending_approval_without_incomplete_turn() -> None:
    snapshot = _runnable_snapshot().model_copy(
        update={
            "state": _runnable_snapshot().state.model_copy(
                update={
                    "pending_approvals": [
                        _approval_snapshot(
                            expires_at="2026-01-01T00:05:00Z"
                        ).state.pending_approvals[0]
                    ]
                }
            )
        }
    )

    resumed = resume_session(snapshot)

    assert resumed.disposition is ResumeDisposition.CORRUPT
    assert resumed.issues[0].code == "dangling_pending_approval"


@pytest.mark.parametrize(
    "scenario",
    [
        "missing_workflow_result",
        "parsed_response",
        "no_outcomes",
        "wrong_status",
        "missing_request",
        "approval_id",
        "pending_index",
    ],
)
def test_resume_flags_pending_approval_mismatches(scenario: str) -> None:
    snapshot = _approval_snapshot(expires_at="2026-01-01T00:05:00")
    state = snapshot.state
    turn = state.turns[0]
    pending = state.pending_approvals[0]
    turn_update: dict[str, object] = {}
    approval_update: dict[str, object] = {}
    message = ""

    if scenario == "missing_workflow_result":
        turn_update = {"workflow_result": None}
        message = "must persist workflow_result"
    elif scenario == "parsed_response":
        other_response = ParsedModelResponse(
            invocations=[
                ToolInvocationRequest(
                    tool_name="read_file", arguments={"path": "other.txt"}
                )
            ]
        )
        turn_update = {
            "workflow_result": WorkflowTurnResult(
                parsed_response=other_response,
                outcomes=[
                    WorkflowInvocationOutcome(
                        invocation_index=1,
                        request=other_response.invocations[0],
                        status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                        approval_request=pending.approval_request,
                    )
                ],
            )
        }
        message = "parsed_response must match"
    elif scenario == "no_outcomes":
        turn_update = {
            "workflow_result": WorkflowTurnResult(
                parsed_response=pending.parsed_response,
                outcomes=[],
            )
        }
        message = "must persist at least one workflow outcome"
    elif scenario == "wrong_status":
        turn_update = {
            "workflow_result": WorkflowTurnResult(
                parsed_response=pending.parsed_response,
                outcomes=[
                    WorkflowInvocationOutcome(
                        invocation_index=1,
                        request=pending.parsed_response.invocations[0],
                        status=WorkflowInvocationStatus.EXECUTED,
                        tool_result=ToolResult(
                            ok=True,
                            tool_name="write_file",
                            tool_version="0.1.0",
                            output={"status": "ok"},
                        ),
                    )
                ],
            )
        }
        message = "must end with an approval_requested"
    elif scenario == "missing_request":
        turn_update = {
            "workflow_result": WorkflowTurnResult.model_construct(
                parsed_response=pending.parsed_response,
                outcomes=[
                    WorkflowInvocationOutcome.model_construct(
                        invocation_index=1,
                        request=pending.parsed_response.invocations[0],
                        status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                        tool_result=None,
                        approval_request=None,
                    )
                ],
            )
        }
        message = "must include approval_request"
    elif scenario == "approval_id":
        approval_update = {
            "approval_request": pending.approval_request.model_copy(
                update={"approval_id": "other-approval"}
            )
        }
        message = "approval_request id"
    elif scenario == "pending_index":
        approval_update = {"pending_index": 2}
        message = "pending_index must match approval_request.invocation_index"

    broken_state = state.model_copy(
        update={
            "turns": [turn.model_copy(update=turn_update)],
            "pending_approvals": [pending.model_copy(update=approval_update)],
        }
    )

    resumed = resume_session(
        snapshot.model_copy(update={"state": broken_state}),
        now=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
    )

    assert resumed.disposition is ResumeDisposition.CORRUPT
    assert message in resumed.issues[0].message
