"""Direct tests for canonical harness API models."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import ValidationError

from llm_tools.harness_api import (
    BudgetPolicy,
    HarnessSession,
    HarnessState,
    HarnessStopReason,
    HarnessTurn,
    NoProgressSignal,
    NoProgressSignalKind,
    PendingApprovalRecord,
    TaskLifecycleStatus,
    TaskOrigin,
    TaskRecord,
    TurnDecision,
    TurnDecisionAction,
    VerificationEvidenceRecord,
    VerificationExpectation,
    VerificationOutcome,
    VerificationStatus,
)
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolContext, ToolInvocationRequest
from llm_tools.workflow_api.models import ApprovalRequest, WorkflowTurnResult


def _workflow_result() -> WorkflowTurnResult:
    return WorkflowTurnResult(
        parsed_response=ParsedModelResponse(final_response="done")
    )


def _root_task() -> TaskRecord:
    return TaskRecord(
        task_id="task-1",
        title="Root task",
        intent="Complete the root request.",
        origin=TaskOrigin.USER_REQUESTED,
    )


def _pending_approval_record() -> PendingApprovalRecord:
    parsed_response = ParsedModelResponse(
        invocations=[
            ToolInvocationRequest(
                tool_name="write_file", arguments={"path": "note.txt"}
            )
        ]
    )
    return PendingApprovalRecord(
        approval_request=ApprovalRequest(
            approval_id="approval-1",
            invocation_index=1,
            request=parsed_response.invocations[0],
            tool_name="write_file",
            tool_version="0.1.0",
            policy_reason="approval required",
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:05:00Z",
        ),
        parsed_response=parsed_response,
        base_context=ToolContext(invocation_id="turn-1"),
        pending_index=1,
    )


def test_enum_values_are_stable() -> None:
    assert [member.value for member in TaskLifecycleStatus] == [
        "pending",
        "in_progress",
        "blocked",
        "completed",
        "failed",
        "canceled",
        "superseded",
    ]
    assert [member.value for member in TaskOrigin] == [
        "user_requested",
        "derived",
    ]
    assert [member.value for member in VerificationStatus] == [
        "not_run",
        "passed",
        "failed",
        "inconclusive",
    ]
    assert [member.value for member in HarnessStopReason] == [
        "completed",
        "budget_exhausted",
        "verification_failed",
        "no_progress",
        "canceled",
        "error",
    ]
    assert [member.value for member in TurnDecisionAction] == [
        "continue",
        "select_tasks",
        "stop",
    ]


def test_budget_policy_requires_at_least_one_positive_limit() -> None:
    with pytest.raises(ValidationError, match="at least one limit"):
        BudgetPolicy()

    with pytest.raises(ValidationError):
        BudgetPolicy(max_turns=0)


def test_verification_models_validate_expected_shape() -> None:
    with pytest.raises(ValidationError, match="checked_at is required"):
        VerificationOutcome(status=VerificationStatus.PASSED)

    with pytest.raises(ValidationError, match="evidence_refs must be unique"):
        VerificationOutcome(evidence_refs=["artifact-1", "artifact-1"])

    with pytest.raises(ValidationError, match="description must not be empty"):
        VerificationExpectation(expectation_id="expectation-1", description=" ")


def test_task_record_validates_verification_expectations_and_completion_rule() -> None:
    with pytest.raises(
        ValidationError, match="verification expectation ids must be unique"
    ):
        TaskRecord(
            task_id="task-1",
            title="Task",
            intent="Do work",
            origin=TaskOrigin.USER_REQUESTED,
            verification_expectations=[
                VerificationExpectation(
                    expectation_id="expectation-1",
                    description="Check output",
                ),
                VerificationExpectation(
                    expectation_id="expectation-1",
                    description="Check logs",
                ),
            ],
        )

    with pytest.raises(
        ValidationError, match="required verification expectations must have"
    ):
        TaskRecord(
            task_id="task-1",
            title="Task",
            intent="Do work",
            origin=TaskOrigin.USER_REQUESTED,
            status=TaskLifecycleStatus.COMPLETED,
            verification_expectations=[
                VerificationExpectation(
                    expectation_id="expectation-1",
                    description="Run tests",
                    required_for_completion=True,
                )
            ],
            verification=VerificationOutcome(
                status=VerificationStatus.FAILED,
                checked_at="2026-01-01T00:00:10Z",
            ),
        )

    completed = TaskRecord(
        task_id="task-1",
        title="Task",
        intent="Do work",
        origin=TaskOrigin.USER_REQUESTED,
        status=TaskLifecycleStatus.COMPLETED,
        verification_expectations=[
            VerificationExpectation(
                expectation_id="expectation-1",
                description="Run tests",
                required_for_completion=True,
            )
        ],
        verification=VerificationOutcome(
            status=VerificationStatus.PASSED,
            checked_at="2026-01-01T00:00:10Z",
        ),
    )

    assert completed.verification.status is VerificationStatus.PASSED


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "task_id": " ",
                "title": "Task",
                "intent": "Do work",
                "origin": "user_requested",
            },
            "task_id must not be empty",
        ),
        (
            {
                "task_id": "task-1",
                "title": " ",
                "intent": "Do work",
                "origin": "user_requested",
            },
            "title must not be empty",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": " ",
                "origin": "user_requested",
            },
            "intent must not be empty",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": "Do work",
                "origin": "user_requested",
                "depends_on_task_ids": ["task-1"],
            },
            "must not depend on itself",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": "Do work",
                "origin": "derived",
            },
            "derived tasks must set parent_task_id",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": "Do work",
                "origin": "user_requested",
                "parent_task_id": "task-0",
            },
            "user_requested tasks must not set parent_task_id",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": "Do work",
                "origin": "user_requested",
                "status": "superseded",
            },
            "SUPERSEDED tasks must set superseded_by_task_id",
        ),
    ],
)
def test_task_record_validates_identity_and_dependencies(
    payload: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        TaskRecord.model_validate(payload)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": "Do work",
                "origin": "derived",
                "parent_task_id": " ",
            },
            "derived tasks must set parent_task_id",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": "Do work",
                "origin": "user_requested",
                "artifact_refs": ["a.txt", "a.txt"],
            },
            "artifact_refs must be unique",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": "Do work",
                "origin": "user_requested",
                "superseded_by_task_id": " ",
                "status": "superseded",
            },
            "superseded_by_task_id must not be empty",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": "Do work",
                "origin": "user_requested",
                "superseded_by_task_id": "task-1",
                "status": "superseded",
            },
            "must not supersede itself",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": "Do work",
                "origin": "user_requested",
                "superseded_by_task_id": "task-2",
            },
            "only allowed for SUPERSEDED tasks",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": "Do work",
                "origin": "user_requested",
                "started_at": " ",
            },
            "started_at must not be empty",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": "Do work",
                "origin": "user_requested",
                "finished_at": " ",
            },
            "finished_at must not be empty",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "intent": "Do work",
                "origin": "user_requested",
                "status_summary": " ",
            },
            "status_summary must not be empty",
        ),
    ],
)
def test_task_record_validates_extended_optional_fields(
    payload: dict[str, Any], message: str
) -> None:
    with pytest.raises(ValidationError, match=message):
        TaskRecord.model_validate(payload)


def test_turn_decision_requires_stop_reason_only_for_stop() -> None:
    with pytest.raises(ValidationError, match="stop_reason is required"):
        TurnDecision(action=TurnDecisionAction.STOP)

    with pytest.raises(ValidationError, match="only allowed for stop decisions"):
        TurnDecision(
            action=TurnDecisionAction.CONTINUE,
            stop_reason=HarnessStopReason.BUDGET_EXHAUSTED,
        )

    with pytest.raises(ValidationError, match="selected_task_ids must be unique"):
        TurnDecision(
            action=TurnDecisionAction.SELECT_TASKS,
            selected_task_ids=["task-1", "task-1"],
        )


def test_pending_approval_record_validates_blocked_invocation_reference() -> None:
    record = _pending_approval_record()
    assert record.pending_index == 1

    with pytest.raises(ValidationError, match="must match approval_request"):
        PendingApprovalRecord(
            approval_request=record.approval_request,
            parsed_response=record.parsed_response,
            base_context=record.base_context,
            pending_index=2,
        )

    with pytest.raises(ValidationError, match="must reference an invocation"):
        PendingApprovalRecord(
            approval_request=record.approval_request.model_copy(
                update={"invocation_index": 2}
            ),
            parsed_response=record.parsed_response,
            base_context=record.base_context,
            pending_index=2,
        )


def test_harness_turn_requires_end_time_once_decided() -> None:
    with pytest.raises(ValidationError, match="started_at must not be empty"):
        HarnessTurn(turn_index=1, started_at=" ")

    with pytest.raises(ValidationError, match="ended_at is required"):
        HarnessTurn(
            turn_index=1,
            started_at="2026-01-01T00:00:00Z",
            decision=TurnDecision(
                action=TurnDecisionAction.STOP,
                stop_reason=HarnessStopReason.ERROR,
            ),
        )


def test_harness_turn_no_progress_stop_requires_signal() -> None:
    with pytest.raises(ValidationError, match="must include at least one"):
        HarnessTurn(
            turn_index=1,
            started_at="2026-01-01T00:00:00Z",
            decision=TurnDecision(
                action=TurnDecisionAction.STOP,
                stop_reason=HarnessStopReason.NO_PROGRESS,
            ),
            ended_at="2026-01-01T00:00:05Z",
        )

    turn = HarnessTurn(
        turn_index=1,
        started_at="2026-01-01T00:00:00Z",
        decision=TurnDecision(
            action=TurnDecisionAction.STOP,
            stop_reason=HarnessStopReason.NO_PROGRESS,
        ),
        no_progress_signals=[
            NoProgressSignal(
                signal_id="signal-1",
                kind=NoProgressSignalKind.STALLED_TASK,
                task_id="task-1",
            )
        ],
        ended_at="2026-01-01T00:00:05Z",
    )

    assert turn.no_progress_signals[0].kind is NoProgressSignalKind.STALLED_TASK


def test_harness_session_rejects_invalid_terminal_shape() -> None:
    with pytest.raises(ValidationError, match="session_id must not be empty"):
        HarnessSession(
            session_id=" ",
            root_task_id="task-1",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at="2026-01-01T00:00:01Z",
        )

    with pytest.raises(ValidationError, match="root_task_id must not be empty"):
        HarnessSession(
            session_id="session-1",
            root_task_id=" ",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at="2026-01-01T00:00:01Z",
        )

    with pytest.raises(ValidationError, match="ended_at requires stop_reason"):
        HarnessSession(
            session_id="session-1",
            root_task_id="task-1",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at="2026-01-01T00:00:00Z",
            ended_at="2026-01-01T00:00:05Z",
        )

    with pytest.raises(ValidationError, match="started_at must not be empty"):
        HarnessSession(
            session_id="session-1",
            root_task_id="task-1",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at=" ",
        )


def test_harness_state_rejects_bad_cross_record_references() -> None:
    session = HarnessSession(
        session_id="session-1",
        root_task_id="task-1",
        budget_policy=BudgetPolicy(max_turns=3),
        started_at="2026-01-01T00:00:00Z",
        current_turn_index=1,
    )
    root_task = _root_task()

    with pytest.raises(ValidationError, match="schema_version"):
        HarnessState.model_validate(
            {
                "session": session.model_dump(mode="json"),
                "tasks": [
                    root_task.model_dump(mode="json"),
                    root_task.model_dump(mode="json"),
                ],
            }
        )

    with pytest.raises(ValidationError, match="task ids must be unique"):
        HarnessState(schema_version="2", session=session, tasks=[root_task, root_task])

    with pytest.raises(ValidationError, match="depends_on_task_ids must resolve"):
        HarnessState(
            schema_version="2",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-1",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
            ),
            tasks=[
                root_task,
                TaskRecord(
                    task_id="task-2",
                    title="Child task",
                    intent="Derived work",
                    origin=TaskOrigin.DERIVED,
                    parent_task_id="task-1",
                    depends_on_task_ids=["missing-task"],
                ),
            ],
        )

    with pytest.raises(ValidationError, match="root user_requested task"):
        HarnessState(
            schema_version="2",
            session=session,
            tasks=[
                root_task,
                TaskRecord(
                    task_id="task-2",
                    title="Other root",
                    intent="Another root",
                    origin=TaskOrigin.USER_REQUESTED,
                ),
            ],
            turns=[
                HarnessTurn(
                    turn_index=1,
                    started_at="2026-01-01T00:00:00Z",
                )
            ],
        )

    with pytest.raises(ValidationError, match="contiguous indices from 1"):
        HarnessState(
            schema_version="2",
            session=session,
            tasks=[root_task],
            turns=[HarnessTurn(turn_index=2, started_at="2026-01-01T00:00:00Z")],
        )

    with pytest.raises(ValidationError, match="dependency graph must be acyclic"):
        HarnessState(
            schema_version="2",
            session=session,
            tasks=[
                root_task,
                TaskRecord(
                    task_id="task-2",
                    title="Child task",
                    intent="Derived work",
                    origin=TaskOrigin.DERIVED,
                    parent_task_id="task-1",
                    depends_on_task_ids=["task-3"],
                ),
                TaskRecord(
                    task_id="task-3",
                    title="Sibling task",
                    intent="More derived work",
                    origin=TaskOrigin.DERIVED,
                    parent_task_id="task-1",
                    depends_on_task_ids=["task-2"],
                ),
            ],
            turns=[HarnessTurn(turn_index=1, started_at="2026-01-01T00:00:00Z")],
        )

    with pytest.raises(
        ValidationError, match="superseded_by_task_id graph must be acyclic"
    ):
        HarnessState(
            schema_version="2",
            session=session,
            tasks=[
                TaskRecord(
                    task_id="task-1",
                    title="Root task",
                    intent="Complete the root request.",
                    origin=TaskOrigin.USER_REQUESTED,
                    status=TaskLifecycleStatus.SUPERSEDED,
                    superseded_by_task_id="task-2",
                ),
                TaskRecord(
                    task_id="task-2",
                    title="Replacement task",
                    intent="Replacement",
                    origin=TaskOrigin.DERIVED,
                    parent_task_id="task-1",
                    status=TaskLifecycleStatus.SUPERSEDED,
                    superseded_by_task_id="task-1",
                ),
            ],
            turns=[HarnessTurn(turn_index=1, started_at="2026-01-01T00:00:00Z")],
        )

    with pytest.raises(ValidationError, match="parent_task_id graph must be acyclic"):
        HarnessState(
            schema_version="2",
            session=session,
            tasks=[
                TaskRecord(
                    task_id="task-1",
                    title="Root task",
                    intent="Complete the root request.",
                    origin=TaskOrigin.USER_REQUESTED,
                ),
                TaskRecord(
                    task_id="task-2",
                    title="Child task",
                    intent="Derived work",
                    origin=TaskOrigin.DERIVED,
                    parent_task_id="task-3",
                ),
                TaskRecord(
                    task_id="task-3",
                    title="Grandchild task",
                    intent="More derived work",
                    origin=TaskOrigin.DERIVED,
                    parent_task_id="task-2",
                ),
            ],
            turns=[HarnessTurn(turn_index=1, started_at="2026-01-01T00:00:00Z")],
        )

    with pytest.raises(ValidationError, match="selected_task_ids must resolve"):
        HarnessState(
            schema_version="2",
            session=session,
            tasks=[root_task],
            turns=[
                HarnessTurn(
                    turn_index=1,
                    started_at="2026-01-01T00:00:00Z",
                    decision=TurnDecision(
                        action=TurnDecisionAction.SELECT_TASKS,
                        selected_task_ids=["task-2"],
                    ),
                    ended_at="2026-01-01T00:00:05Z",
                )
            ],
        )

    with pytest.raises(
        ValidationError, match="incomplete turn is only allowed at the tail"
    ):
        HarnessState(
            schema_version="2",
            session=session,
            tasks=[root_task],
            turns=[
                HarnessTurn(turn_index=1, started_at="2026-01-01T00:00:00Z"),
                HarnessTurn(
                    turn_index=2,
                    started_at="2026-01-01T00:00:10Z",
                    decision=TurnDecision(action=TurnDecisionAction.CONTINUE),
                    ended_at="2026-01-01T00:00:20Z",
                ),
            ],
        )

    with pytest.raises(
        ValidationError, match="current_turn_index must equal len\\(turns\\)"
    ):
        HarnessState(
            schema_version="2",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-1",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
                current_turn_index=1,
            ),
            tasks=[root_task],
        )

    with pytest.raises(
        ValidationError, match="Ended sessions must not retain incomplete turns"
    ):
        HarnessState(
            schema_version="2",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-1",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
                current_turn_index=1,
                ended_at="2026-01-01T00:00:10Z",
                stop_reason=HarnessStopReason.ERROR,
            ),
            tasks=[root_task],
            turns=[HarnessTurn(turn_index=1, started_at="2026-01-01T00:00:00Z")],
        )

    with pytest.raises(ValidationError, match="evidence_refs must resolve"):
        HarnessState(
            schema_version="2",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-1",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
            ),
            tasks=[
                TaskRecord(
                    task_id="task-1",
                    title="Root task",
                    intent="Complete the root request.",
                    origin=TaskOrigin.USER_REQUESTED,
                    verification=VerificationOutcome(evidence_refs=["evidence-1"]),
                )
            ],
        )

    with pytest.raises(ValidationError, match="evidence task ids must resolve"):
        HarnessState(
            schema_version="2",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-1",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
            ),
            tasks=[root_task],
            verification_evidence=[
                VerificationEvidenceRecord(
                    evidence_id="evidence-1",
                    task_id="task-2",
                )
            ],
        )

    with pytest.raises(ValidationError, match="owned by a different task"):
        HarnessState(
            schema_version="2",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-1",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
            ),
            tasks=[
                TaskRecord(
                    task_id="task-1",
                    title="Root task",
                    intent="Complete the root request.",
                    origin=TaskOrigin.USER_REQUESTED,
                    verification=VerificationOutcome(evidence_refs=["evidence-1"]),
                ),
                TaskRecord(
                    task_id="task-2",
                    title="Sibling task",
                    intent="Sibling work",
                    origin=TaskOrigin.DERIVED,
                    parent_task_id="task-1",
                ),
            ],
            verification_evidence=[
                VerificationEvidenceRecord(
                    evidence_id="evidence-1",
                    task_id="task-2",
                )
            ],
        )

    with pytest.raises(ValidationError, match="signal task ids must resolve"):
        HarnessState(
            schema_version="2",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-1",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
                current_turn_index=1,
            ),
            tasks=[root_task],
            turns=[
                HarnessTurn(
                    turn_index=1,
                    started_at="2026-01-01T00:00:00Z",
                    decision=TurnDecision(action=TurnDecisionAction.CONTINUE),
                    no_progress_signals=[
                        NoProgressSignal(
                            signal_id="signal-1",
                            kind=NoProgressSignalKind.REPEATED_RETRY,
                            task_id="task-2",
                        )
                    ],
                    ended_at="2026-01-01T00:00:05Z",
                )
            ],
        )


def test_harness_state_validates_additional_cross_record_integrity() -> None:
    session = HarnessSession(
        session_id="session-1",
        root_task_id="task-1",
        budget_policy=BudgetPolicy(max_turns=3),
        started_at="2026-01-01T00:00:00Z",
        current_turn_index=1,
    )
    root_task = _root_task()

    with pytest.raises(
        ValidationError, match="root_task_id must resolve to a known task"
    ):
        HarnessState(
            schema_version="2",
            session=session.model_copy(update={"root_task_id": "missing"}),
            tasks=[root_task],
            turns=[HarnessTurn(turn_index=1, started_at="2026-01-01T00:00:00Z")],
        )

    with pytest.raises(
        ValidationError,
        match="root_task_id must resolve to the root user_requested task",
    ):
        HarnessState(
            schema_version="2",
            session=session.model_copy(update={"root_task_id": "task-2"}),
            tasks=[
                root_task,
                TaskRecord(
                    task_id="task-2",
                    title="Child task",
                    intent="Derived work",
                    origin=TaskOrigin.DERIVED,
                    parent_task_id="task-1",
                ),
            ],
            turns=[HarnessTurn(turn_index=1, started_at="2026-01-01T00:00:00Z")],
        )

    with pytest.raises(
        ValidationError, match="parent_task_id must resolve to a known task"
    ):
        HarnessState(
            schema_version="2",
            session=session,
            tasks=[
                root_task,
                TaskRecord(
                    task_id="task-2",
                    title="Child task",
                    intent="Derived work",
                    origin=TaskOrigin.DERIVED,
                    parent_task_id="missing",
                ),
            ],
            turns=[HarnessTurn(turn_index=1, started_at="2026-01-01T00:00:00Z")],
        )

    with pytest.raises(
        ValidationError, match="superseded_by_task_id must resolve to a known task"
    ):
        HarnessState(
            schema_version="2",
            session=session,
            tasks=[
                TaskRecord(
                    task_id="task-1",
                    title="Root task",
                    intent="Complete the root request.",
                    origin=TaskOrigin.USER_REQUESTED,
                    status=TaskLifecycleStatus.SUPERSEDED,
                    superseded_by_task_id="missing",
                )
            ],
            turns=[HarnessTurn(turn_index=1, started_at="2026-01-01T00:00:00Z")],
        )

    with pytest.raises(
        ValidationError, match="may contain at most one incomplete turn"
    ):
        HarnessState(
            schema_version="2",
            session=session.model_copy(update={"current_turn_index": 2}),
            tasks=[root_task],
            turns=[
                HarnessTurn(turn_index=1, started_at="2026-01-01T00:00:00Z"),
                HarnessTurn(turn_index=2, started_at="2026-01-01T00:00:10Z"),
            ],
        )

    approval = _pending_approval_record()
    with pytest.raises(ValidationError, match="pending approval ids must be unique"):
        HarnessState(
            schema_version="2",
            session=session,
            tasks=[root_task],
            turns=[HarnessTurn(turn_index=1, started_at="2026-01-01T00:00:00Z")],
            pending_approvals=[approval, approval],
        )

    with pytest.raises(ValidationError, match="at most one pending approval"):
        HarnessState(
            schema_version="2",
            session=session,
            tasks=[root_task],
            turns=[HarnessTurn(turn_index=1, started_at="2026-01-01T00:00:00Z")],
            pending_approvals=[
                approval,
                _pending_approval_record().model_copy(
                    update={
                        "approval_request": approval.approval_request.model_copy(
                            update={"approval_id": "approval-2"}
                        )
                    }
                ),
            ],
        )

    with pytest.raises(
        ValidationError, match="Ended sessions must not retain pending approvals"
    ):
        HarnessState(
            schema_version="2",
            session=session.model_copy(
                update={
                    "ended_at": "2026-01-01T00:00:10Z",
                    "stop_reason": HarnessStopReason.ERROR,
                }
            ),
            tasks=[root_task],
            turns=[
                HarnessTurn(
                    turn_index=1,
                    started_at="2026-01-01T00:00:00Z",
                    decision=TurnDecision(action=TurnDecisionAction.CONTINUE),
                    ended_at="2026-01-01T00:00:05Z",
                )
            ],
            pending_approvals=[approval],
        )


def test_harness_state_round_trips_with_richer_task_state() -> None:
    state = HarnessState(
        schema_version="2",
        session=HarnessSession(
            session_id="session-1",
            root_task_id="task-1",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at="2026-01-01T00:00:00Z",
            current_turn_index=1,
        ),
        tasks=[
            _root_task(),
            TaskRecord(
                task_id="task-2",
                title="Derived task",
                intent="Follow-up work",
                origin=TaskOrigin.DERIVED,
                parent_task_id="task-1",
                depends_on_task_ids=["task-1"],
                verification_expectations=[
                    VerificationExpectation(
                        expectation_id="expectation-1",
                        description="Run targeted tests.",
                    )
                ],
                artifact_refs=["artifacts/log.txt"],
                status=TaskLifecycleStatus.BLOCKED,
                started_at="2026-01-01T00:00:05Z",
                status_summary="Waiting on a dependency.",
            ),
        ],
        turns=[
            HarnessTurn(
                turn_index=1,
                started_at="2026-01-01T00:00:00Z",
                workflow_result=_workflow_result(),
                decision=TurnDecision(action=TurnDecisionAction.CONTINUE),
                ended_at="2026-01-01T00:00:10Z",
            )
        ],
    )

    reloaded = HarnessState.model_validate_json(state.model_dump_json())
    assert reloaded == state


def test_harness_state_schema_requires_schema_version() -> None:
    schema = HarnessState.model_json_schema()

    assert schema["required"] == ["schema_version", "session"]
    assert schema["properties"]["schema_version"]["minLength"] == 1
    assert schema["properties"]["session"] == {"$ref": "#/$defs/HarnessSession"}
    assert schema["properties"]["tasks"]["items"] == {"$ref": "#/$defs/TaskRecord"}
    assert schema["$defs"]["TaskOrigin"]["enum"] == ["user_requested", "derived"]
    assert schema["$defs"]["TurnDecisionAction"]["enum"] == [
        "continue",
        "select_tasks",
        "stop",
    ]
