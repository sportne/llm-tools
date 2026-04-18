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
    TaskLifecycleStatus,
    TaskRecord,
    TurnDecision,
    TurnDecisionAction,
    VerificationEvidenceRecord,
    VerificationExpectation,
    VerificationOutcome,
    VerificationStatus,
)
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.workflow_api.models import WorkflowTurnResult


def _workflow_result() -> WorkflowTurnResult:
    return WorkflowTurnResult(
        parsed_response=ParsedModelResponse(final_response="done")
    )


def test_enum_values_are_stable() -> None:
    assert [member.value for member in TaskLifecycleStatus] == [
        "pending",
        "in_progress",
        "blocked",
        "completed",
        "failed",
        "canceled",
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


def test_verification_outcome_requires_checked_at_and_unique_evidence_refs() -> None:
    with pytest.raises(ValidationError, match="checked_at is required"):
        VerificationOutcome(status=VerificationStatus.PASSED)

    with pytest.raises(ValidationError, match="evidence_refs must be unique"):
        VerificationOutcome(evidence_refs=["artifact-1", "artifact-1"])


def test_task_record_validates_verification_expectations_and_completion_rule() -> None:
    with pytest.raises(
        ValidationError, match="verification expectation ids must be unique"
    ):
        TaskRecord(
            task_id="task-1",
            title="Task",
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
        ({"task_id": " ", "title": "Task"}, "task_id must not be empty"),
        ({"task_id": "task-1", "title": " "}, "title must not be empty"),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "depends_on_task_ids": ["task-1"],
            },
            "must not depend on itself",
        ),
        (
            {
                "task_id": "task-1",
                "title": "Task",
                "depends_on_task_ids": ["task-2", "task-2"],
            },
            "dependency ids must be unique",
        ),
    ],
)
def test_task_record_validates_identity_and_dependencies(
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


def test_harness_turn_requires_end_time_once_decided() -> None:
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


def test_harness_state_rejects_bad_cross_record_references() -> None:
    session = HarnessSession(
        session_id="session-1",
        root_task_id="task-1",
        budget_policy=BudgetPolicy(max_turns=3),
        started_at="2026-01-01T00:00:00Z",
        current_turn_index=1,
    )
    task = TaskRecord(task_id="task-1", title="Root task")

    with pytest.raises(ValidationError, match="schema_version"):
        HarnessState.model_validate(
            {
                "session": session.model_dump(mode="json"),
                "tasks": [
                    task.model_dump(mode="json"),
                    task.model_dump(mode="json"),
                ],
            }
        )

    with pytest.raises(ValidationError, match="task ids must be unique"):
        HarnessState(schema_version="1", session=session, tasks=[task, task])

    with pytest.raises(ValidationError, match="depends_on_task_ids must resolve"):
        HarnessState(
            schema_version="1",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-1",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
            ),
            tasks=[
                task,
                TaskRecord(
                    task_id="task-2",
                    title="Child task",
                    depends_on_task_ids=["missing-task"],
                ),
            ],
        )

    with pytest.raises(ValidationError, match="contiguous indices from 1"):
        HarnessState(
            schema_version="1",
            session=session,
            tasks=[task],
            turns=[
                HarnessTurn(
                    turn_index=2,
                    started_at="2026-01-01T00:00:00Z",
                )
            ],
        )

    with pytest.raises(ValidationError, match="selected_task_ids must resolve"):
        HarnessState(
            schema_version="1",
            session=session,
            tasks=[task],
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

    with pytest.raises(ValidationError, match="root_task_id must resolve"):
        HarnessState(
            schema_version="1",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-2",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
            ),
            tasks=[task],
        )

    with pytest.raises(
        ValidationError, match="current_turn_index must equal len\\(turns\\)"
    ):
        HarnessState(
            schema_version="1",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-1",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
                current_turn_index=1,
            ),
            tasks=[task],
        )

    with pytest.raises(ValidationError, match="evidence_refs must resolve"):
        HarnessState(
            schema_version="1",
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
                    verification=VerificationOutcome(evidence_refs=["evidence-1"]),
                )
            ],
        )

    with pytest.raises(ValidationError, match="evidence task ids must resolve"):
        HarnessState(
            schema_version="1",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-1",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
            ),
            tasks=[task],
            verification_evidence=[
                VerificationEvidenceRecord(
                    evidence_id="evidence-1",
                    task_id="task-2",
                )
            ],
        )

    with pytest.raises(ValidationError, match="owned by a different task"):
        HarnessState(
            schema_version="1",
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
                    verification=VerificationOutcome(evidence_refs=["evidence-1"]),
                ),
                TaskRecord(task_id="task-2", title="Sibling task"),
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
            schema_version="1",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-1",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
                current_turn_index=1,
            ),
            tasks=[task],
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


def test_harness_state_rejects_cross_task_evidence_aliasing() -> None:
    task_1 = TaskRecord(task_id="task-1", title="Root task")
    task_2 = TaskRecord(task_id="task-2", title="Child task")

    with pytest.raises(
        ValidationError, match="must not reference evidence owned by a different task"
    ):
        HarnessState(
            schema_version="1",
            session=HarnessSession(
                session_id="session-1",
                root_task_id="task-1",
                budget_policy=BudgetPolicy(max_turns=3),
                started_at="2026-01-01T00:00:00Z",
            ),
            tasks=[
                task_1.model_copy(
                    update={
                        "verification": VerificationOutcome(
                            evidence_refs=["evidence-1"]
                        )
                    }
                ),
                task_2,
            ],
            verification_evidence=[
                VerificationEvidenceRecord(
                    evidence_id="evidence-1",
                    task_id="task-2",
                )
            ],
        )


def test_harness_state_schema_requires_schema_version() -> None:
    schema = HarnessState.model_json_schema()

    assert schema["required"] == ["schema_version", "session"]
    assert schema["properties"]["schema_version"]["minLength"] == 1
    assert schema["properties"]["session"] == {"$ref": "#/$defs/HarnessSession"}
    assert schema["$defs"]["TurnDecisionAction"]["enum"] == [
        "continue",
        "select_tasks",
        "stop",
    ]
