"""Tests for harness protection scrubbing helpers."""

from __future__ import annotations

from llm_tools.harness_api import (
    DEFAULT_PURGED_RESPONSE,
    BudgetPolicy,
    HarnessSession,
    HarnessState,
    HarnessTurn,
    TaskOrigin,
    TaskRecord,
    scrub_state_for_protection,
    scrub_workflow_result,
)
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.workflow_api import WorkflowTurnResult


def test_scrub_state_for_protection_rewrites_tail_final_response() -> None:
    state = HarnessState(
        schema_version="3",
        session=HarnessSession(
            session_id="session-1",
            root_task_id="task-1",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at="2026-01-01T00:00:00Z",
            current_turn_index=1,
        ),
        tasks=[
            TaskRecord(
                task_id="task-1",
                title="Task",
                intent="Do work",
                origin=TaskOrigin.USER_REQUESTED,
            )
        ],
        turns=[
            HarnessTurn(
                turn_index=1,
                started_at="2026-01-01T00:00:00Z",
                selected_task_ids=["task-1"],
                workflow_result=WorkflowTurnResult(
                    parsed_response=ParsedModelResponse(
                        final_response="Sensitive answer"
                    )
                ),
                decision={
                    "action": "stop",
                    "selected_task_ids": ["task-1"],
                    "stop_reason": "completed",
                },
                ended_at="2026-01-01T00:00:01Z",
            )
        ],
    )

    scrubbed = scrub_state_for_protection(state, safe_message="Safe answer")

    assert scrubbed.turns[-1].workflow_result is not None
    assert (
        scrubbed.turns[-1].workflow_result.parsed_response.final_response
        == "Safe answer"
    )
    assert scrubbed.pending_approvals == []


def test_scrub_state_for_protection_returns_original_when_no_turns_or_result() -> None:
    empty_state = HarnessState(
        schema_version="3",
        session=HarnessSession(
            session_id="session-2",
            root_task_id="task-1",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at="2026-01-01T00:00:00Z",
            current_turn_index=0,
        ),
        tasks=[
            TaskRecord(
                task_id="task-1",
                title="Task",
                intent="Do work",
                origin=TaskOrigin.USER_REQUESTED,
            )
        ],
    )
    no_result_state = empty_state.model_copy(
        update={
            "turns": [
                HarnessTurn(
                    turn_index=1,
                    started_at="2026-01-01T00:00:00Z",
                    selected_task_ids=["task-1"],
                    decision={
                        "action": "stop",
                        "selected_task_ids": ["task-1"],
                        "stop_reason": "completed",
                    },
                    ended_at="2026-01-01T00:00:01Z",
                )
            ]
        },
        deep=True,
    )

    assert scrub_state_for_protection(empty_state) is empty_state
    assert scrub_state_for_protection(no_result_state) is no_result_state


def test_scrub_workflow_result_uses_default_placeholder_and_skips_missing_final_response() -> (
    None
):
    untouched = WorkflowTurnResult(
        parsed_response=ParsedModelResponse(
            invocations=[{"tool_name": "list_directory", "arguments": {"path": "."}}]
        )
    )
    scrubbed = scrub_workflow_result(
        WorkflowTurnResult(
            parsed_response=ParsedModelResponse(final_response="Sensitive answer")
        )
    )

    assert scrub_workflow_result(untouched) is untouched
    assert scrubbed.parsed_response.final_response == DEFAULT_PURGED_RESPONSE
