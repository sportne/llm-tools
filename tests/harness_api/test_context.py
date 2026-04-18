"""Focused tests for harness turn-context projection and budgeting."""

from __future__ import annotations

import json

from llm_tools.harness_api import (
    CURRENT_HARNESS_STATE_SCHEMA_VERSION,
    BudgetPolicy,
    DefaultHarnessContextBuilder,
    HarnessState,
    HarnessStopReason,
    HarnessTurn,
    TaskLifecycleStatus,
    TurnContextBudget,
    TurnDecision,
    TurnDecisionAction,
    create_derived_task,
    create_root_task,
    start_task,
)
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolInvocationRequest, ToolResult
from llm_tools.workflow_api.models import (
    WorkflowInvocationOutcome,
    WorkflowInvocationStatus,
    WorkflowTurnResult,
)


def _root_state() -> HarnessState:
    return create_root_task(
        schema_version=CURRENT_HARNESS_STATE_SCHEMA_VERSION,
        session_id="session-1",
        root_task_id="task-1",
        title="Root task",
        intent="Complete the request.",
        budget_policy=BudgetPolicy(max_turns=4),
        started_at="2026-01-01T00:00:00Z",
    )


def _state_with_turns() -> HarnessState:
    request = ToolInvocationRequest(
        tool_name="harness_work",
        arguments={"value": "ok"},
    )
    turn = HarnessTurn(
        turn_index=1,
        started_at="2026-01-01T00:00:00Z",
        selected_task_ids=["task-1"],
        workflow_result=WorkflowTurnResult(
            parsed_response=ParsedModelResponse(invocations=[request]),
            outcomes=[
                WorkflowInvocationOutcome(
                    invocation_index=1,
                    request=request,
                    status=WorkflowInvocationStatus.EXECUTED,
                    tool_result=ToolResult(
                        ok=True,
                        tool_name="harness_work",
                        tool_version="0.1.0",
                        output={"value": "ok"},
                    ),
                )
            ],
        ),
        decision=TurnDecision(
            action=TurnDecisionAction.STOP,
            selected_task_ids=["task-1"],
            stop_reason=HarnessStopReason.COMPLETED,
            summary="Completed prior work.",
        ),
        ended_at="2026-01-01T00:00:05Z",
    )
    base = _root_state()
    state = base.model_copy(
        update={
            "session": base.session.model_copy(update={"current_turn_index": 1}),
            "turns": [turn],
        },
        deep=True,
    )
    state = create_derived_task(
        state,
        task_id="task-2",
        title="Derived task",
        intent="Do related follow-up work.",
        parent_task_id="task-1",
    )
    return start_task(
        state,
        task_id="task-2",
        started_at="2026-01-01T00:00:10Z",
        status_summary="Actively working.",
    )


def test_context_builder_copies_only_allowed_canonical_fields() -> None:
    builder = DefaultHarnessContextBuilder()
    state = _state_with_turns()

    bundle = builder.build(
        state=state,
        selected_task_ids=["task-1"],
        turn_index=2,
        workspace="/workspace/repo",
    )

    projection = bundle.projection
    selected = projection.selected_tasks[0]
    assert projection.session_id == "session-1"
    assert projection.root_task_id == "task-1"
    assert projection.current_turn_index == 1
    assert projection.session_started_at == "2026-01-01T00:00:00Z"
    assert projection.session_retry_count == 0
    assert projection.session_budget_policy.max_turns == 4
    assert selected.task_id == "task-1"
    assert selected.title == "Root task"
    assert selected.intent == "Complete the request."
    assert selected.status is TaskLifecycleStatus.PENDING
    assert selected.parent_task_id is None
    assert selected.depends_on_task_ids == []
    assert selected.status_summary is None
    assert selected.retry_count == 0
    assert selected.artifact_refs == []


def test_context_builder_keeps_derived_summaries_separate_from_canonical_fields() -> (
    None
):
    builder = DefaultHarnessContextBuilder()
    bundle = builder.build(
        state=_state_with_turns(),
        selected_task_ids=["task-1"],
        turn_index=2,
    )

    related = bundle.projection.related_tasks[0]
    assert related.is_selected is False
    assert related.is_actionable is True
    assert related.status_summary == "Actively working."
    assert bundle.projection.context_budget.omitted_related_task_count == 0


def test_context_builder_prioritizes_selected_then_related_then_recent_turns() -> None:
    builder = DefaultHarnessContextBuilder(
        budget=TurnContextBudget(
            max_selected_tasks=1,
            max_related_tasks=1,
            max_recent_turns=1,
        )
    )
    state = _state_with_turns()

    bundle = builder.build(
        state=state,
        selected_task_ids=["task-1"],
        turn_index=2,
    )

    assert [task.task_id for task in bundle.projection.selected_tasks] == ["task-1"]
    assert [task.task_id for task in bundle.projection.related_tasks] == ["task-2"]
    assert [turn.turn_index for turn in bundle.projection.recent_turns] == [1]


def test_context_builder_truncates_text_fields_and_total_budget_deterministically() -> (
    None
):
    builder = DefaultHarnessContextBuilder(
        budget=TurnContextBudget(
            max_selected_tasks=1,
            max_related_tasks=2,
            max_recent_turns=1,
            max_chars_per_text_field=5,
            max_total_chars=12,
        )
    )
    state = _state_with_turns()
    state = state.model_copy(
        update={
            "tasks": [
                state.tasks[0].model_copy(
                    update={
                        "title": "SelectedTitle",
                        "intent": "SelectedIntent",
                    }
                ),
                state.tasks[1].model_copy(
                    update={
                        "title": "RelatedTitle",
                        "intent": "RelatedIntent",
                        "status_summary": "RelatedSummary",
                    }
                ),
            ]
        },
        deep=True,
    )
    state = create_derived_task(
        state,
        task_id="task-3",
        title="ExtraTitle",
        intent="ExtraIntent",
        parent_task_id="task-1",
    )

    bundle = builder.build(
        state=state,
        selected_task_ids=["task-1"],
        turn_index=2,
    )

    selected = bundle.projection.selected_tasks[0]
    assert selected.title == "Selec"
    assert selected.intent == "Selec"
    assert "title" in selected.truncated_fields
    assert "intent" in selected.truncated_fields
    assert bundle.projection.context_budget.budget_exhausted is True
    assert bundle.projection.context_budget.truncated_field_count >= 2
    assert bundle.projection.context_budget.omitted_related_task_count >= 1


def test_context_builder_records_json_serializable_projection_in_tool_context() -> None:
    builder = DefaultHarnessContextBuilder()
    bundle = builder.build(
        state=_state_with_turns(),
        selected_task_ids=["task-1"],
        turn_index=2,
        workspace="/workspace/repo",
    )

    assert bundle.tool_context.invocation_id == "turn-2"
    assert bundle.tool_context.workspace == "/workspace/repo"
    payload = bundle.tool_context.metadata["harness_turn_context"]
    assert payload["turn_index"] == 2
    assert payload["selected_tasks"][0]["task_id"] == "task-1"
    json.dumps(payload)


def test_context_output_and_canonical_state_do_not_include_provider_prompt_or_token_fields() -> (
    None
):
    builder = DefaultHarnessContextBuilder()
    state = _state_with_turns()
    bundle = builder.build(state=state, selected_task_ids=["task-1"], turn_index=2)

    projection_dump = bundle.projection.model_dump(mode="json")
    state_dump = state.model_dump(mode="json")
    serialized_projection = json.dumps(projection_dump)
    serialized_state = json.dumps(state_dump)

    for forbidden in ("prompt", "provider", "token", "messages"):
        assert forbidden not in serialized_projection
        assert forbidden not in serialized_state
