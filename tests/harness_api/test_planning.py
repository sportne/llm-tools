"""Focused tests for harness planning and replanning signals."""

from __future__ import annotations

from llm_tools.harness_api import (
    CURRENT_HARNESS_STATE_SCHEMA_VERSION,
    BudgetPolicy,
    DeterministicHarnessPlanner,
    HarnessState,
    ReplanningTrigger,
    TurnDecision,
    TurnDecisionAction,
    block_task,
    complete_task,
    create_derived_task,
    create_root_task,
    start_task,
)


def _root_state() -> HarnessState:
    return create_root_task(
        schema_version=CURRENT_HARNESS_STATE_SCHEMA_VERSION,
        session_id="session-1",
        root_task_id="task-1",
        title="Root task",
        intent="Complete the user request.",
        budget_policy=BudgetPolicy(max_turns=4),
        started_at="2026-01-01T00:00:00Z",
    )


def test_planner_prefers_first_actionable_in_progress_task() -> None:
    planner = DeterministicHarnessPlanner()
    state = create_derived_task(
        _root_state(),
        task_id="task-2",
        title="Second task",
        intent="Do follow-up work.",
        parent_task_id="task-1",
    )
    state = start_task(
        state,
        task_id="task-2",
        started_at="2026-01-01T00:00:10Z",
    )

    selection = planner.select_tasks(state=state)

    assert selection.selected_task_ids == ["task-2"]
    assert selection.blocked_reasons == []


def test_planner_ignores_blocked_and_terminal_tasks() -> None:
    planner = DeterministicHarnessPlanner()
    state = create_derived_task(
        complete_task(
            start_task(
                _root_state(),
                task_id="task-1",
                started_at="2026-01-01T00:00:05Z",
            ),
            task_id="task-1",
            finished_at="2026-01-01T00:00:10Z",
        ),
        task_id="task-2",
        title="Blocked task",
        intent="Wait for input.",
        parent_task_id="task-1",
    )
    state = block_task(state, task_id="task-2", status_summary="Waiting.")

    selection = planner.select_tasks(state=state)

    assert selection.selected_task_ids == []
    assert selection.blocked_reasons == ["task-2:blocked"]


def test_planner_ignores_pending_tasks_with_incomplete_dependencies() -> None:
    planner = DeterministicHarnessPlanner()
    state = create_derived_task(
        _root_state(),
        task_id="task-2",
        title="Dependent task",
        intent="Wait for root task completion.",
        parent_task_id="task-1",
        depends_on_task_ids=["task-1"],
    )

    selection = planner.select_tasks(state=state)

    assert selection.selected_task_ids == ["task-1"]
    state = complete_task(
        start_task(state, task_id="task-1", started_at="2026-01-01T00:00:10Z"),
        task_id="task-1",
        finished_at="2026-01-01T00:00:20Z",
    )
    selection_after_completion = planner.select_tasks(state=state)
    assert selection_after_completion.selected_task_ids == ["task-2"]


def test_planner_selection_is_stable_for_identical_state_and_order() -> None:
    planner = DeterministicHarnessPlanner()
    state = create_derived_task(
        create_derived_task(
            _root_state(),
            task_id="task-2",
            title="First sibling",
            intent="First pending task.",
            parent_task_id="task-1",
        ),
        task_id="task-3",
        title="Second sibling",
        intent="Second pending task.",
        parent_task_id="task-1",
    )

    first = planner.select_tasks(state=state)
    second = planner.select_tasks(state=state)

    assert first.selected_task_ids == ["task-1"]
    assert second == first


def test_detect_replanning_triggers_covers_requested_blocked_terminal_and_new_work() -> (
    None
):
    planner = DeterministicHarnessPlanner()
    previous_state = start_task(
        create_derived_task(
            _root_state(),
            task_id="task-2",
            title="Follow-up",
            intent="Run derived work.",
            parent_task_id="task-1",
        ),
        task_id="task-1",
        started_at="2026-01-01T00:00:10Z",
    )
    current_state = create_derived_task(
        block_task(
            complete_task(
                previous_state,
                task_id="task-1",
                finished_at="2026-01-01T00:00:20Z",
            ),
            task_id="task-2",
            status_summary="Waiting for approval.",
        ),
        task_id="task-3",
        title="Replacement",
        intent="New derived work.",
        parent_task_id="task-1",
    )

    triggers = planner.detect_replanning_triggers(
        previous_state=previous_state,
        current_state=current_state,
        previous_selected_task_ids=["task-1", "task-2"],
        decision=TurnDecision(
            action=TurnDecisionAction.SELECT_TASKS,
            selected_task_ids=["task-1"],
        ),
    )

    assert triggers == [
        ReplanningTrigger.SELECT_TASKS_REQUESTED,
        ReplanningTrigger.SELECTED_TASK_BLOCKED,
        ReplanningTrigger.SELECTED_TASK_TERMINAL,
        ReplanningTrigger.NEW_DERIVED_TASK_CREATED,
    ]


def test_detect_replanning_triggers_reports_no_actionable_non_terminal_work() -> None:
    planner = DeterministicHarnessPlanner()
    previous_state = create_derived_task(
        _root_state(),
        task_id="task-2",
        title="Blocked derived task",
        intent="Cannot proceed yet.",
        parent_task_id="task-1",
    )
    current_state = block_task(
        previous_state,
        task_id="task-1",
        status_summary="Waiting on operator input.",
    )
    current_state = block_task(
        current_state,
        task_id="task-2",
        status_summary="Still waiting.",
    )

    triggers = planner.detect_replanning_triggers(
        previous_state=previous_state,
        current_state=current_state,
        previous_selected_task_ids=[],
        decision=None,
    )

    assert triggers == [ReplanningTrigger.NO_ACTIONABLE_TASKS_REMAINING]
