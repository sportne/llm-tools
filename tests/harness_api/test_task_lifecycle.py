"""Lifecycle tests for canonical harness task operations."""

from __future__ import annotations

import pytest

from llm_tools.harness_api import (
    CURRENT_HARNESS_STATE_SCHEMA_VERSION,
    BudgetPolicy,
    InvalidTaskLifecycleError,
    TaskLifecycleStatus,
    TaskOrigin,
    block_task,
    cancel_task,
    complete_task,
    create_derived_task,
    create_root_task,
    fail_task,
    start_task,
    supersede_task,
    unblock_task,
)


def _root_state():
    return create_root_task(
        schema_version=CURRENT_HARNESS_STATE_SCHEMA_VERSION,
        session_id="session-1",
        root_task_id="task-1",
        title="Root task",
        intent="Complete the user request.",
        budget_policy=BudgetPolicy(max_turns=5),
        started_at="2026-01-01T00:00:00Z",
    )


def test_create_root_task_initializes_canonical_state() -> None:
    state = _root_state()

    assert state.schema_version == CURRENT_HARNESS_STATE_SCHEMA_VERSION
    assert state.session.root_task_id == "task-1"
    assert len(state.tasks) == 1
    assert state.tasks[0].origin is TaskOrigin.USER_REQUESTED
    assert state.tasks[0].status is TaskLifecycleStatus.PENDING


def test_create_derived_task_requires_known_parent_and_unique_id() -> None:
    state = _root_state()
    updated = create_derived_task(
        state,
        task_id="task-2",
        title="Derived task",
        intent="Run follow-up work.",
        parent_task_id="task-1",
    )

    child = next(task for task in updated.tasks if task.task_id == "task-2")
    assert child.origin is TaskOrigin.DERIVED
    assert child.parent_task_id == "task-1"

    with pytest.raises(InvalidTaskLifecycleError, match="already exists"):
        create_derived_task(
            updated,
            task_id="task-2",
            title="Duplicate",
            intent="Duplicate task.",
            parent_task_id="task-1",
        )

    with pytest.raises(InvalidTaskLifecycleError, match="does not exist"):
        create_derived_task(
            state,
            task_id="task-3",
            title="Missing parent",
            intent="Broken task.",
            parent_task_id="missing",
        )


def test_start_task_requires_completed_dependencies() -> None:
    state = _root_state()
    state = create_derived_task(
        state,
        task_id="task-2",
        title="Dependent task",
        intent="Wait for root.",
        parent_task_id="task-1",
        depends_on_task_ids=["task-1"],
    )

    with pytest.raises(InvalidTaskLifecycleError, match="dependencies are completed"):
        start_task(
            state,
            task_id="task-2",
            started_at="2026-01-01T00:01:00Z",
        )

    state = start_task(
        state,
        task_id="task-1",
        started_at="2026-01-01T00:00:10Z",
    )
    state = complete_task(
        state,
        task_id="task-1",
        finished_at="2026-01-01T00:00:20Z",
    )
    state = start_task(
        state,
        task_id="task-2",
        started_at="2026-01-01T00:01:00Z",
    )

    task = next(task for task in state.tasks if task.task_id == "task-2")
    assert task.status is TaskLifecycleStatus.IN_PROGRESS
    assert task.started_at == "2026-01-01T00:01:00Z"


def test_block_unblock_complete_and_fail_follow_state_machine_rules() -> None:
    state = _root_state()

    with pytest.raises(InvalidTaskLifecycleError, match="cannot be completed"):
        complete_task(
            state,
            task_id="task-1",
            finished_at="2026-01-01T00:00:10Z",
        )

    state = block_task(state, task_id="task-1", status_summary="Waiting on input.")
    task = next(task for task in state.tasks if task.task_id == "task-1")
    assert task.status is TaskLifecycleStatus.BLOCKED
    assert task.status_summary == "Waiting on input."

    state = unblock_task(state, task_id="task-1", status_summary="Ready again.")
    task = next(task for task in state.tasks if task.task_id == "task-1")
    assert task.status is TaskLifecycleStatus.PENDING
    assert task.status_summary == "Ready again."

    state = start_task(
        state,
        task_id="task-1",
        started_at="2026-01-01T00:00:30Z",
    )
    state = fail_task(
        state,
        task_id="task-1",
        finished_at="2026-01-01T00:00:40Z",
        status_summary="Command failed.",
    )
    task = next(task for task in state.tasks if task.task_id == "task-1")
    assert task.status is TaskLifecycleStatus.FAILED
    assert task.finished_at == "2026-01-01T00:00:40Z"

    with pytest.raises(InvalidTaskLifecycleError, match="terminal"):
        unblock_task(state, task_id="task-1")


def test_cancel_and_supersede_are_terminal_transitions() -> None:
    state = _root_state()
    state = create_derived_task(
        state,
        task_id="task-2",
        title="Replacement task",
        intent="Retry with a better approach.",
        parent_task_id="task-1",
    )
    state = cancel_task(
        state,
        task_id="task-2",
        finished_at="2026-01-01T00:00:10Z",
        status_summary="Canceled by operator.",
    )

    canceled = next(task for task in state.tasks if task.task_id == "task-2")
    assert canceled.status is TaskLifecycleStatus.CANCELED

    with pytest.raises(InvalidTaskLifecycleError, match="terminal"):
        start_task(
            state,
            task_id="task-2",
            started_at="2026-01-01T00:00:11Z",
        )

    state = create_derived_task(
        _root_state(),
        task_id="task-3",
        title="Replacement task",
        intent="New plan.",
        parent_task_id="task-1",
    )
    state = supersede_task(
        state,
        task_id="task-1",
        replacement_task_id="task-3",
        finished_at="2026-01-01T00:00:12Z",
        status_summary="Replaced by derived task.",
    )
    root = next(task for task in state.tasks if task.task_id == "task-1")
    assert root.status is TaskLifecycleStatus.SUPERSEDED
    assert root.superseded_by_task_id == "task-3"

    with pytest.raises(
        InvalidTaskLifecycleError, match="cannot be superseded by itself"
    ):
        supersede_task(
            create_derived_task(
                _root_state(),
                task_id="task-3",
                title="Replacement task",
                intent="New plan.",
                parent_task_id="task-1",
            ),
            task_id="task-3",
            replacement_task_id="task-3",
            finished_at="2026-01-01T00:00:13Z",
        )


def test_lifecycle_operations_reject_invalid_states_and_missing_tasks() -> None:
    root_state = _root_state()

    with pytest.raises(InvalidTaskLifecycleError, match="at least 1 character"):
        create_root_task(
            schema_version="",
            session_id="session-1",
            root_task_id="task-1",
            title="Root task",
            intent="Complete the user request.",
            budget_policy=BudgetPolicy(max_turns=5),
            started_at="2026-01-01T00:00:00Z",
        )

    with pytest.raises(InvalidTaskLifecycleError, match="cannot be started"):
        start_task(
            block_task(root_state, task_id="task-1"),
            task_id="task-1",
            started_at="2026-01-01T00:00:10Z",
        )

    with pytest.raises(InvalidTaskLifecycleError, match="cannot be unblocked"):
        unblock_task(root_state, task_id="task-1")

    completed = complete_task(
        start_task(root_state, task_id="task-1", started_at="2026-01-01T00:00:10Z"),
        task_id="task-1",
        finished_at="2026-01-01T00:00:20Z",
    )
    with pytest.raises(InvalidTaskLifecycleError, match="terminal"):
        block_task(completed, task_id="task-1")

    with pytest.raises(InvalidTaskLifecycleError, match="cannot fail"):
        fail_task(root_state, task_id="task-1", finished_at="2026-01-01T00:00:30Z")

    with pytest.raises(InvalidTaskLifecycleError, match="does not exist"):
        cancel_task(root_state, task_id="missing", finished_at="2026-01-01T00:00:30Z")

    with pytest.raises(InvalidTaskLifecycleError, match="does not exist"):
        supersede_task(
            root_state,
            task_id="task-1",
            replacement_task_id="missing",
            finished_at="2026-01-01T00:00:30Z",
        )
