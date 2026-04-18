"""Pure task lifecycle operations over canonical harness state."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import ValidationError

from llm_tools.harness_api.models import (
    BudgetPolicy,
    HarnessSession,
    HarnessState,
    TaskLifecycleStatus,
    TaskOrigin,
    TaskRecord,
    VerificationExpectation,
)

_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskLifecycleStatus.COMPLETED,
        TaskLifecycleStatus.FAILED,
        TaskLifecycleStatus.CANCELED,
        TaskLifecycleStatus.SUPERSEDED,
    }
)


class InvalidTaskLifecycleError(ValueError):
    """Raised when a requested task mutation violates harness lifecycle rules."""


class _Unset:
    pass


_UNSET = _Unset()


def create_root_task(
    *,
    schema_version: str,
    session_id: str,
    root_task_id: str,
    title: str,
    intent: str,
    budget_policy: BudgetPolicy,
    started_at: str,
    verification_expectations: Iterable[VerificationExpectation] = (),
    artifact_refs: Iterable[str] = (),
) -> HarnessState:
    """Create the initial harness state with its single user-requested root task."""
    root_task = TaskRecord(
        task_id=root_task_id,
        title=title,
        intent=intent,
        origin=TaskOrigin.USER_REQUESTED,
        verification_expectations=list(verification_expectations),
        artifact_refs=list(artifact_refs),
    )
    try:
        return HarnessState(
            schema_version=schema_version,
            session=HarnessSession(
                session_id=session_id,
                root_task_id=root_task_id,
                budget_policy=budget_policy,
                started_at=started_at,
            ),
            tasks=[root_task],
        )
    except ValidationError as exc:
        raise InvalidTaskLifecycleError(_validation_message(exc)) from exc


def create_derived_task(
    state: HarnessState,
    *,
    task_id: str,
    title: str,
    intent: str,
    parent_task_id: str,
    depends_on_task_ids: Iterable[str] = (),
    verification_expectations: Iterable[VerificationExpectation] = (),
    artifact_refs: Iterable[str] = (),
) -> HarnessState:
    """Append a new derived task to the durable harness task graph."""
    task_map = _task_map(state)
    if task_id in task_map:
        raise InvalidTaskLifecycleError(f"Task '{task_id}' already exists.")
    if parent_task_id not in task_map:
        raise InvalidTaskLifecycleError(
            f"Parent task '{parent_task_id}' does not exist."
        )

    derived_task = TaskRecord(
        task_id=task_id,
        title=title,
        intent=intent,
        origin=TaskOrigin.DERIVED,
        parent_task_id=parent_task_id,
        depends_on_task_ids=list(depends_on_task_ids),
        verification_expectations=list(verification_expectations),
        artifact_refs=list(artifact_refs),
    )
    return _rebuild_state(state, tasks=[*state.tasks, derived_task])


def start_task(
    state: HarnessState,
    *,
    task_id: str,
    started_at: str,
    status_summary: str | None = None,
) -> HarnessState:
    """Move a pending task into in-progress once dependencies resolve."""
    task = _require_task(state, task_id)
    _require_mutable_task(task)
    if task.status is not TaskLifecycleStatus.PENDING:
        raise InvalidTaskLifecycleError(
            f"Task '{task_id}' cannot be started from status '{task.status.value}'."
        )

    blocked_dependencies = [
        dependency_id
        for dependency_id in task.depends_on_task_ids
        if _require_task(state, dependency_id).status
        is not TaskLifecycleStatus.COMPLETED
    ]
    if blocked_dependencies:
        raise InvalidTaskLifecycleError(
            "Task "
            f"'{task_id}' cannot start until dependencies are completed: "
            + ", ".join(blocked_dependencies)
            + "."
        )

    return _update_task(
        state,
        task_id,
        status=TaskLifecycleStatus.IN_PROGRESS,
        started_at=task.started_at or started_at,
        finished_at=None,
        status_summary=status_summary,
    )


def unblock_task(
    state: HarnessState,
    *,
    task_id: str,
    status_summary: str | None = None,
) -> HarnessState:
    """Move a blocked task back to pending so it can be reconsidered."""
    task = _require_task(state, task_id)
    _require_mutable_task(task)
    if task.status is not TaskLifecycleStatus.BLOCKED:
        raise InvalidTaskLifecycleError(
            f"Task '{task_id}' cannot be unblocked from status '{task.status.value}'."
        )

    return _update_task(
        state,
        task_id,
        status=TaskLifecycleStatus.PENDING,
        status_summary=status_summary,
    )


def block_task(
    state: HarnessState,
    *,
    task_id: str,
    status_summary: str | None = None,
) -> HarnessState:
    """Mark a pending or in-progress task as blocked."""
    task = _require_task(state, task_id)
    _require_mutable_task(task)
    if task.status not in {
        TaskLifecycleStatus.PENDING,
        TaskLifecycleStatus.IN_PROGRESS,
    }:
        raise InvalidTaskLifecycleError(
            f"Task '{task_id}' cannot be blocked from status '{task.status.value}'."
        )

    return _update_task(
        state,
        task_id,
        status=TaskLifecycleStatus.BLOCKED,
        status_summary=status_summary,
    )


def complete_task(
    state: HarnessState,
    *,
    task_id: str,
    finished_at: str,
    status_summary: str | None = None,
) -> HarnessState:
    """Move an in-progress task to the completed terminal state."""
    task = _require_task(state, task_id)
    _require_mutable_task(task)
    if task.status is not TaskLifecycleStatus.IN_PROGRESS:
        raise InvalidTaskLifecycleError(
            f"Task '{task_id}' cannot be completed from status '{task.status.value}'."
        )

    return _update_task(
        state,
        task_id,
        status=TaskLifecycleStatus.COMPLETED,
        finished_at=finished_at,
        status_summary=status_summary,
    )


def fail_task(
    state: HarnessState,
    *,
    task_id: str,
    finished_at: str,
    status_summary: str | None = None,
) -> HarnessState:
    """Move an in-progress or blocked task to the failed terminal state."""
    task = _require_task(state, task_id)
    _require_mutable_task(task)
    if task.status not in {
        TaskLifecycleStatus.IN_PROGRESS,
        TaskLifecycleStatus.BLOCKED,
    }:
        raise InvalidTaskLifecycleError(
            f"Task '{task_id}' cannot fail from status '{task.status.value}'."
        )

    return _update_task(
        state,
        task_id,
        status=TaskLifecycleStatus.FAILED,
        finished_at=finished_at,
        status_summary=status_summary,
    )


def cancel_task(
    state: HarnessState,
    *,
    task_id: str,
    finished_at: str,
    status_summary: str | None = None,
) -> HarnessState:
    """Move a non-terminal task to the canceled terminal state."""
    task = _require_task(state, task_id)
    _require_mutable_task(task)

    return _update_task(
        state,
        task_id,
        status=TaskLifecycleStatus.CANCELED,
        finished_at=finished_at,
        status_summary=status_summary,
    )


def supersede_task(
    state: HarnessState,
    *,
    task_id: str,
    replacement_task_id: str,
    finished_at: str,
    status_summary: str | None = None,
) -> HarnessState:
    """Move a non-terminal task to the superseded terminal state."""
    task = _require_task(state, task_id)
    _require_mutable_task(task)
    if replacement_task_id == task_id:
        raise InvalidTaskLifecycleError(
            f"Task '{task_id}' cannot be superseded by itself."
        )
    if replacement_task_id not in _task_map(state):
        raise InvalidTaskLifecycleError(
            f"Replacement task '{replacement_task_id}' does not exist."
        )

    return _update_task(
        state,
        task_id,
        status=TaskLifecycleStatus.SUPERSEDED,
        superseded_by_task_id=replacement_task_id,
        finished_at=finished_at,
        status_summary=status_summary,
    )


def _task_map(state: HarnessState) -> dict[str, TaskRecord]:
    return {task.task_id: task for task in state.tasks}


def _require_task(state: HarnessState, task_id: str) -> TaskRecord:
    task = _task_map(state).get(task_id)
    if task is None:
        raise InvalidTaskLifecycleError(f"Task '{task_id}' does not exist.")
    return task


def _require_mutable_task(task: TaskRecord) -> None:
    if task.status in _TERMINAL_TASK_STATUSES:
        raise InvalidTaskLifecycleError(
            f"Task '{task.task_id}' is terminal and cannot transition from "
            f"'{task.status.value}'."
        )


def _update_task(
    state: HarnessState,
    task_id: str,
    *,
    status: TaskLifecycleStatus,
    started_at: str | None | object = _UNSET,
    finished_at: str | None | object = _UNSET,
    status_summary: str | None | object = _UNSET,
    superseded_by_task_id: str | None | object = _UNSET,
) -> HarnessState:
    updated_tasks: list[TaskRecord] = []

    for task in state.tasks:
        if task.task_id != task_id:
            updated_tasks.append(task)
            continue

        update_payload: dict[str, object | None] = {"status": status}
        if started_at is not _UNSET:
            update_payload["started_at"] = started_at
        if finished_at is not _UNSET:
            update_payload["finished_at"] = finished_at
        if status_summary is not _UNSET:
            update_payload["status_summary"] = status_summary
        if superseded_by_task_id is not _UNSET:
            update_payload["superseded_by_task_id"] = superseded_by_task_id

        updated_tasks.append(task.model_copy(update=update_payload))

    return _rebuild_state(state, tasks=updated_tasks)


def _rebuild_state(
    state: HarnessState,
    *,
    tasks: list[TaskRecord] | None = None,
) -> HarnessState:
    payload = state.model_dump(mode="json")
    if tasks is not None:
        payload["tasks"] = [task.model_dump(mode="json") for task in tasks]

    try:
        return HarnessState.model_validate(payload)
    except ValidationError as exc:
        raise InvalidTaskLifecycleError(_validation_message(exc)) from exc


def _validation_message(exc: ValidationError) -> str:
    return "; ".join(error["msg"] for error in exc.errors())
