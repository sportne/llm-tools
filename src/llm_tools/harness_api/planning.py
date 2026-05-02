"""Planner contracts for deterministic harness task selection."""

from __future__ import annotations

from collections.abc import Sequence
from enum import Enum
from typing import Protocol, runtime_checkable

from llm_tools.harness_api.models import (
    HarnessState,
    TaskLifecycleStatus,
    TaskOrigin,
    TaskRecord,
    TurnDecision,
    TurnDecisionAction,
)
from llm_tools.harness_api.planning_models import (
    TaskSelection as TaskSelection,
)

_TERMINAL_TASK_STATUSES = frozenset(
    {
        TaskLifecycleStatus.COMPLETED,
        TaskLifecycleStatus.FAILED,
        TaskLifecycleStatus.CANCELED,
        TaskLifecycleStatus.SUPERSEDED,
    }
)


class ReplanningTrigger(str, Enum):  # noqa: UP042
    """Stable replanning reasons derived from harness state transitions."""

    SELECT_TASKS_REQUESTED = "select_tasks_requested"
    SELECTED_TASK_BLOCKED = "selected_task_blocked"
    SELECTED_TASK_TERMINAL = "selected_task_terminal"
    NEW_DERIVED_TASK_CREATED = "new_derived_task_created"
    NO_ACTIONABLE_TASKS_REMAINING = "no_actionable_tasks_remaining"


@runtime_checkable
class HarnessPlanner(Protocol):
    """Own deterministic task selection and replanning signal derivation."""

    def select_tasks(self, *, state: HarnessState) -> TaskSelection:
        """Return the next actionable task ids in deterministic order."""

    def detect_replanning_triggers(
        self,
        *,
        previous_state: HarnessState,
        current_state: HarnessState,
        previous_selected_task_ids: Sequence[str],
        decision: TurnDecision | None,
    ) -> list[ReplanningTrigger]:
        """Return derived replanning triggers in stable priority order."""


class DeterministicHarnessPlanner:
    """Minimal planner that selects one actionable task at a time."""

    def select_tasks(self, *, state: HarnessState) -> TaskSelection:
        """Select the first actionable task, preferring in-progress work."""
        task_map = {task.task_id: task for task in state.tasks}
        actionable_in_progress: list[str] = []
        actionable_pending: list[str] = []
        blocked_reasons: list[str] = []

        for task in state.tasks:
            if _is_task_actionable(task, task_map=task_map):
                if task.status is TaskLifecycleStatus.IN_PROGRESS:
                    actionable_in_progress.append(task.task_id)
                else:
                    actionable_pending.append(task.task_id)
                continue

            reason = _blocked_reason(task, task_map=task_map)
            if reason is not None:
                blocked_reasons.append(reason)

        if actionable_in_progress:
            return TaskSelection(selected_task_ids=[actionable_in_progress[0]])
        if actionable_pending:
            return TaskSelection(selected_task_ids=[actionable_pending[0]])
        return TaskSelection(blocked_reasons=blocked_reasons)

    def detect_replanning_triggers(
        self,
        *,
        previous_state: HarnessState,
        current_state: HarnessState,
        previous_selected_task_ids: Sequence[str],
        decision: TurnDecision | None,
    ) -> list[ReplanningTrigger]:
        """Derive stable replanning triggers without mutating harness state."""
        triggers: list[ReplanningTrigger] = []
        current_task_map = {task.task_id: task for task in current_state.tasks}
        previous_task_ids = {task.task_id for task in previous_state.tasks}

        if decision is not None and decision.action is TurnDecisionAction.SELECT_TASKS:
            triggers.append(ReplanningTrigger.SELECT_TASKS_REQUESTED)

        if any(
            current_task_map.get(task_id) is not None
            and current_task_map[task_id].status is TaskLifecycleStatus.BLOCKED
            for task_id in previous_selected_task_ids
        ):
            triggers.append(ReplanningTrigger.SELECTED_TASK_BLOCKED)

        if any(
            current_task_map.get(task_id) is not None
            and current_task_map[task_id].status in _TERMINAL_TASK_STATUSES
            for task_id in previous_selected_task_ids
        ):
            triggers.append(ReplanningTrigger.SELECTED_TASK_TERMINAL)

        if any(
            task.origin is TaskOrigin.DERIVED and task.task_id not in previous_task_ids
            for task in current_state.tasks
        ):
            triggers.append(ReplanningTrigger.NEW_DERIVED_TASK_CREATED)

        has_non_terminal_task = any(
            task.status not in _TERMINAL_TASK_STATUSES for task in current_state.tasks
        )
        if (
            has_non_terminal_task
            and not self.select_tasks(state=current_state).selected_task_ids
        ):
            triggers.append(ReplanningTrigger.NO_ACTIONABLE_TASKS_REMAINING)

        return triggers


def _is_task_actionable(
    task: TaskRecord,
    *,
    task_map: dict[str, TaskRecord],
) -> bool:
    if task.status not in {
        TaskLifecycleStatus.PENDING,
        TaskLifecycleStatus.IN_PROGRESS,
    }:
        return False
    return all(
        task_map[dependency_id].status is TaskLifecycleStatus.COMPLETED
        for dependency_id in task.depends_on_task_ids
    )


def _blocked_reason(
    task: TaskRecord,
    *,
    task_map: dict[str, TaskRecord],
) -> str | None:
    if task.status is TaskLifecycleStatus.BLOCKED:
        return f"{task.task_id}:blocked"
    if task.status in _TERMINAL_TASK_STATUSES:
        return None
    if task.status in {
        TaskLifecycleStatus.PENDING,
        TaskLifecycleStatus.IN_PROGRESS,
    } and any(
        task_map[dependency_id].status is not TaskLifecycleStatus.COMPLETED
        for dependency_id in task.depends_on_task_ids
    ):
        return f"{task.task_id}:waiting_on_dependencies"
    return None
