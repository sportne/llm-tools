"""Provider-neutral turn-context projections derived from harness state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from pydantic import BaseModel, Field

from llm_tools.harness_api.models import (
    BudgetPolicy,
    HarnessState,
    HarnessStopReason,
    HarnessTurn,
    TaskLifecycleStatus,
    TaskRecord,
    TurnDecisionAction,
)
from llm_tools.harness_api.verification import VerificationStatus
from llm_tools.tool_api import ToolContext
from llm_tools.workflow_api.models import WorkflowInvocationStatus


class TurnContextBudget(BaseModel):
    """Explicit provider-neutral limits for harness-derived turn context."""

    max_selected_tasks: int = Field(default=1, ge=1)
    max_related_tasks: int = Field(default=6, ge=0)
    max_recent_turns: int = Field(default=5, ge=0)
    max_chars_per_text_field: int = Field(default=400, ge=0)
    max_total_chars: int = Field(default=6000, ge=0)


class TaskContextProjection(BaseModel):
    """Derived task view copied from canonical task state plus selection hints."""

    task_id: str
    title: str
    intent: str
    status: TaskLifecycleStatus
    parent_task_id: str | None = None
    depends_on_task_ids: list[str] = Field(default_factory=list)
    status_summary: str | None = None
    retry_count: int = Field(default=0, ge=0)
    verification_status: VerificationStatus
    artifact_refs: list[str] = Field(default_factory=list)
    is_selected: bool = False
    is_actionable: bool = False
    truncated_fields: list[str] = Field(default_factory=list)


class TurnContextProjection(BaseModel):
    """Derived compact view of one completed harness turn."""

    turn_index: int = Field(ge=1)
    selected_task_ids: list[str] = Field(default_factory=list)
    decision_action: TurnDecisionAction | None = None
    decision_stop_reason: HarnessStopReason | None = None
    decision_summary: str | None = None
    workflow_outcome_statuses: list[WorkflowInvocationStatus] = Field(
        default_factory=list
    )
    truncated_fields: list[str] = Field(default_factory=list)


class BudgetContextProjection(BaseModel):
    """Budget config plus derived omission and truncation accounting."""

    configured: TurnContextBudget
    remaining_total_chars: int = Field(ge=0)
    omitted_selected_task_count: int = Field(default=0, ge=0)
    omitted_related_task_count: int = Field(default=0, ge=0)
    omitted_recent_turn_count: int = Field(default=0, ge=0)
    truncated_field_count: int = Field(default=0, ge=0)
    budget_exhausted: bool = False


class HarnessTurnContext(BaseModel):
    """Provider-neutral turn context derived from canonical harness state."""

    session_id: str
    root_task_id: str
    current_turn_index: int = Field(ge=0)
    session_started_at: str
    session_retry_count: int = Field(default=0, ge=0)
    session_budget_policy: BudgetPolicy
    turn_index: int = Field(ge=1)
    selected_tasks: list[TaskContextProjection] = Field(default_factory=list)
    related_tasks: list[TaskContextProjection] = Field(default_factory=list)
    recent_turns: list[TurnContextProjection] = Field(default_factory=list)
    context_budget: BudgetContextProjection


class TurnContextBundle(BaseModel):
    """Built tool context plus the structured projection stored inside it."""

    tool_context: ToolContext
    projection: HarnessTurnContext


@runtime_checkable
class HarnessContextBuilder(Protocol):
    """Build provider-neutral turn context from canonical harness state."""

    def build(
        self,
        *,
        state: HarnessState,
        selected_task_ids: Sequence[str],
        turn_index: int,
        workspace: str | None = None,
    ) -> TurnContextBundle:
        """Return a tool context plus its derived harness projection."""


class DefaultHarnessContextBuilder:
    """Build a bounded derived context without prompt or provider artifacts."""

    def __init__(self, *, budget: TurnContextBudget | None = None) -> None:
        self._budget = budget or TurnContextBudget()

    def build(
        self,
        *,
        state: HarnessState,
        selected_task_ids: Sequence[str],
        turn_index: int,
        workspace: str | None = None,
    ) -> TurnContextBundle:
        """Project canonical harness state into a provider-neutral turn context."""
        task_map = {task.task_id: task for task in state.tasks}
        tracker = _TextBudgetTracker(self._budget)

        selected_ids = list(selected_task_ids[: self._budget.max_selected_tasks])
        selected_tasks: list[TaskContextProjection] = []
        for task_id in selected_ids:
            task = task_map.get(task_id)
            if task is None:
                continue
            selected_tasks.append(
                _project_task(
                    task,
                    task_map=task_map,
                    tracker=tracker,
                    is_selected=True,
                )
            )

        selected_id_set = set(selected_ids)
        actionable_related = [
            task
            for task in state.tasks
            if task.task_id not in selected_id_set
            and _is_task_actionable(task, task_map=task_map)
        ]
        included_related = actionable_related[: self._budget.max_related_tasks]
        related_tasks: list[TaskContextProjection] = []
        omitted_related_count = max(0, len(actionable_related) - len(included_related))
        for task in included_related:
            if tracker.budget_exhausted:
                break
            related_tasks.append(
                _project_task(
                    task,
                    task_map=task_map,
                    tracker=tracker,
                    is_selected=False,
                )
            )
        omitted_related_count += max(0, len(included_related) - len(related_tasks))

        completed_turns = [
            turn for turn in reversed(state.turns) if turn.decision is not None
        ]
        included_turns = completed_turns[: self._budget.max_recent_turns]
        recent_turns: list[TurnContextProjection] = []
        omitted_recent_turn_count = max(0, len(completed_turns) - len(included_turns))
        for turn in included_turns:
            if tracker.budget_exhausted:
                break
            recent_turns.append(_project_turn(turn, tracker=tracker))
        omitted_recent_turn_count += max(0, len(included_turns) - len(recent_turns))

        projection = HarnessTurnContext(
            session_id=state.session.session_id,
            root_task_id=state.session.root_task_id,
            current_turn_index=state.session.current_turn_index,
            session_started_at=state.session.started_at,
            session_retry_count=state.session.retry_count,
            session_budget_policy=state.session.budget_policy,
            turn_index=turn_index,
            selected_tasks=selected_tasks,
            related_tasks=related_tasks,
            recent_turns=recent_turns,
            context_budget=BudgetContextProjection(
                configured=self._budget,
                remaining_total_chars=tracker.remaining_total_chars,
                omitted_selected_task_count=max(
                    0, len(selected_task_ids) - len(selected_tasks)
                ),
                omitted_related_task_count=omitted_related_count,
                omitted_recent_turn_count=omitted_recent_turn_count,
                truncated_field_count=tracker.truncated_field_count,
                budget_exhausted=tracker.budget_exhausted,
            ),
        )
        tool_context = ToolContext(
            invocation_id=f"turn-{turn_index}",
            workspace=workspace,
            metadata={"harness_turn_context": projection.model_dump(mode="json")},
        )
        return TurnContextBundle(tool_context=tool_context, projection=projection)


class _TextBudgetTracker:
    def __init__(self, budget: TurnContextBudget) -> None:
        self._budget = budget
        self.remaining_total_chars = budget.max_total_chars
        self.truncated_field_count = 0

    @property
    def budget_exhausted(self) -> bool:
        return self.remaining_total_chars <= 0

    def consume(
        self,
        value: str | None,
        *,
        optional: bool,
    ) -> tuple[str | None, bool]:
        if value is None:
            return None, False

        truncated = False
        limited = value
        if len(limited) > self._budget.max_chars_per_text_field:
            limited = limited[: self._budget.max_chars_per_text_field]
            truncated = True

        if len(limited) > self.remaining_total_chars:
            limited = limited[: self.remaining_total_chars]
            truncated = True

        self.remaining_total_chars = max(0, self.remaining_total_chars - len(limited))
        if truncated:
            self.truncated_field_count += 1

        if optional and limited == "":
            return None, truncated
        return limited, truncated


def _project_task(
    task: TaskRecord,
    *,
    task_map: dict[str, TaskRecord],
    tracker: _TextBudgetTracker,
    is_selected: bool,
) -> TaskContextProjection:
    title, title_truncated = tracker.consume(task.title, optional=False)
    intent, intent_truncated = tracker.consume(task.intent, optional=False)
    status_summary, summary_truncated = tracker.consume(
        task.status_summary,
        optional=True,
    )
    truncated_fields = [
        field_name
        for field_name, was_truncated in (
            ("title", title_truncated),
            ("intent", intent_truncated),
            ("status_summary", summary_truncated),
        )
        if was_truncated
    ]
    return TaskContextProjection(
        task_id=task.task_id,
        title=title or "",
        intent=intent or "",
        status=task.status,
        parent_task_id=task.parent_task_id,
        depends_on_task_ids=list(task.depends_on_task_ids),
        status_summary=status_summary,
        retry_count=task.retry_count,
        verification_status=task.verification.status,
        artifact_refs=list(task.artifact_refs),
        is_selected=is_selected,
        is_actionable=_is_task_actionable(task, task_map=task_map),
        truncated_fields=truncated_fields,
    )


def _project_turn(
    turn: HarnessTurn,
    *,
    tracker: _TextBudgetTracker,
) -> TurnContextProjection:
    decision = turn.decision
    decision_summary, summary_truncated = tracker.consume(
        decision.summary if decision is not None else None,
        optional=True,
    )
    return TurnContextProjection(
        turn_index=turn.turn_index,
        selected_task_ids=list(turn.selected_task_ids),
        decision_action=decision.action if decision is not None else None,
        decision_stop_reason=decision.stop_reason if decision is not None else None,
        decision_summary=decision_summary,
        workflow_outcome_statuses=[
            outcome.status for outcome in (turn.workflow_result.outcomes or [])
        ]
        if turn.workflow_result is not None
        else [],
        truncated_fields=["decision_summary"] if summary_truncated else [],
    )


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
