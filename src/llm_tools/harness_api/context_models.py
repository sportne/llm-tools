"""Provider-neutral turn-context projections derived from harness state."""

from __future__ import annotations

from pydantic import BaseModel, Field

from llm_tools.harness_api.models import (
    BudgetPolicy,
    HarnessStopReason,
    TaskLifecycleStatus,
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


__all__ = [
    "BudgetContextProjection",
    "HarnessTurnContext",
    "TaskContextProjection",
    "TurnContextBudget",
    "TurnContextBundle",
    "TurnContextProjection",
]
