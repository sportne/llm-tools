"""Planner contracts for deterministic harness task selection."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


class TaskSelection(BaseModel):
    """Deterministic planner output for one harness turn."""

    selected_task_ids: list[str] = Field(default_factory=list)
    blocked_reasons: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_selection(self) -> TaskSelection:
        """Require stable unique task ids and diagnostic entries."""
        if len(set(self.selected_task_ids)) != len(self.selected_task_ids):
            raise ValueError("selected_task_ids must be unique.")
        if any(reason.strip() == "" for reason in self.blocked_reasons):
            raise ValueError("blocked_reasons must not contain empty entries.")
        return self


__all__ = [
    "TaskSelection",
]
