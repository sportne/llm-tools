"""App-layer helpers for building protection controllers and classifiers."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field

from llm_tools.workflow_api import (
    ProtectionAction,
)


class _LLMProtectionAssessmentModel(BaseModel):
    sensitivity_label: str | None = None
    reasoning: str
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    referenced_document_ids: list[str] = Field(default_factory=list)
    source_document_ids_used: list[str] = Field(default_factory=list)
    requires_cross_source_synthesis: bool | None = None
    requires_inference_beyond_source: bool | None = None
    recommended_action: ProtectionAction | None = None
    guard_text: str | None = None
    sanitized_text: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


__all__ = [
    "_LLMProtectionAssessmentModel",
]
