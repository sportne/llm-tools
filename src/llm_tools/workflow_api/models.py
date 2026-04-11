"""Workflow-layer result models built on top of the core tool runtime."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolResult


class WorkflowTurnResult(BaseModel):
    """Normalized result of one parsed model turn."""

    parsed_response: ParsedModelResponse
    tool_results: list[ToolResult] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_consistency(self) -> WorkflowTurnResult:
        """Require result shape to match the parsed response mode."""
        if self.parsed_response.final_response is not None:
            if self.tool_results:
                raise ValueError(
                    "WorkflowTurnResult with final_response must not include "
                    "tool_results."
                )
            return self

        expected_results = len(self.parsed_response.invocations)
        actual_results = len(self.tool_results)
        if actual_results != expected_results:
            raise ValueError(
                "WorkflowTurnResult tool_results length must match the number of "
                "parsed invocations."
            )

        return self
