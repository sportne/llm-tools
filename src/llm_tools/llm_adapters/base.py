"""Shared adapter models for LLM-facing interaction layers."""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator

from llm_tools.tool_api import ToolInvocationRequest


class ParsedModelResponse(BaseModel):
    """Normalized outcome of one model turn."""

    invocations: list[ToolInvocationRequest] = Field(default_factory=list)
    final_response: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> ParsedModelResponse:
        """Require exactly one output mode."""
        has_invocations = len(self.invocations) > 0
        final_response = self.final_response
        has_final_response = final_response is not None

        if final_response is not None and final_response.strip() == "":
            raise ValueError("final_response must not be empty.")

        if has_invocations == has_final_response:
            raise ValueError(
                "ParsedModelResponse must contain either invocations or final_response."
            )

        return self
