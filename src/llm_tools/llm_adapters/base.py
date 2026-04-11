"""Shared abstractions for LLM-facing adapter layers."""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel, Field, model_validator

from llm_tools.tool_api import ToolInvocationRequest, ToolSpec


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


class ToolExposureAdapter(ABC):
    """Export canonical tool definitions into an LLM-facing format."""

    @abstractmethod
    def export_tool_descriptions(
        self,
        specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
    ) -> object:
        """Return an adapter-specific representation of available tools."""


class ModelOutputParsingAdapter(ABC):
    """Parse model output into a normalized turn outcome."""

    @abstractmethod
    def parse_model_output(self, payload: object) -> ParsedModelResponse:
        """Normalize model output into tool invocations or a final reply."""


class LLMAdapter(ToolExposureAdapter, ModelOutputParsingAdapter, ABC):
    """Combined adapter interface for tool exposure and model parsing."""
