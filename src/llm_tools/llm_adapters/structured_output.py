"""Structured-output adapter helpers and parser."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, model_validator

from llm_tools.llm_adapters.base import LLMAdapter, ParsedModelResponse
from llm_tools.tool_api import ToolInvocationRequest, ToolSpec


class StructuredToolAction(BaseModel):
    """One structured tool action emitted by a model."""

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class StructuredOutputEnvelope(BaseModel):
    """Canonical structured-output envelope for one model turn."""

    actions: list[StructuredToolAction] = Field(default_factory=list)
    final_response: str | None = None

    @model_validator(mode="after")
    def validate_mode(self) -> StructuredOutputEnvelope:
        """Require either actions or a final response."""
        has_actions = len(self.actions) > 0
        final_response = self.final_response
        has_final_response = final_response is not None

        if final_response is not None and final_response.strip() == "":
            raise ValueError("final_response must not be empty.")

        if has_actions == has_final_response:
            raise ValueError(
                "StructuredOutputEnvelope must contain either actions or "
                "final_response."
            )

        return self


def build_structured_output_schema(tool_names: list[str]) -> dict[str, Any]:
    """Create the canonical structured-output schema."""
    return {
        "title": "StructuredOutputEnvelope",
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "actions": {
                "type": "array",
                "default": [],
                "items": {
                    "title": "StructuredToolAction",
                    "type": "object",
                    "additionalProperties": False,
                    "properties": {
                        "tool_name": {
                            "type": "string",
                            "enum": tool_names,
                        },
                        "arguments": {
                            "type": "object",
                            "default": {},
                            "additionalProperties": True,
                        },
                    },
                    "required": ["tool_name"],
                },
            },
            "final_response": {
                "type": ["string", "null"],
                "default": None,
            },
        },
    }


def normalize_structured_output_payload(payload: object) -> dict[str, Any]:
    """Normalize convenience payload shapes into the canonical envelope."""
    normalized = payload
    if isinstance(normalized, str):
        try:
            normalized = json.loads(normalized)
        except json.JSONDecodeError as exc:
            raise ValueError("Structured output payload is not valid JSON.") from exc

    if isinstance(normalized, list):
        return {"actions": normalized, "final_response": None}

    if not isinstance(normalized, dict):
        raise ValueError("Structured output payload must decode to an object or list.")

    if "actions" in normalized or "final_response" in normalized:
        return dict(normalized)

    if "tool_name" in normalized:
        return {"actions": [normalized], "final_response": None}

    raise ValueError("Structured output payload does not match a supported shape.")


class StructuredOutputAdapter(LLMAdapter):
    """Adapter for structured JSON model output."""

    def export_tool_descriptions(
        self,
        specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
    ) -> dict[str, Any]:
        """Return the canonical structured-output schema."""
        del input_models
        tool_names = [spec.name for spec in specs]
        return build_structured_output_schema(tool_names)

    def parse_model_output(self, payload: object) -> ParsedModelResponse:
        """Parse structured payloads into a canonical model-turn outcome."""
        normalized = normalize_structured_output_payload(payload)
        envelope = StructuredOutputEnvelope.model_validate(normalized)

        return ParsedModelResponse(
            invocations=[
                ToolInvocationRequest(
                    tool_name=action.tool_name,
                    arguments=action.arguments,
                )
                for action in envelope.actions
            ],
            final_response=envelope.final_response,
        )
