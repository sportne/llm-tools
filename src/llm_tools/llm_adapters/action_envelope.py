"""Canonical adapter for structured action-envelope interactions."""

from __future__ import annotations

import json
import re
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    create_model,
    model_validator,
)

from llm_tools.llm_adapters.base import ParsedModelResponse
from llm_tools.tool_api import ToolInvocationRequest, ToolSpec


class _LooseAction(BaseModel):
    """Fallback action model for local/offline payload parsing."""

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field(min_length=1)
    arguments: dict[str, Any] = Field(default_factory=dict)


class _EnvelopeModeMixin(BaseModel):
    """Enforce one-turn output mode consistency."""

    model_config = ConfigDict(extra="forbid")

    final_response: str | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> _EnvelopeModeMixin:
        actions = getattr(self, "actions", [])
        has_actions = len(actions) > 0
        has_final_response = self.final_response is not None

        if self.final_response is not None and self.final_response.strip() == "":
            raise ValueError("final_response must not be empty.")
        if has_actions == has_final_response:
            raise ValueError(
                "Action envelope must contain either actions or final_response."
            )
        return self


class _LooseEnvelope(_EnvelopeModeMixin):
    """Lax envelope used when no typed response model is provided."""

    actions: list[_LooseAction] = Field(default_factory=list)


class ActionEnvelopeAdapter:
    """Build and parse canonical structured action envelopes."""

    def build_response_model(
        self,
        specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
    ) -> type[BaseModel]:
        """Build a dynamic response model constrained to the visible tools."""
        action_models: list[type[BaseModel]] = []
        for spec in specs:
            input_model = input_models[spec.name]
            action_models.append(self._build_action_model(spec.name, input_model))

        if not action_models:

            class _NoToolEnvelope(_EnvelopeModeMixin):
                actions: list[dict[str, Any]] = Field(default_factory=list)

                @model_validator(mode="after")
                def _validate_no_actions(self) -> _NoToolEnvelope:
                    if self.actions:
                        raise ValueError(
                            "No tools are available; actions must be empty."
                        )
                    return self

            _NoToolEnvelope.__name__ = "ActionEnvelopeNoTools"
            return _NoToolEnvelope

        if len(action_models) == 1:
            action_union: type[BaseModel] | Any = action_models[0]
        else:
            action_union = action_models[0]
            for action_model in action_models[1:]:
                action_union = action_union | action_model
            action_union = Annotated[action_union, Field(discriminator="tool_name")]

        actions_annotation = list[action_union]  # type: ignore[valid-type]

        envelope_model = create_model(
            "ActionEnvelope",
            __base__=_EnvelopeModeMixin,
            actions=(actions_annotation, Field(default_factory=list)),
        )
        envelope_model.__module__ = __name__
        return envelope_model

    def export_tool_descriptions(
        self,
        specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
    ) -> dict[str, Any]:
        """Return the canonical envelope schema for inspection/debugging."""
        return self.export_schema(self.build_response_model(specs, input_models))

    def export_schema(self, response_model: type[BaseModel]) -> dict[str, Any]:
        """Return JSON schema for the supplied action-envelope model."""
        return response_model.model_json_schema()

    def parse_model_output(
        self,
        payload: object,
        *,
        response_model: type[BaseModel] | None = None,
    ) -> ParsedModelResponse:
        """Parse a payload into canonical invocations/final response."""
        model = response_model or _LooseEnvelope
        normalized = self._normalize_payload(payload)
        try:
            envelope = model.model_validate(normalized)
        except ValidationError as exc:
            raise ValueError("Invalid action envelope payload.") from exc

        actions_payload = getattr(envelope, "actions", [])
        invocations: list[ToolInvocationRequest] = []
        for action in actions_payload:
            action_obj = self._normalize_payload(action)
            if not isinstance(action_obj, dict):
                raise ValueError("Action payload must decode to an object.")

            tool_name = action_obj.get("tool_name")
            if not isinstance(tool_name, str) or tool_name.strip() == "":
                raise ValueError("Action payload is missing tool_name.")

            arguments = action_obj.get("arguments", {})
            if isinstance(arguments, BaseModel):
                arguments = arguments.model_dump(mode="json")
            if not isinstance(arguments, dict):
                raise ValueError("Action arguments must decode to an object.")

            invocations.append(
                ToolInvocationRequest(tool_name=tool_name, arguments=arguments)
            )

        final_response = getattr(envelope, "final_response", None)
        return ParsedModelResponse(
            invocations=invocations,
            final_response=final_response,
        )

    @staticmethod
    def _build_action_model(
        tool_name: str,
        input_model: type[BaseModel],
    ) -> type[BaseModel]:
        class_name = f"{_sanitize_name(tool_name)}Action"
        action_model = create_model(
            class_name,
            __config__=ConfigDict(extra="forbid"),
            tool_name=(Literal[tool_name], Field(default=tool_name)),
            arguments=(input_model, ...),
        )
        action_model.__module__ = __name__
        return action_model

    @staticmethod
    def _normalize_payload(payload: object) -> object:
        if isinstance(payload, str):
            try:
                return json.loads(payload)
            except json.JSONDecodeError:
                return payload

        if isinstance(payload, BaseModel):
            return payload.model_dump(mode="json", exclude_none=True)

        model_dump = getattr(payload, "model_dump", None)
        if callable(model_dump):
            return model_dump(mode="json", exclude_none=True)

        return payload


def _sanitize_name(name: str) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", name)
    normalized = "".join(part.capitalize() for part in parts if part)
    if normalized == "":
        return "Tool"
    if normalized[0].isdigit():
        return f"Tool{normalized}"
    return normalized
