"""Canonical adapter for structured action-envelope interactions."""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from typing import Annotated, Any, Literal, cast

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


class _SimplifiedEnvelopeMixin(BaseModel):
    """Lightweight envelope for JSON-schema-constrained providers."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["actions", "final"]

    @model_validator(mode="after")
    def _validate_mode(self) -> _SimplifiedEnvelopeMixin:
        actions = getattr(self, "actions", [])
        final_response = getattr(self, "final_response", None)
        if self.mode == "actions":
            if len(actions) == 0 or final_response is not None:
                raise ValueError(
                    "Simplified action envelope requires actions-only payloads in actions mode."
                )
            return self
        if len(actions) != 0 or final_response is None:
            raise ValueError(
                "Simplified action envelope requires final_response-only payloads in final mode."
            )
        return self


class _DecisionStepMixin(BaseModel):
    """Strict decision step for staged structured interaction."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["tool", "finalize"]
    tool_name: str | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> _DecisionStepMixin:
        if self.mode == "tool":
            if self.tool_name is None:
                raise ValueError("tool_name is required when mode='tool'.")
            return self
        if self.tool_name is not None:
            raise ValueError("tool_name is only allowed when mode='tool'.")
        return self


class _ToolInvocationStepMixin(BaseModel):
    """Strict tool-invocation step for staged structured interaction."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["tool"] = "tool"


class _FinalResponseStepMixin(BaseModel):
    """Strict finalization step for staged structured interaction."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["finalize"] = "finalize"


class _SingleActionStepMixin(BaseModel):
    """Single-stage structured agent step."""

    model_config = ConfigDict(extra="forbid")

    mode: Literal["tool", "finalize"]
    tool_name: str | None = None
    arguments: dict[str, Any] = Field(default_factory=dict)
    final_response: Any | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> _SingleActionStepMixin:
        if self.mode == "tool":
            if self.tool_name is None:
                raise ValueError("tool_name is required when mode='tool'.")
            if self.final_response is not None:
                raise ValueError("final_response is not allowed when mode='tool'.")
            return self
        if self.tool_name is not None:
            raise ValueError("tool_name is only allowed when mode='tool'.")
        if self.arguments:
            raise ValueError("arguments are only allowed when mode='tool'.")
        if self.final_response is None:
            raise ValueError("final_response is required when mode='finalize'.")
        return self


class _EnvelopeModeMixin(BaseModel):
    """Enforce one-turn output mode consistency."""

    model_config = ConfigDict(extra="forbid")
    final_response: Any | None = None

    @model_validator(mode="after")
    def _validate_mode(self) -> _EnvelopeModeMixin:
        actions = getattr(self, "actions", [])
        has_actions = len(actions) > 0
        has_final_response = self.final_response is not None

        if isinstance(self.final_response, str) and self.final_response.strip() == "":
            raise ValueError("final_response must not be empty.")
        if has_actions == has_final_response:
            raise ValueError(
                "Action envelope must contain either actions or final_response."
            )
        return self


class _LooseEnvelope(_EnvelopeModeMixin):
    """Lax envelope used when no typed response model is provided."""

    actions: list[_LooseAction] = Field(default_factory=list)
    final_response: Any | None = None


class _NoActionsEnvelopeMixin(_EnvelopeModeMixin):
    """Reject action payloads when no tools are exposed."""

    @model_validator(mode="after")
    def _validate_no_actions(self) -> _NoActionsEnvelopeMixin:
        if getattr(self, "actions", []):
            raise ValueError("No tools are available; actions must be empty.")
        return self


class ActionEnvelopeAdapter:
    """Build and parse canonical structured action envelopes."""

    def build_response_model(
        self,
        specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
        *,
        final_response_model: object = str,
        simplify_json_schema: bool = False,
    ) -> type[BaseModel]:
        """Build a dynamic response model constrained to the visible tools."""
        if simplify_json_schema:
            return self._build_simplified_response_model(
                final_response_model=final_response_model
            )
        action_models: list[type[BaseModel]] = []
        for spec in specs:
            input_model = input_models[spec.name]
            action_models.append(self._build_action_model(spec.name, input_model))

        if not action_models:
            envelope_model = cast(
                type[BaseModel],
                create_model(
                    "ActionEnvelopeNoTools",
                    __base__=_NoActionsEnvelopeMixin,
                    actions=(list[dict[str, Any]], Field(default_factory=list)),
                    final_response=(_optional_annotation(final_response_model), None),
                ),
            )
            envelope_model.__module__ = __name__
            return envelope_model

        if len(action_models) == 1:
            action_union: type[BaseModel] | Any = action_models[0]
        else:
            action_union = action_models[0]
            for action_model in action_models[1:]:
                action_union = action_union | action_model
            action_union = Annotated[action_union, Field(discriminator="tool_name")]

        actions_annotation = list[action_union]  # type: ignore[valid-type]

        envelope_model = cast(
            type[BaseModel],
            create_model(
                "ActionEnvelope",
                __base__=_EnvelopeModeMixin,
                actions=(actions_annotation, Field(default_factory=list)),
                final_response=(_optional_annotation(final_response_model), None),
            ),
        )
        envelope_model.__module__ = __name__
        return envelope_model

    def export_tool_descriptions(
        self,
        specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
        *,
        final_response_model: object = str,
    ) -> dict[str, Any]:
        """Return the canonical envelope schema for inspection/debugging."""
        return self.export_schema(
            self.build_response_model(
                specs,
                input_models,
                final_response_model=final_response_model,
            )
        )

    def build_decision_step_model(self, specs: list[ToolSpec]) -> type[BaseModel]:
        """Build the strict staged decision model for one tool-or-finalize step."""
        allowed_tool_names = [spec.name for spec in specs]
        mode_annotation = (
            Literal["tool", "finalize"] if allowed_tool_names else Literal["finalize"]
        )
        tool_name_annotation = (
            _literal_annotation(allowed_tool_names, allow_none=True)
            if allowed_tool_names
            else type(None)
        )
        model = cast(
            type[BaseModel],
            create_model(
                "DecisionStep",
                __base__=_DecisionStepMixin,
                mode=(mode_annotation, ...),
                tool_name=(tool_name_annotation, None),
            ),
        )
        model.__module__ = __name__
        return model

    def build_tool_invocation_step_model(
        self,
        *,
        tool_name: str,
        input_model: type[BaseModel],
    ) -> type[BaseModel]:
        """Build the strict staged invocation model for one selected tool."""
        model = cast(
            type[BaseModel],
            create_model(
                f"{_sanitize_name(tool_name)}InvocationStep",
                __base__=_ToolInvocationStepMixin,
                tool_name=(Literal[tool_name], Field(default=tool_name)),
                arguments=(input_model, ...),
            ),
        )
        model.__module__ = __name__
        return model

    def build_final_response_step_model(
        self,
        *,
        final_response_model: object,
    ) -> type[BaseModel]:
        """Build the strict staged finalization model for one final answer."""
        model = cast(
            type[BaseModel],
            create_model(
                "FinalResponseStep",
                __base__=_FinalResponseStepMixin,
                final_response=(final_response_model, ...),
            ),
        )
        model.__module__ = __name__
        return model

    def build_single_action_step_model(
        self,
        specs: list[ToolSpec],
        *,
        final_response_model: object,
    ) -> type[BaseModel]:
        """Build a single-stage one-tool-or-final response model."""
        allowed_tool_names = [spec.name for spec in specs]
        tool_name_annotation = (
            _literal_annotation(allowed_tool_names, allow_none=True)
            if allowed_tool_names
            else type(None)
        )
        model = cast(
            type[BaseModel],
            create_model(
                "SingleActionStep",
                __base__=_SingleActionStepMixin,
                tool_name=(tool_name_annotation, None),
                final_response=(_optional_annotation(final_response_model), None),
            ),
        )
        model.__module__ = __name__
        return model

    def parse_tool_invocation_step(
        self,
        payload: object,
        *,
        response_model: type[BaseModel],
    ) -> ParsedModelResponse:
        """Parse one strict staged tool-invocation payload."""
        normalized = self._normalize_payload(payload)
        try:
            step = response_model.model_validate(normalized)
        except ValidationError as exc:
            raise ValueError("Invalid staged tool-invocation payload.") from exc
        step_payload = cast(Any, step)
        return ParsedModelResponse(
            invocations=[
                ToolInvocationRequest(
                    tool_name=step_payload.tool_name,
                    arguments=self._normalize_arguments(step_payload.arguments),
                )
            ]
        )

    def parse_final_response_step(
        self,
        payload: object,
        *,
        response_model: type[BaseModel],
    ) -> ParsedModelResponse:
        """Parse one strict staged finalization payload."""
        normalized = self._normalize_payload(payload)
        try:
            step = response_model.model_validate(normalized)
        except ValidationError as exc:
            raise ValueError("Invalid staged final-response payload.") from exc
        return ParsedModelResponse(
            final_response=self._normalize_final_response(
                getattr(step, "final_response", None)
            )
        )

    def parse_single_action_step(
        self,
        payload: object,
        *,
        response_model: type[BaseModel],
        tool_specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
    ) -> ParsedModelResponse:
        """Parse one single-stage structured tool-or-final payload."""
        normalized = self._normalize_payload(payload)
        try:
            step = response_model.model_validate(normalized)
        except ValidationError as exc:
            raise ValueError("Invalid single-action step payload.") from exc
        step_payload = cast(Any, step)
        if step_payload.mode == "finalize":
            return ParsedModelResponse(
                final_response=self._normalize_final_response(
                    step_payload.final_response
                )
            )

        tool_name = step_payload.tool_name
        allowed_tool_names = {spec.name for spec in tool_specs}
        if tool_name not in allowed_tool_names:
            raise ValueError(f"Unknown tool selected: {tool_name}")
        input_model = input_models.get(tool_name)
        if input_model is None:
            raise ValueError(f"Selected tool '{tool_name}' was not prepared.")
        arguments = input_model.model_validate(
            self._normalize_arguments(step_payload.arguments)
        ).model_dump(mode="json")
        return ParsedModelResponse(
            invocations=[
                ToolInvocationRequest(tool_name=tool_name, arguments=arguments)
            ]
        )

    @staticmethod
    def export_compact_tool_catalog(specs: list[ToolSpec]) -> list[dict[str, str]]:
        """Return compact tool summaries for staged decision prompts."""
        return [{"name": spec.name, "description": spec.description} for spec in specs]

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
            tool_name, arguments = self._extract_action_fields(action)
            invocations.append(
                ToolInvocationRequest(tool_name=tool_name, arguments=arguments)
            )

        return ParsedModelResponse(
            invocations=invocations,
            final_response=self._normalize_final_response(
                getattr(envelope, "final_response", None)
            ),
        )

    @staticmethod
    def _extract_action_fields(action: object) -> tuple[str, dict[str, Any]]:
        if isinstance(action, BaseModel):
            tool_name = getattr(action, "tool_name", None)
            arguments = getattr(action, "arguments", {})
        elif isinstance(action, Mapping):
            tool_name = action.get("tool_name")
            arguments = action.get("arguments", {})
        else:
            raise ValueError("Action payload must decode to an object.")

        if not isinstance(tool_name, str) or tool_name.strip() == "":
            raise ValueError("Action payload is missing tool_name.")

        return tool_name, ActionEnvelopeAdapter._normalize_arguments(arguments)

    @staticmethod
    def _normalize_arguments(arguments: object) -> dict[str, Any]:
        if isinstance(arguments, BaseModel):
            return arguments.model_dump(mode="json")

        if isinstance(arguments, Mapping):
            return dict(arguments)

        raise ValueError("Action arguments must decode to an object.")

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

    def _build_simplified_response_model(
        self,
        *,
        final_response_model: object,
    ) -> type[BaseModel]:
        simplified_final_response = self._simplified_final_response_annotation(
            final_response_model
        )
        envelope_model = cast(
            type[BaseModel],
            create_model(
                "ActionEnvelopeSimplified",
                __base__=_SimplifiedEnvelopeMixin,
                actions=(list[_LooseAction], Field(default_factory=list)),
                final_response=(simplified_final_response, None),
            ),
        )
        envelope_model.__module__ = __name__
        return envelope_model

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

    @staticmethod
    def _normalize_final_response(final_response: object) -> object:
        if isinstance(final_response, BaseModel):
            return final_response.model_dump(mode="json", exclude_none=True)
        return final_response

    @staticmethod
    def _simplified_final_response_annotation(final_response_model: object) -> object:
        if final_response_model is str:
            return str | None
        if isinstance(final_response_model, type) and issubclass(
            final_response_model, BaseModel
        ):
            return dict[str, Any] | None
        return _optional_annotation(final_response_model)


def _sanitize_name(name: str) -> str:
    parts = re.split(r"[^0-9A-Za-z]+", name)
    normalized = "".join(part.capitalize() for part in parts if part)
    if normalized == "":
        return "Tool"
    if normalized[0].isdigit():
        return f"Tool{normalized}"
    return normalized


def _optional_annotation(annotation: object) -> object:
    return annotation | None  # type: ignore[operator]


def _literal_annotation(values: list[str], *, allow_none: bool) -> object:
    annotation: object = Literal.__getitem__(tuple(values))
    if allow_none:
        return annotation | None  # type: ignore[operator]
    return annotation
