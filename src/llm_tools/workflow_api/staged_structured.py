"""Shared staged structured agent protocol helpers."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ValidationError

from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import ToolSpec
from llm_tools.workflow_api.executor import PreparedModelInteraction

_StagedStepT = TypeVar("_StagedStepT")


class StagedStructuredProvider(Protocol):
    """Provider surface required for staged structured tool rounds."""

    def run_structured(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> object:
        """Return one structured payload for a staged workflow step."""

    async def run_structured_async(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> object:
        """Return one structured payload for a staged workflow step."""


class StagedStructuredToolRunner:
    """Run one staged structured decision/tool/final round."""

    def __init__(
        self,
        *,
        adapter: ActionEnvelopeAdapter,
        temperature: float,
        repair_attempts: int = 2,
    ) -> None:
        self._adapter = adapter
        self._temperature = temperature
        self._repair_attempts = repair_attempts

    def run_round(
        self,
        *,
        provider: StagedStructuredProvider,
        messages: list[dict[str, Any]],
        prepared: PreparedModelInteraction,
        final_response_model: object,
    ) -> ParsedModelResponse:
        """Run one decision/final or decision/tool structured round."""
        decision_model = self._adapter.build_decision_step_model(prepared.tool_specs)
        decision = self.run_step(
            provider=provider,
            stage_name="decision",
            messages=self.decision_stage_messages(
                base_messages=messages,
                tool_specs=prepared.tool_specs,
            ),
            response_model=decision_model,
            parser=lambda payload: decision_model.model_validate(payload),
        )
        if getattr(decision, "mode", None) == "finalize":
            return self.run_final_response_step(
                provider=provider,
                messages=messages,
                final_response_model=final_response_model,
            )
        return self.run_tool_invocation_step(
            provider=provider,
            messages=messages,
            prepared=prepared,
            tool_name=getattr(decision, "tool_name", None),
        )

    async def run_round_async(
        self,
        *,
        provider: StagedStructuredProvider,
        messages: list[dict[str, Any]],
        prepared: PreparedModelInteraction,
        final_response_model: object,
    ) -> ParsedModelResponse:
        """Run one decision/final or decision/tool structured round."""
        decision_model = self._adapter.build_decision_step_model(prepared.tool_specs)
        decision = await self.run_step_async(
            provider=provider,
            stage_name="decision",
            messages=self.decision_stage_messages(
                base_messages=messages,
                tool_specs=prepared.tool_specs,
            ),
            response_model=decision_model,
            parser=lambda payload: decision_model.model_validate(payload),
        )
        if getattr(decision, "mode", None) == "finalize":
            return await self.run_final_response_step_async(
                provider=provider,
                messages=messages,
                final_response_model=final_response_model,
            )
        return await self.run_tool_invocation_step_async(
            provider=provider,
            messages=messages,
            prepared=prepared,
            tool_name=getattr(decision, "tool_name", None),
        )

    def run_final_response_step(
        self,
        *,
        provider: StagedStructuredProvider,
        messages: list[dict[str, Any]],
        final_response_model: object,
    ) -> ParsedModelResponse:
        """Run one final response stage."""
        response_model = self._adapter.build_final_response_step_model(
            final_response_model=final_response_model
        )
        return self.run_step(
            provider=provider,
            stage_name="final_response",
            messages=self.final_response_stage_messages(
                base_messages=messages,
                response_model=response_model,
            ),
            response_model=response_model,
            parser=lambda payload: self._adapter.parse_final_response_step(
                payload,
                response_model=response_model,
            ),
        )

    async def run_final_response_step_async(
        self,
        *,
        provider: StagedStructuredProvider,
        messages: list[dict[str, Any]],
        final_response_model: object,
    ) -> ParsedModelResponse:
        """Run one final response stage."""
        response_model = self._adapter.build_final_response_step_model(
            final_response_model=final_response_model
        )
        return await self.run_step_async(
            provider=provider,
            stage_name="final_response",
            messages=self.final_response_stage_messages(
                base_messages=messages,
                response_model=response_model,
            ),
            response_model=response_model,
            parser=lambda payload: self._adapter.parse_final_response_step(
                payload,
                response_model=response_model,
            ),
        )

    def run_tool_invocation_step(
        self,
        *,
        provider: StagedStructuredProvider,
        messages: list[dict[str, Any]],
        prepared: PreparedModelInteraction,
        tool_name: object,
    ) -> ParsedModelResponse:
        """Run one selected tool invocation stage."""
        tool_spec, tool_input_model = self.selected_tool(
            tool_specs=prepared.tool_specs,
            input_models=prepared.input_models,
            tool_name=tool_name,
        )
        response_model = self._adapter.build_tool_invocation_step_model(
            tool_name=tool_spec.name,
            input_model=tool_input_model,
        )
        return self.run_step(
            provider=provider,
            stage_name=f"tool:{tool_spec.name}",
            messages=self.tool_invocation_stage_messages(
                base_messages=messages,
                tool_spec=tool_spec,
                tool_input_model=tool_input_model,
                response_model=response_model,
            ),
            response_model=response_model,
            parser=lambda payload: self._adapter.parse_tool_invocation_step(
                payload,
                response_model=response_model,
            ),
        )

    async def run_tool_invocation_step_async(
        self,
        *,
        provider: StagedStructuredProvider,
        messages: list[dict[str, Any]],
        prepared: PreparedModelInteraction,
        tool_name: object,
    ) -> ParsedModelResponse:
        """Run one selected tool invocation stage."""
        tool_spec, tool_input_model = self.selected_tool(
            tool_specs=prepared.tool_specs,
            input_models=prepared.input_models,
            tool_name=tool_name,
        )
        response_model = self._adapter.build_tool_invocation_step_model(
            tool_name=tool_spec.name,
            input_model=tool_input_model,
        )
        return await self.run_step_async(
            provider=provider,
            stage_name=f"tool:{tool_spec.name}",
            messages=self.tool_invocation_stage_messages(
                base_messages=messages,
                tool_spec=tool_spec,
                tool_input_model=tool_input_model,
                response_model=response_model,
            ),
            response_model=response_model,
            parser=lambda payload: self._adapter.parse_tool_invocation_step(
                payload,
                response_model=response_model,
            ),
        )

    def run_step(
        self,
        *,
        provider: StagedStructuredProvider,
        stage_name: str,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        parser: Callable[[object], _StagedStepT],
    ) -> _StagedStepT:
        """Run one structured stage with bounded repair retries."""
        attempt_messages = list(messages)
        repair_attempts = 0
        while True:
            payload: object | None = None
            try:
                payload = provider.run_structured(
                    messages=attempt_messages,
                    response_model=response_model,
                    request_params={"temperature": self._temperature},
                )
                return parser(payload)
            except Exception as exc:
                if (
                    repair_attempts >= self._repair_attempts
                    or not is_repairable_stage_error(exc)
                ):
                    raise
                repair_attempts += 1
                invalid_payload = payload
                if invalid_payload is None:
                    invalid_payload = getattr(exc, "invalid_payload", None)
                attempt_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": self.repair_stage_message(
                            stage_name=stage_name,
                            response_model=response_model,
                            error=exc,
                            invalid_payload=invalid_payload,
                        ),
                    },
                ]

    async def run_step_async(
        self,
        *,
        provider: StagedStructuredProvider,
        stage_name: str,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        parser: Callable[[object], _StagedStepT],
    ) -> _StagedStepT:
        """Run one structured stage with bounded repair retries."""
        attempt_messages = list(messages)
        repair_attempts = 0
        while True:
            payload: object | None = None
            try:
                payload = await provider.run_structured_async(
                    messages=attempt_messages,
                    response_model=response_model,
                    request_params={"temperature": self._temperature},
                )
                return parser(payload)
            except Exception as exc:
                if (
                    repair_attempts >= self._repair_attempts
                    or not is_repairable_stage_error(exc)
                ):
                    raise
                repair_attempts += 1
                invalid_payload = payload
                if invalid_payload is None:
                    invalid_payload = getattr(exc, "invalid_payload", None)
                attempt_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": self.repair_stage_message(
                            stage_name=stage_name,
                            response_model=response_model,
                            error=exc,
                            invalid_payload=invalid_payload,
                        ),
                    },
                ]

    def decision_stage_messages(
        self,
        *,
        base_messages: Sequence[dict[str, Any]],
        tool_specs: list[ToolSpec],
    ) -> list[dict[str, Any]]:
        """Return the small decision-stage prompt."""
        tool_catalog = json.dumps(
            self._adapter.export_compact_tool_catalog(tool_specs),
            indent=2,
            sort_keys=True,
        )
        return [
            *base_messages,
            {
                "role": "system",
                "content": (
                    "Current step: choose the next action.\n"
                    "Return mode='tool' with exactly one tool_name, or mode='finalize'.\n"
                    "Do not provide tool arguments or a final answer in this step.\n"
                    f"Available tools:\n{tool_catalog}"
                ),
            },
        ]

    def tool_invocation_stage_messages(
        self,
        *,
        base_messages: Sequence[dict[str, Any]],
        tool_spec: ToolSpec,
        tool_input_model: type[BaseModel],
        response_model: type[BaseModel],
    ) -> list[dict[str, Any]]:
        """Return the small selected-tool invocation prompt."""
        schema = json.dumps(
            self._adapter.export_schema(response_model),
            indent=2,
            sort_keys=True,
            default=str,
        )
        tool_schema = json.dumps(
            tool_input_model.model_json_schema(),
            indent=2,
            sort_keys=True,
            default=str,
        )
        return [
            *base_messages,
            {
                "role": "system",
                "content": (
                    f"Current step: invoke the selected tool '{tool_spec.name}'.\n"
                    f"Tool description: {tool_spec.description}\n"
                    "Return exactly one invocation for this tool.\n"
                    "Do not choose another tool and do not finalize in this step.\n"
                    f"Tool argument schema:\n{tool_schema}\n\n"
                    f"Full response schema for this step:\n{schema}"
                ),
            },
        ]

    def final_response_stage_messages(
        self,
        *,
        base_messages: Sequence[dict[str, Any]],
        response_model: type[BaseModel],
    ) -> list[dict[str, Any]]:
        """Return the small final-response prompt."""
        schema = json.dumps(
            self._adapter.export_schema(response_model),
            indent=2,
            sort_keys=True,
            default=str,
        )
        return [
            *base_messages,
            {
                "role": "system",
                "content": (
                    "Current step: finalize the answer.\n"
                    "Return only the final response and no tool invocation.\n"
                    "The response must satisfy this schema exactly:\n"
                    f"{schema}"
                ),
            },
        ]

    def repair_stage_message(
        self,
        *,
        stage_name: str,
        response_model: type[BaseModel],
        error: Exception,
        invalid_payload: object | None,
    ) -> str:
        """Return a schema-visible repair prompt for a malformed stage."""
        schema = json.dumps(
            self._adapter.export_schema(response_model),
            indent=2,
            sort_keys=True,
            default=str,
        )
        invalid_payload_text = format_invalid_payload(invalid_payload)
        return (
            f"The previous {stage_name} response was invalid.\n"
            "Correct the response for the same stage only.\n"
            f"{repair_stage_guidance(stage_name)}\n"
            f"Validation summary: {validation_error_summary(error)}\n"
            f"Previous invalid payload:\n{invalid_payload_text}\n"
            "Return a corrected payload that matches this schema exactly:\n"
            f"{schema}"
        )

    @staticmethod
    def selected_tool(
        *,
        tool_specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
        tool_name: object,
    ) -> tuple[ToolSpec, type[BaseModel]]:
        """Return the selected tool spec and input model."""
        if not isinstance(tool_name, str) or tool_name.strip() == "":
            raise ValueError("Decision stage did not select a valid tool name.")
        tool_spec = tool_spec_by_name(tool_specs, tool_name)
        tool_input_model = input_models.get(tool_name)
        if tool_input_model is None:
            raise ValueError(
                f"Selected tool '{tool_name}' was not prepared for this interaction."
            )
        return tool_spec, tool_input_model


def validation_error_summary(error: Exception) -> str:
    """Return a short, model-visible validation summary."""
    message = str(error).strip()
    return message or type(error).__name__


def is_repairable_stage_error(error: Exception) -> bool:
    """Return whether one staged failure should receive a repair prompt."""
    nonrepairable_markers = (
        "transport-related",
        "connection",
        "connect",
        "timed out",
        "timeout",
        "rate limit",
        "rate_limit",
        "unauthorized",
        "authentication",
        "api key",
        "forbidden",
        "model not found",
        "model_not_found",
        "name resolution",
        "dns",
        "refused",
        "unreachable",
    )
    repair_markers = (
        "validation",
        "parse",
        "json",
        "schema",
        "structured output",
        "invalid staged",
        "invalid action envelope",
    )
    for candidate in iter_exception_chain(error):
        if isinstance(candidate, (ValidationError, json.JSONDecodeError)):
            return True

        message = str(candidate).lower()
        if any(marker in message for marker in nonrepairable_markers):
            return False

        candidate_type = type(candidate)
        module_name = candidate_type.__module__
        if module_name.startswith("openai"):
            return False
        if module_name.startswith("pydantic"):
            return True

        class_name = candidate_type.__name__.lower()
        if isinstance(candidate, ValueError) and (
            any(marker in class_name for marker in repair_markers)
            or any(marker in message for marker in repair_markers)
        ):
            return True

        if module_name.startswith("instructor") and (
            any(marker in class_name for marker in repair_markers)
            or any(marker in message for marker in repair_markers)
        ):
            return True

    return False


def iter_exception_chain(error: Exception) -> list[BaseException]:
    """Return an exception plus any causal/context chain without duplicates."""
    seen: set[int] = set()
    pending: list[BaseException] = [error]
    chain: list[BaseException] = []
    while pending:
        current = pending.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        chain.append(current)

        cause = getattr(current, "__cause__", None)
        if isinstance(cause, BaseException):
            pending.append(cause)

        context = getattr(current, "__context__", None)
        if isinstance(context, BaseException):
            pending.append(context)

    return chain


def repair_stage_guidance(stage_name: str) -> str:
    """Return stage-specific repair rules."""
    if stage_name == "decision":
        return (
            "Decision stage rules: return only mode and, when mode='tool', one tool_name. "
            "Do not include arguments or final_response."
        )
    if stage_name.startswith("tool:"):
        return (
            "Tool stage rules: return exactly one invocation for the selected tool. "
            "Include mode='tool', the fixed tool_name, and arguments only."
        )
    if stage_name == "final_response":
        return (
            "Finalization stage rules: return only mode='finalize' and final_response. "
            "Do not include tool_name or arguments."
        )
    return "Return only the fields required for this stage."


def format_invalid_payload(invalid_payload: object | None) -> str:
    """Return a compact representation of the invalid payload."""
    if invalid_payload is None:
        return "(unavailable)"
    if isinstance(invalid_payload, str):
        return invalid_payload
    try:
        return json.dumps(invalid_payload, indent=2, sort_keys=True, default=str)
    except TypeError:
        return str(invalid_payload)


def tool_spec_by_name(tool_specs: list[ToolSpec], tool_name: str) -> ToolSpec:
    """Return one tool spec by name."""
    for spec in tool_specs:
        if spec.name == tool_name:
            return spec
    raise ValueError(f"Unknown tool selected during staged interaction: {tool_name}")
