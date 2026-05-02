"""Shared model-turn protocol for provider-facing workflow turns."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Protocol, TypeVar, cast

from pydantic import BaseModel

from llm_tools.llm_adapters import (
    ActionEnvelopeAdapter,
    ParsedModelResponse,
    PromptToolAdapter,
    PromptToolCategory,
)
from llm_tools.tool_api import ProtectionProvenanceSnapshot, ToolSpec
from llm_tools.tool_api.redaction import RedactionConfig, RedactionTarget, Redactor
from llm_tools.workflow_api.executor import PreparedModelInteraction
from llm_tools.workflow_api.protection import ProtectionAction, ProtectionController
from llm_tools.workflow_api.staged_structured import (
    StagedStructuredToolRunner,
    is_repairable_stage_error,
    repair_stage_guidance,
    tool_spec_by_name,
)

ModelTurnProtocolMode = Literal["auto", "final_response"]
ModelTurnProtocolKind = Literal[
    "native",
    "staged",
    "prompt_tools",
    "protection",
]
ModelTurnProtocolEventKind = Literal[
    "stage_start",
    "stage_repair",
    "fallback",
    "provider_call_complete",
    "protection_action",
    "parsed_response",
]

_StagedStepT = TypeVar("_StagedStepT")
_PROMPT_TOOL_STRATEGY_ENV = "LLM_TOOLS_PROMPT_TOOL_STRATEGY"
_PROMPT_TOOL_CATEGORY_THRESHOLD_ENV = "LLM_TOOLS_PROMPT_TOOL_CATEGORY_THRESHOLD"
_JSON_AGENT_STRATEGY_ENV = "LLM_TOOLS_JSON_AGENT_STRATEGY"
_DEFAULT_PROMPT_TOOL_CATEGORY_THRESHOLD = 7
_PROMPT_TOOL_SINGLE_ACTION_STRATEGIES = {"single_action", "single-action", "single"}
_PROMPT_TOOL_SPLIT_STRATEGIES = {"split", "decision", "decision_tool"}
_CONTEXT_LIMIT_ERROR_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "max context length",
    "context length",
    "too many tokens",
    "prompt is too long",
    "prompt too long",
    "token limit",
    "tokens exceeds",
    "input is too long",
    "request too large",
)


class ModelTurnProtocolProvider(Protocol):
    """Provider surface used by the model-turn protocol runner."""

    def run(
        self,
        *,
        adapter: ActionEnvelopeAdapter,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        """Return one parsed native structured response."""


@dataclass(slots=True, frozen=True)
class ModelTurnProtocolEvent:
    """One redacted protocol progress event."""

    kind: ModelTurnProtocolEventKind
    protocol: ModelTurnProtocolKind
    stage_name: str | None = None
    payload: dict[str, object] = field(default_factory=dict)


@dataclass(slots=True)
class ModelTurnProtectionContext:
    """Optional protection inputs for one model-turn protocol run."""

    controller: ProtectionController
    provenance: ProtectionProvenanceSnapshot
    metadata_sink: dict[str, object] | None = None
    original_user_message: str | None = None
    session_id: str | None = None


@dataclass(slots=True)
class ModelTurnProtocolRequest:
    """Inputs required to produce one parsed model turn."""

    provider: object
    messages: list[dict[str, Any]]
    prepared_interaction: PreparedModelInteraction
    adapter: ActionEnvelopeAdapter
    final_response_model: object
    temperature: float
    decision_context: str | None = None
    protection: ModelTurnProtectionContext | None = None
    redaction_config: RedactionConfig | None = None
    mode: ModelTurnProtocolMode = "auto"
    use_json_single_action_strategy: bool | None = None


ModelTurnProtocolObserver = Callable[[ModelTurnProtocolEvent], object]


class ModelTurnProtocolRunner:
    """Run one provider-facing model-turn protocol and return one parsed turn."""

    def __init__(self, *, repair_attempts: int = 2) -> None:
        self._repair_attempts = repair_attempts

    def run(
        self,
        request: ModelTurnProtocolRequest,
        *,
        observer: ModelTurnProtocolObserver | None = None,
    ) -> ParsedModelResponse:
        """Run one synchronous model-turn protocol."""
        messages = self._apply_prompt_protection(request, observer=observer)
        if isinstance(messages, ParsedModelResponse):
            self._emit_parsed_response(messages, observer=observer)
            return messages
        parsed = self._run_unprotected(
            request,
            messages=messages,
            observer=observer,
        )
        parsed = self._apply_response_protection(
            request,
            parsed=parsed,
            observer=observer,
        )
        self._emit_parsed_response(parsed, observer=observer)
        return parsed

    async def run_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        observer: ModelTurnProtocolObserver | None = None,
    ) -> ParsedModelResponse:
        """Run one asynchronous model-turn protocol."""
        messages = await self._apply_prompt_protection_async(request, observer=observer)
        if isinstance(messages, ParsedModelResponse):
            self._emit_parsed_response(messages, observer=observer)
            return messages
        parsed = await self._run_unprotected_async(
            request,
            messages=messages,
            observer=observer,
        )
        parsed = await self._apply_response_protection_async(
            request,
            parsed=parsed,
            observer=observer,
        )
        self._emit_parsed_response(parsed, observer=observer)
        return parsed

    def _run_unprotected(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        if self._uses_prompt_tool_protocol(request.provider):
            return self._run_prompt_tool(
                request,
                messages=messages,
                observer=observer,
            )
        if self._uses_staged_schema_protocol(request.provider):
            try:
                return self._run_staged(
                    request,
                    messages=messages,
                    observer=observer,
                )
            except Exception as exc:
                if not self._can_fallback_to_prompt_tools(request.provider, exc):
                    raise
                self._emit(
                    "fallback",
                    "prompt_tools",
                    payload={
                        "from_protocol": "staged",
                        "error_type": type(exc).__name__,
                        "error_summary": _redacted_error_summary(
                            exc,
                            request.redaction_config,
                        ),
                    },
                    observer=observer,
                    redaction_config=request.redaction_config,
                )
                return self._run_prompt_tool(
                    request,
                    messages=self._prompt_tool_base_messages(messages),
                    observer=observer,
                )
        return self._run_native(request, messages=messages, observer=observer)

    async def _run_unprotected_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        if self._uses_prompt_tool_protocol(request.provider):
            return await self._run_prompt_tool_async(
                request,
                messages=messages,
                observer=observer,
            )
        if self._uses_staged_schema_protocol(request.provider):
            try:
                return await self._run_staged_async(
                    request,
                    messages=messages,
                    observer=observer,
                )
            except Exception as exc:
                if not self._can_fallback_to_prompt_tools(request.provider, exc):
                    raise
                self._emit(
                    "fallback",
                    "prompt_tools",
                    payload={
                        "from_protocol": "staged",
                        "error_type": type(exc).__name__,
                        "error_summary": _redacted_error_summary(
                            exc,
                            request.redaction_config,
                        ),
                    },
                    observer=observer,
                    redaction_config=request.redaction_config,
                )
                return await self._run_prompt_tool_async(
                    request,
                    messages=self._prompt_tool_base_messages(messages),
                    observer=observer,
                )
        return await self._run_native_async(
            request, messages=messages, observer=observer
        )

    def _run_native(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        if request.mode == "final_response":
            return self._run_native_final_response(
                request,
                messages=messages,
                observer=observer,
            )
        native_messages = self._native_tool_round_messages(
            messages=messages,
            decision_context=request.decision_context,
        )
        self._emit(
            "stage_start",
            "native",
            stage_name="action",
            payload={"message_count": len(native_messages)},
            observer=observer,
            redaction_config=request.redaction_config,
        )
        try:
            run = cast(Any, request.provider).run
            parsed = run(
                adapter=request.adapter,
                messages=native_messages,
                response_model=request.prepared_interaction.response_model,
                request_params={"temperature": request.temperature},
            )
        except Exception as exc:
            if not self._can_fallback_to_prompt_tools(request.provider, exc):
                raise
            self._emit(
                "fallback",
                "prompt_tools",
                payload={
                    "from_protocol": "native",
                    "error_type": type(exc).__name__,
                    "error_summary": _redacted_error_summary(
                        exc,
                        request.redaction_config,
                    ),
                },
                observer=observer,
                redaction_config=request.redaction_config,
            )
            return self._run_prompt_tool(
                request,
                messages=self._prompt_tool_base_messages(messages),
                observer=observer,
            )
        self._emit_provider_call_complete(
            "native",
            stage_name="action",
            observer=observer,
            redaction_config=request.redaction_config,
        )
        return cast(ParsedModelResponse, parsed)

    async def _run_native_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        if request.mode == "final_response":
            return await self._run_native_final_response_async(
                request,
                messages=messages,
                observer=observer,
            )
        native_messages = self._native_tool_round_messages(
            messages=messages,
            decision_context=request.decision_context,
        )
        self._emit(
            "stage_start",
            "native",
            stage_name="action",
            payload={"message_count": len(native_messages)},
            observer=observer,
            redaction_config=request.redaction_config,
        )
        run_async = cast(Any, request.provider).run_async
        try:
            parsed = await run_async(
                adapter=request.adapter,
                messages=native_messages,
                response_model=request.prepared_interaction.response_model,
                request_params={"temperature": request.temperature},
            )
        except Exception as exc:
            if not self._can_fallback_to_prompt_tools(request.provider, exc):
                raise
            self._emit(
                "fallback",
                "prompt_tools",
                payload={
                    "from_protocol": "native",
                    "error_type": type(exc).__name__,
                    "error_summary": _redacted_error_summary(
                        exc,
                        request.redaction_config,
                    ),
                },
                observer=observer,
                redaction_config=request.redaction_config,
            )
            return await self._run_prompt_tool_async(
                request,
                messages=self._prompt_tool_base_messages(messages),
                observer=observer,
            )
        self._emit_provider_call_complete(
            "native",
            stage_name="action",
            observer=observer,
            redaction_config=request.redaction_config,
        )
        return cast(ParsedModelResponse, parsed)

    def _run_native_final_response(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        run_structured = getattr(request.provider, "run_structured", None)
        if callable(run_structured):
            try:
                self._emit(
                    "stage_start",
                    "native",
                    stage_name="final_response",
                    payload={"message_count": len(messages)},
                    observer=observer,
                    redaction_config=request.redaction_config,
                )
                payload = run_structured(
                    messages=messages,
                    response_model=request.final_response_model,
                    request_params={"temperature": request.temperature},
                )
                self._emit_provider_call_complete(
                    "native",
                    stage_name="final_response",
                    observer=observer,
                    redaction_config=request.redaction_config,
                )
                return ParsedModelResponse(
                    final_response=_final_response_payload(
                        payload,
                        request.final_response_model,
                    )
                )
            except Exception as exc:
                if not is_repairable_stage_error(exc):
                    raise
        run_text = getattr(request.provider, "run_text", None)
        if not callable(run_text):
            raise RuntimeError(
                "Native forced finalization requires run_structured() or run_text()."
            )
        self._emit(
            "stage_start",
            "native",
            stage_name="final_response_text",
            payload={"message_count": len(messages)},
            observer=observer,
            redaction_config=request.redaction_config,
        )
        text = run_text(
            messages=messages,
            request_params={"temperature": request.temperature},
        )
        self._emit_provider_call_complete(
            "native",
            stage_name="final_response_text",
            observer=observer,
            redaction_config=request.redaction_config,
        )
        return ParsedModelResponse(
            final_response=_safe_final_response_payload(
                text,
                request.final_response_model,
            )
        )

    async def _run_native_final_response_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        run_structured_async = getattr(request.provider, "run_structured_async", None)
        if callable(run_structured_async):
            try:
                self._emit(
                    "stage_start",
                    "native",
                    stage_name="final_response",
                    payload={"message_count": len(messages)},
                    observer=observer,
                    redaction_config=request.redaction_config,
                )
                payload = await run_structured_async(
                    messages=messages,
                    response_model=request.final_response_model,
                    request_params={"temperature": request.temperature},
                )
                self._emit_provider_call_complete(
                    "native",
                    stage_name="final_response",
                    observer=observer,
                    redaction_config=request.redaction_config,
                )
                return ParsedModelResponse(
                    final_response=_final_response_payload(
                        payload,
                        request.final_response_model,
                    )
                )
            except Exception as exc:
                if not is_repairable_stage_error(exc):
                    raise
        run_text_async = getattr(request.provider, "run_text_async", None)
        if not callable(run_text_async):
            raise RuntimeError(
                "Native forced finalization requires run_structured_async() or "
                "run_text_async()."
            )
        self._emit(
            "stage_start",
            "native",
            stage_name="final_response_text",
            payload={"message_count": len(messages)},
            observer=observer,
            redaction_config=request.redaction_config,
        )
        text = await run_text_async(
            messages=messages,
            request_params={"temperature": request.temperature},
        )
        self._emit_provider_call_complete(
            "native",
            stage_name="final_response_text",
            observer=observer,
            redaction_config=request.redaction_config,
        )
        return ParsedModelResponse(
            final_response=_safe_final_response_payload(
                text,
                request.final_response_model,
            )
        )

    def _run_staged(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        provider = self._structured_provider(request.provider)
        staged = StagedStructuredToolRunner(
            adapter=request.adapter,
            temperature=request.temperature,
            repair_attempts=self._repair_attempts,
        )
        if request.mode == "final_response":
            return self._run_staged_final_response(
                request,
                provider=provider,
                staged=staged,
                messages=messages,
                observer=observer,
            )
        if self._uses_json_single_action_strategy(request):
            response_model = request.adapter.build_single_action_step_model(
                request.prepared_interaction.tool_specs,
                final_response_model=request.final_response_model,
            )
            parsed = self._run_staged_step(
                request,
                provider=provider,
                staged=staged,
                stage_name="single_action",
                messages=staged.single_action_stage_messages(
                    base_messages=messages,
                    tool_specs=request.prepared_interaction.tool_specs,
                    response_model=response_model,
                    decision_context=request.decision_context,
                ),
                response_model=response_model,
                parser=lambda payload: request.adapter.parse_single_action_step(
                    payload,
                    response_model=response_model,
                    tool_specs=request.prepared_interaction.tool_specs,
                    input_models=request.prepared_interaction.input_models,
                ),
                observer=observer,
            )
            if parsed.invocations:
                self._emit_preparing(parsed.invocations[0].tool_name, observer)
            return parsed
        decision_model = request.adapter.build_decision_step_model(
            request.prepared_interaction.tool_specs
        )
        decision = self._run_staged_step(
            request,
            provider=provider,
            staged=staged,
            stage_name="decision",
            messages=staged.decision_stage_messages(
                base_messages=messages,
                tool_specs=request.prepared_interaction.tool_specs,
                decision_context=request.decision_context,
            ),
            response_model=decision_model,
            parser=lambda payload: decision_model.model_validate(payload),
            observer=observer,
        )
        if getattr(decision, "mode", None) == "finalize":
            return self._run_staged_final_response(
                request,
                provider=provider,
                staged=staged,
                messages=messages,
                observer=observer,
            )
        tool_spec, tool_input_model = staged.selected_tool(
            tool_specs=request.prepared_interaction.tool_specs,
            input_models=request.prepared_interaction.input_models,
            tool_name=getattr(decision, "tool_name", None),
        )
        self._emit_preparing(tool_spec.name, observer)
        response_model = request.adapter.build_tool_invocation_step_model(
            tool_name=tool_spec.name,
            input_model=tool_input_model,
        )
        return self._run_staged_step(
            request,
            provider=provider,
            staged=staged,
            stage_name=f"tool:{tool_spec.name}",
            messages=staged.tool_invocation_stage_messages(
                base_messages=messages,
                tool_spec=tool_spec,
                tool_input_model=tool_input_model,
                response_model=response_model,
            ),
            response_model=response_model,
            parser=lambda payload: request.adapter.parse_tool_invocation_step(
                payload,
                response_model=response_model,
            ),
            observer=observer,
        )

    async def _run_staged_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        provider = self._structured_provider(request.provider)
        staged = StagedStructuredToolRunner(
            adapter=request.adapter,
            temperature=request.temperature,
            repair_attempts=self._repair_attempts,
        )
        if request.mode == "final_response":
            return await self._run_staged_final_response_async(
                request,
                provider=provider,
                staged=staged,
                messages=messages,
                observer=observer,
            )
        if self._uses_json_single_action_strategy(request):
            response_model = request.adapter.build_single_action_step_model(
                request.prepared_interaction.tool_specs,
                final_response_model=request.final_response_model,
            )
            parsed = await self._run_staged_step_async(
                request,
                provider=provider,
                staged=staged,
                stage_name="single_action",
                messages=staged.single_action_stage_messages(
                    base_messages=messages,
                    tool_specs=request.prepared_interaction.tool_specs,
                    response_model=response_model,
                    decision_context=request.decision_context,
                ),
                response_model=response_model,
                parser=lambda payload: request.adapter.parse_single_action_step(
                    payload,
                    response_model=response_model,
                    tool_specs=request.prepared_interaction.tool_specs,
                    input_models=request.prepared_interaction.input_models,
                ),
                observer=observer,
            )
            if parsed.invocations:
                self._emit_preparing(parsed.invocations[0].tool_name, observer)
            return parsed
        decision_model = request.adapter.build_decision_step_model(
            request.prepared_interaction.tool_specs
        )
        decision = await self._run_staged_step_async(
            request,
            provider=provider,
            staged=staged,
            stage_name="decision",
            messages=staged.decision_stage_messages(
                base_messages=messages,
                tool_specs=request.prepared_interaction.tool_specs,
                decision_context=request.decision_context,
            ),
            response_model=decision_model,
            parser=lambda payload: decision_model.model_validate(payload),
            observer=observer,
        )
        if getattr(decision, "mode", None) == "finalize":
            return await self._run_staged_final_response_async(
                request,
                provider=provider,
                staged=staged,
                messages=messages,
                observer=observer,
            )
        tool_spec, tool_input_model = staged.selected_tool(
            tool_specs=request.prepared_interaction.tool_specs,
            input_models=request.prepared_interaction.input_models,
            tool_name=getattr(decision, "tool_name", None),
        )
        self._emit_preparing(tool_spec.name, observer)
        response_model = request.adapter.build_tool_invocation_step_model(
            tool_name=tool_spec.name,
            input_model=tool_input_model,
        )
        return await self._run_staged_step_async(
            request,
            provider=provider,
            staged=staged,
            stage_name=f"tool:{tool_spec.name}",
            messages=staged.tool_invocation_stage_messages(
                base_messages=messages,
                tool_spec=tool_spec,
                tool_input_model=tool_input_model,
                response_model=response_model,
            ),
            response_model=response_model,
            parser=lambda payload: request.adapter.parse_tool_invocation_step(
                payload,
                response_model=response_model,
            ),
            observer=observer,
        )

    def _run_staged_final_response(
        self,
        request: ModelTurnProtocolRequest,
        *,
        provider: object,
        staged: StagedStructuredToolRunner,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        response_model = request.adapter.build_final_response_step_model(
            final_response_model=request.final_response_model
        )
        return self._run_staged_step(
            request,
            provider=provider,
            staged=staged,
            stage_name="final_response",
            messages=staged.final_response_stage_messages(
                base_messages=messages,
                response_model=response_model,
            ),
            response_model=response_model,
            parser=lambda payload: request.adapter.parse_final_response_step(
                payload,
                response_model=response_model,
            ),
            observer=observer,
        )

    async def _run_staged_final_response_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        provider: object,
        staged: StagedStructuredToolRunner,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        response_model = request.adapter.build_final_response_step_model(
            final_response_model=request.final_response_model
        )
        return await self._run_staged_step_async(
            request,
            provider=provider,
            staged=staged,
            stage_name="final_response",
            messages=staged.final_response_stage_messages(
                base_messages=messages,
                response_model=response_model,
            ),
            response_model=response_model,
            parser=lambda payload: request.adapter.parse_final_response_step(
                payload,
                response_model=response_model,
            ),
            observer=observer,
        )

    def _run_staged_step(
        self,
        request: ModelTurnProtocolRequest,
        *,
        provider: object,
        staged: StagedStructuredToolRunner,
        stage_name: str,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        parser: Callable[[object], _StagedStepT],
        observer: ModelTurnProtocolObserver | None,
    ) -> _StagedStepT:
        run_structured = cast(Any, provider).run_structured
        attempt_messages = list(messages)
        repair_attempts = 0
        while True:
            self._emit(
                "stage_start",
                "staged",
                stage_name=stage_name,
                payload={"message_count": len(attempt_messages)},
                observer=observer,
                redaction_config=request.redaction_config,
            )
            payload: object | None = None
            try:
                payload = run_structured(
                    messages=attempt_messages,
                    response_model=response_model,
                    request_params={"temperature": request.temperature},
                )
                self._emit_provider_call_complete(
                    "staged",
                    stage_name=stage_name,
                    observer=observer,
                    redaction_config=request.redaction_config,
                )
                return parser(payload)
            except Exception as exc:
                if (
                    repair_attempts >= self._repair_attempts
                    or not is_repairable_stage_error(exc)
                ):
                    raise
                repair_attempts += 1
                self._emit_repair(request, "staged", stage_name, exc, observer)
                invalid_payload = payload
                if invalid_payload is None:
                    invalid_payload = getattr(exc, "invalid_payload", None)
                attempt_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": staged.repair_stage_message(
                            stage_name=stage_name,
                            response_model=response_model,
                            error=exc,
                            invalid_payload=invalid_payload,
                        ),
                    },
                ]

    async def _run_staged_step_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        provider: object,
        staged: StagedStructuredToolRunner,
        stage_name: str,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        parser: Callable[[object], _StagedStepT],
        observer: ModelTurnProtocolObserver | None,
    ) -> _StagedStepT:
        run_structured_async = cast(Any, provider).run_structured_async
        attempt_messages = list(messages)
        repair_attempts = 0
        while True:
            self._emit(
                "stage_start",
                "staged",
                stage_name=stage_name,
                payload={"message_count": len(attempt_messages)},
                observer=observer,
                redaction_config=request.redaction_config,
            )
            payload: object | None = None
            try:
                payload = await run_structured_async(
                    messages=attempt_messages,
                    response_model=response_model,
                    request_params={"temperature": request.temperature},
                )
                self._emit_provider_call_complete(
                    "staged",
                    stage_name=stage_name,
                    observer=observer,
                    redaction_config=request.redaction_config,
                )
                return parser(payload)
            except Exception as exc:
                if (
                    repair_attempts >= self._repair_attempts
                    or not is_repairable_stage_error(exc)
                ):
                    raise
                repair_attempts += 1
                self._emit_repair(request, "staged", stage_name, exc, observer)
                invalid_payload = payload
                if invalid_payload is None:
                    invalid_payload = getattr(exc, "invalid_payload", None)
                attempt_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": staged.repair_stage_message(
                            stage_name=stage_name,
                            response_model=response_model,
                            error=exc,
                            invalid_payload=invalid_payload,
                        ),
                    },
                ]

    def _run_prompt_tool(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        if self._uses_prompt_tool_category_strategy(
            request.prepared_interaction.tool_specs
        ):
            return self._run_prompt_tool_category(
                request,
                messages=messages,
                observer=observer,
            )
        if self._uses_prompt_tool_single_action_strategy():
            return self._run_prompt_tool_single_action(
                request,
                messages=messages,
                tool_specs=request.prepared_interaction.tool_specs,
                selected_category=None,
                observer=observer,
            )
        return self._run_prompt_tool_split(
            request,
            messages=messages,
            observer=observer,
        )

    async def _run_prompt_tool_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        if self._uses_prompt_tool_category_strategy(
            request.prepared_interaction.tool_specs
        ):
            return await self._run_prompt_tool_category_async(
                request,
                messages=messages,
                observer=observer,
            )
        if self._uses_prompt_tool_single_action_strategy():
            return await self._run_prompt_tool_single_action_async(
                request,
                messages=messages,
                tool_specs=request.prepared_interaction.tool_specs,
                selected_category=None,
                observer=observer,
            )
        return await self._run_prompt_tool_split_async(
            request,
            messages=messages,
            observer=observer,
        )

    def _run_prompt_tool_split(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        prompt = PromptToolAdapter()
        decision = self._run_prompt_tool_step(
            request,
            stage_name="decision",
            messages=prompt.decision_stage_messages(
                base_messages=messages,
                tool_specs=request.prepared_interaction.tool_specs,
                decision_context=request.decision_context,
            ),
            parser=lambda text: prompt.parse_decision(
                text,
                tool_specs=request.prepared_interaction.tool_specs,
            ),
            repair_context={"tool_specs": request.prepared_interaction.tool_specs},
            observer=observer,
        )
        if decision.mode == "finalize":
            return self._run_prompt_tool_final_response(
                request,
                prompt=prompt,
                messages=messages,
                observer=observer,
            )
        tool_name = decision.tool_name
        if not isinstance(tool_name, str) or tool_name.strip() == "":
            raise ValueError("Prompt-tool decision did not select a valid tool name.")
        tool_spec = tool_spec_by_name(
            request.prepared_interaction.tool_specs, tool_name
        )
        tool_input_model = request.prepared_interaction.input_models.get(tool_name)
        if tool_input_model is None:
            raise ValueError(
                f"Selected tool '{tool_name}' was not prepared for this interaction."
            )
        self._emit_preparing(tool_name, observer)
        return self._run_prompt_tool_step(
            request,
            stage_name=f"tool:{tool_name}",
            messages=prompt.tool_invocation_stage_messages(
                base_messages=messages,
                tool_spec=tool_spec,
                input_model=tool_input_model,
            ),
            parser=lambda text: prompt.parse_tool_invocation(
                text,
                tool_name=tool_name,
                input_model=tool_input_model,
            ),
            repair_context={
                "selected_tool": tool_spec,
                "input_model": tool_input_model,
            },
            observer=observer,
        )

    async def _run_prompt_tool_split_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        prompt = PromptToolAdapter()
        decision = await self._run_prompt_tool_step_async(
            request,
            stage_name="decision",
            messages=prompt.decision_stage_messages(
                base_messages=messages,
                tool_specs=request.prepared_interaction.tool_specs,
                decision_context=request.decision_context,
            ),
            parser=lambda text: prompt.parse_decision(
                text,
                tool_specs=request.prepared_interaction.tool_specs,
            ),
            repair_context={"tool_specs": request.prepared_interaction.tool_specs},
            observer=observer,
        )
        if decision.mode == "finalize":
            return await self._run_prompt_tool_final_response_async(
                request,
                prompt=prompt,
                messages=messages,
                observer=observer,
            )
        tool_name = decision.tool_name
        if not isinstance(tool_name, str) or tool_name.strip() == "":
            raise ValueError("Prompt-tool decision did not select a valid tool name.")
        tool_spec = tool_spec_by_name(
            request.prepared_interaction.tool_specs, tool_name
        )
        tool_input_model = request.prepared_interaction.input_models.get(tool_name)
        if tool_input_model is None:
            raise ValueError(
                f"Selected tool '{tool_name}' was not prepared for this interaction."
            )
        self._emit_preparing(tool_name, observer)
        return await self._run_prompt_tool_step_async(
            request,
            stage_name=f"tool:{tool_name}",
            messages=prompt.tool_invocation_stage_messages(
                base_messages=messages,
                tool_spec=tool_spec,
                input_model=tool_input_model,
            ),
            parser=lambda text: prompt.parse_tool_invocation(
                text,
                tool_name=tool_name,
                input_model=tool_input_model,
            ),
            repair_context={
                "selected_tool": tool_spec,
                "input_model": tool_input_model,
            },
            observer=observer,
        )

    def _run_prompt_tool_category(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        prompt = PromptToolAdapter()
        categories = prompt.derive_tool_categories(
            request.prepared_interaction.tool_specs
        )
        category_decision = self._run_prompt_tool_step(
            request,
            stage_name="category",
            messages=prompt.category_decision_stage_messages(
                base_messages=messages,
                categories=categories,
                final_response_model=request.final_response_model,
                decision_context=request.decision_context,
            ),
            parser=lambda text: prompt.parse_category_decision(
                text,
                categories=categories,
            ),
            repair_context={
                "categories": categories,
                "final_response_model": request.final_response_model,
            },
            observer=observer,
        )
        if category_decision.mode == "finalize":
            return self._run_prompt_tool_final_response(
                request,
                prompt=prompt,
                messages=messages,
                observer=observer,
            )
        category_name = category_decision.category
        if not isinstance(category_name, str) or category_name.strip() == "":
            raise ValueError("Prompt-tool category step did not select a category.")
        self._emit(
            "stage_start",
            "prompt_tools",
            stage_name="category_selected",
            payload={"category": category_name},
            observer=observer,
            redaction_config=request.redaction_config,
        )
        return self._run_prompt_tool_single_action(
            request,
            messages=messages,
            tool_specs=prompt.category_tool_specs(categories, category_name),
            selected_category=category_name,
            observer=observer,
        )

    async def _run_prompt_tool_category_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        prompt = PromptToolAdapter()
        categories = prompt.derive_tool_categories(
            request.prepared_interaction.tool_specs
        )
        category_decision = await self._run_prompt_tool_step_async(
            request,
            stage_name="category",
            messages=prompt.category_decision_stage_messages(
                base_messages=messages,
                categories=categories,
                final_response_model=request.final_response_model,
                decision_context=request.decision_context,
            ),
            parser=lambda text: prompt.parse_category_decision(
                text,
                categories=categories,
            ),
            repair_context={
                "categories": categories,
                "final_response_model": request.final_response_model,
            },
            observer=observer,
        )
        if category_decision.mode == "finalize":
            return await self._run_prompt_tool_final_response_async(
                request,
                prompt=prompt,
                messages=messages,
                observer=observer,
            )
        category_name = category_decision.category
        if not isinstance(category_name, str) or category_name.strip() == "":
            raise ValueError("Prompt-tool category step did not select a category.")
        self._emit(
            "stage_start",
            "prompt_tools",
            stage_name="category_selected",
            payload={"category": category_name},
            observer=observer,
            redaction_config=request.redaction_config,
        )
        return await self._run_prompt_tool_single_action_async(
            request,
            messages=messages,
            tool_specs=prompt.category_tool_specs(categories, category_name),
            selected_category=category_name,
            observer=observer,
        )

    def _run_prompt_tool_single_action(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        tool_specs: list[ToolSpec],
        selected_category: str | None,
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        prompt = PromptToolAdapter()
        parsed = self._run_prompt_tool_step(
            request,
            stage_name="action",
            messages=prompt.single_action_stage_messages(
                base_messages=messages,
                tool_specs=tool_specs,
                input_models=request.prepared_interaction.input_models,
                final_response_model=request.final_response_model,
                decision_context=request.decision_context,
                selected_category=selected_category,
            ),
            parser=lambda text: prompt.parse_single_action(
                text,
                tool_specs=tool_specs,
                input_models=request.prepared_interaction.input_models,
                final_response_model=request.final_response_model,
            ),
            repair_context={
                "tool_specs": tool_specs,
                "input_models": request.prepared_interaction.input_models,
                "final_response_model": request.final_response_model,
            },
            observer=observer,
        )
        if parsed.invocations:
            self._emit_preparing(parsed.invocations[0].tool_name, observer)
        return parsed

    async def _run_prompt_tool_single_action_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        messages: list[dict[str, Any]],
        tool_specs: list[ToolSpec],
        selected_category: str | None,
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        prompt = PromptToolAdapter()
        parsed = await self._run_prompt_tool_step_async(
            request,
            stage_name="action",
            messages=prompt.single_action_stage_messages(
                base_messages=messages,
                tool_specs=tool_specs,
                input_models=request.prepared_interaction.input_models,
                final_response_model=request.final_response_model,
                decision_context=request.decision_context,
                selected_category=selected_category,
            ),
            parser=lambda text: prompt.parse_single_action(
                text,
                tool_specs=tool_specs,
                input_models=request.prepared_interaction.input_models,
                final_response_model=request.final_response_model,
            ),
            repair_context={
                "tool_specs": tool_specs,
                "input_models": request.prepared_interaction.input_models,
                "final_response_model": request.final_response_model,
            },
            observer=observer,
        )
        if parsed.invocations:
            self._emit_preparing(parsed.invocations[0].tool_name, observer)
        return parsed

    def _run_prompt_tool_final_response(
        self,
        request: ModelTurnProtocolRequest,
        *,
        prompt: PromptToolAdapter,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        return self._run_prompt_tool_step(
            request,
            stage_name="final_response",
            messages=prompt.final_response_stage_messages(
                base_messages=messages,
                final_response_model=request.final_response_model,
            ),
            parser=lambda text: prompt.parse_final_response(
                text,
                final_response_model=request.final_response_model,
            ),
            repair_context={"final_response_model": request.final_response_model},
            observer=observer,
        )

    async def _run_prompt_tool_final_response_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        prompt: PromptToolAdapter,
        messages: list[dict[str, Any]],
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        return await self._run_prompt_tool_step_async(
            request,
            stage_name="final_response",
            messages=prompt.final_response_stage_messages(
                base_messages=messages,
                final_response_model=request.final_response_model,
            ),
            parser=lambda text: prompt.parse_final_response(
                text,
                final_response_model=request.final_response_model,
            ),
            repair_context={"final_response_model": request.final_response_model},
            observer=observer,
        )

    def _run_prompt_tool_step(
        self,
        request: ModelTurnProtocolRequest,
        *,
        stage_name: str,
        messages: list[dict[str, Any]],
        parser: Callable[[str], _StagedStepT],
        repair_context: dict[str, object],
        observer: ModelTurnProtocolObserver | None,
    ) -> _StagedStepT:
        run_text = getattr(request.provider, "run_text", None)
        if not callable(run_text):
            raise RuntimeError(
                "Prompt-tool protocol requires a provider that supports run_text()."
            )
        prompt = PromptToolAdapter()
        attempt_messages = list(messages)
        repair_attempts = 0
        while True:
            self._emit(
                "stage_start",
                "prompt_tools",
                stage_name=stage_name,
                payload={"message_count": len(attempt_messages)},
                observer=observer,
                redaction_config=request.redaction_config,
            )
            text: str | None = None
            try:
                text = run_text(
                    messages=attempt_messages,
                    request_params={"temperature": request.temperature},
                )
                self._emit_provider_call_complete(
                    "prompt_tools",
                    stage_name=stage_name,
                    observer=observer,
                    redaction_config=request.redaction_config,
                )
                return parser(text)
            except Exception as exc:
                if (
                    repair_attempts >= self._repair_attempts
                    or not is_repairable_stage_error(exc)
                ):
                    raise
                repair_attempts += 1
                self._emit_repair(request, "prompt_tools", stage_name, exc, observer)
                invalid_payload: object | None = text
                if invalid_payload is None:
                    invalid_payload = getattr(exc, "invalid_payload", None)
                attempt_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": self._prompt_tool_repair_message(
                            prompt,
                            stage_name=stage_name,
                            error=exc,
                            invalid_payload=invalid_payload,
                            repair_context=repair_context,
                        ),
                    },
                ]

    async def _run_prompt_tool_step_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        stage_name: str,
        messages: list[dict[str, Any]],
        parser: Callable[[str], _StagedStepT],
        repair_context: dict[str, object],
        observer: ModelTurnProtocolObserver | None,
    ) -> _StagedStepT:
        run_text_async = getattr(request.provider, "run_text_async", None)
        if not callable(run_text_async):
            raise RuntimeError(
                "Prompt-tool protocol requires a provider that supports run_text_async()."
            )
        prompt = PromptToolAdapter()
        attempt_messages = list(messages)
        repair_attempts = 0
        while True:
            self._emit(
                "stage_start",
                "prompt_tools",
                stage_name=stage_name,
                payload={"message_count": len(attempt_messages)},
                observer=observer,
                redaction_config=request.redaction_config,
            )
            text: str | None = None
            try:
                text = await run_text_async(
                    messages=attempt_messages,
                    request_params={"temperature": request.temperature},
                )
                self._emit_provider_call_complete(
                    "prompt_tools",
                    stage_name=stage_name,
                    observer=observer,
                    redaction_config=request.redaction_config,
                )
                return parser(text)
            except Exception as exc:
                if (
                    repair_attempts >= self._repair_attempts
                    or not is_repairable_stage_error(exc)
                ):
                    raise
                repair_attempts += 1
                self._emit_repair(request, "prompt_tools", stage_name, exc, observer)
                invalid_payload: object | None = text
                if invalid_payload is None:
                    invalid_payload = getattr(exc, "invalid_payload", None)
                attempt_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": self._prompt_tool_repair_message(
                            prompt,
                            stage_name=stage_name,
                            error=exc,
                            invalid_payload=invalid_payload,
                            repair_context=repair_context,
                        ),
                    },
                ]

    def _apply_prompt_protection(
        self,
        request: ModelTurnProtocolRequest,
        *,
        observer: ModelTurnProtocolObserver | None,
    ) -> list[dict[str, Any]] | ParsedModelResponse:
        if request.protection is None:
            return request.messages
        context = request.protection
        decision = context.controller.assess_prompt(
            messages=request.messages,
            provenance=context.provenance,
        )
        self._emit_protection_action("prompt", decision.action.value, observer)
        if decision.action is ProtectionAction.CONSTRAIN and decision.guard_text:
            return [
                {"role": "system", "content": decision.guard_text},
                *request.messages,
            ]
        if decision.action not in {ProtectionAction.CHALLENGE, ProtectionAction.BLOCK}:
            return request.messages
        safe_message = decision.challenge_message or (
            "The request was withheld for this environment."
        )
        if context.metadata_sink is not None:
            context.metadata_sink["protection_review"] = {
                "purge_requested": True,
                "safe_message": safe_message,
            }
            if (
                decision.action is ProtectionAction.CHALLENGE
                and context.original_user_message is not None
            ):
                context.metadata_sink["pending_protection_prompt"] = (
                    context.controller.build_pending_prompt(
                        original_user_message=context.original_user_message,
                        serialized_messages=request.messages,
                        decision=decision,
                        session_id=context.session_id,
                    )
                )
        return ParsedModelResponse(
            final_response=_safe_final_response_payload(
                safe_message,
                request.final_response_model,
            )
        )

    async def _apply_prompt_protection_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        observer: ModelTurnProtocolObserver | None,
    ) -> list[dict[str, Any]] | ParsedModelResponse:
        if request.protection is None:
            return request.messages
        context = request.protection
        decision = await context.controller.assess_prompt_async(
            messages=request.messages,
            provenance=context.provenance,
        )
        self._emit_protection_action("prompt", decision.action.value, observer)
        if decision.action is ProtectionAction.CONSTRAIN and decision.guard_text:
            return [
                {"role": "system", "content": decision.guard_text},
                *request.messages,
            ]
        if decision.action not in {ProtectionAction.CHALLENGE, ProtectionAction.BLOCK}:
            return request.messages
        safe_message = decision.challenge_message or (
            "The request was withheld for this environment."
        )
        if context.metadata_sink is not None:
            context.metadata_sink["protection_review"] = {
                "purge_requested": True,
                "safe_message": safe_message,
            }
            if (
                decision.action is ProtectionAction.CHALLENGE
                and context.original_user_message is not None
            ):
                context.metadata_sink["pending_protection_prompt"] = (
                    context.controller.build_pending_prompt(
                        original_user_message=context.original_user_message,
                        serialized_messages=request.messages,
                        decision=decision,
                        session_id=context.session_id,
                    )
                )
        return ParsedModelResponse(
            final_response=_safe_final_response_payload(
                safe_message,
                request.final_response_model,
            )
        )

    def _apply_response_protection(
        self,
        request: ModelTurnProtocolRequest,
        *,
        parsed: ParsedModelResponse,
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        if request.protection is None or parsed.final_response is None:
            return parsed
        context = request.protection
        decision = context.controller.review_response(
            response_payload=parsed.final_response,
            provenance=context.provenance,
        )
        self._emit_protection_action("response", decision.action.value, observer)
        if (
            decision.action is ProtectionAction.SANITIZE
            and decision.sanitized_payload is not None
        ):
            if context.metadata_sink is not None:
                context.metadata_sink["protection_review"] = {
                    "purge_requested": decision.should_purge,
                    "safe_message": decision.sanitized_payload,
                }
            return parsed.model_copy(
                update={"final_response": decision.sanitized_payload}
            )
        if decision.action is not ProtectionAction.BLOCK:
            return parsed
        safe_message = decision.safe_message or (
            "The response was withheld for this environment."
        )
        if context.metadata_sink is not None:
            context.metadata_sink["protection_review"] = {
                "purge_requested": decision.should_purge,
                "safe_message": safe_message,
            }
        return parsed.model_copy(
            update={
                "final_response": _safe_final_response_payload(
                    safe_message,
                    request.final_response_model,
                )
            }
        )

    async def _apply_response_protection_async(
        self,
        request: ModelTurnProtocolRequest,
        *,
        parsed: ParsedModelResponse,
        observer: ModelTurnProtocolObserver | None,
    ) -> ParsedModelResponse:
        if request.protection is None or parsed.final_response is None:
            return parsed
        context = request.protection
        decision = await context.controller.review_response_async(
            response_payload=parsed.final_response,
            provenance=context.provenance,
        )
        self._emit_protection_action("response", decision.action.value, observer)
        if (
            decision.action is ProtectionAction.SANITIZE
            and decision.sanitized_payload is not None
        ):
            if context.metadata_sink is not None:
                context.metadata_sink["protection_review"] = {
                    "purge_requested": decision.should_purge,
                    "safe_message": decision.sanitized_payload,
                }
            return parsed.model_copy(
                update={"final_response": decision.sanitized_payload}
            )
        if decision.action is not ProtectionAction.BLOCK:
            return parsed
        safe_message = decision.safe_message or (
            "The response was withheld for this environment."
        )
        if context.metadata_sink is not None:
            context.metadata_sink["protection_review"] = {
                "purge_requested": decision.should_purge,
                "safe_message": safe_message,
            }
        return parsed.model_copy(
            update={
                "final_response": _safe_final_response_payload(
                    safe_message,
                    request.final_response_model,
                )
            }
        )

    def _prompt_tool_repair_message(
        self,
        prompt: PromptToolAdapter,
        *,
        stage_name: str,
        error: Exception,
        invalid_payload: object | None,
        repair_context: dict[str, object],
    ) -> str:
        tool_specs_value = repair_context.get("tool_specs")
        input_models_value = repair_context.get("input_models")
        categories_value = repair_context.get("categories")
        selected_tool_value = repair_context.get("selected_tool")
        input_model_value = repair_context.get("input_model")
        return prompt.repair_stage_message(
            stage_name=stage_name,
            error=error,
            invalid_payload=invalid_payload,
            tool_specs=(
                cast(list[ToolSpec], tool_specs_value)
                if isinstance(tool_specs_value, list)
                else None
            ),
            input_models=(
                cast(dict[str, type[BaseModel]], input_models_value)
                if isinstance(input_models_value, dict)
                else None
            ),
            categories=(
                cast(list[PromptToolCategory], categories_value)
                if isinstance(categories_value, list)
                else None
            ),
            selected_tool=(
                selected_tool_value
                if isinstance(selected_tool_value, ToolSpec)
                else None
            ),
            input_model=(
                cast(type[BaseModel], input_model_value)
                if isinstance(input_model_value, type)
                else None
            ),
            final_response_model=repair_context.get("final_response_model"),
        )

    def _emit_repair(
        self,
        request: ModelTurnProtocolRequest,
        protocol: ModelTurnProtocolKind,
        stage_name: str,
        exc: Exception,
        observer: ModelTurnProtocolObserver | None,
    ) -> None:
        self._emit(
            "stage_repair",
            protocol,
            stage_name=stage_name,
            payload={
                "status": f"repairing {stage_name}",
                "guidance": repair_stage_guidance(stage_name),
                "error_type": type(exc).__name__,
                "error_summary": _redacted_error_summary(exc, request.redaction_config),
            },
            observer=observer,
            redaction_config=request.redaction_config,
        )

    def _emit_provider_call_complete(
        self,
        protocol: ModelTurnProtocolKind,
        *,
        stage_name: str,
        observer: ModelTurnProtocolObserver | None,
        redaction_config: RedactionConfig | None,
    ) -> None:
        self._emit(
            "provider_call_complete",
            protocol,
            stage_name=stage_name,
            payload={},
            observer=observer,
            redaction_config=redaction_config,
        )

    def _emit_preparing(
        self,
        tool_name: str,
        observer: ModelTurnProtocolObserver | None,
    ) -> None:
        self._emit(
            "stage_start",
            "prompt_tools",
            stage_name="prepare_tool",
            payload={"status": f"preparing {tool_name}", "tool_name": tool_name},
            observer=observer,
            redaction_config=None,
        )

    def _emit_protection_action(
        self,
        phase: str,
        action: str,
        observer: ModelTurnProtocolObserver | None,
    ) -> None:
        self._emit(
            "protection_action",
            "protection",
            stage_name=phase,
            payload={"action": action},
            observer=observer,
            redaction_config=None,
        )

    def _emit_parsed_response(
        self,
        parsed: ParsedModelResponse,
        *,
        observer: ModelTurnProtocolObserver | None,
    ) -> None:
        self._emit(
            "parsed_response",
            "native",
            payload=_parsed_response_summary(parsed),
            observer=observer,
            redaction_config=None,
        )

    def _emit(
        self,
        kind: ModelTurnProtocolEventKind,
        protocol: ModelTurnProtocolKind,
        *,
        stage_name: str | None = None,
        payload: dict[str, object],
        observer: ModelTurnProtocolObserver | None,
        redaction_config: RedactionConfig | None,
    ) -> None:
        if observer is None:
            return
        observer(
            ModelTurnProtocolEvent(
                kind=kind,
                protocol=protocol,
                stage_name=stage_name,
                payload=_redact_payload(payload, redaction_config),
            )
        )

    @staticmethod
    def _uses_staged_schema_protocol(provider: object) -> bool:
        preference = getattr(provider, "uses_staged_schema_protocol", None)
        if not callable(preference):
            return False
        return bool(preference())

    @staticmethod
    def _uses_prompt_tool_protocol(provider: object) -> bool:
        preference = getattr(provider, "uses_prompt_tool_protocol", None)
        if not callable(preference):
            return False
        return bool(preference())

    def _uses_json_single_action_strategy(
        self,
        request: ModelTurnProtocolRequest,
    ) -> bool:
        if request.use_json_single_action_strategy is not None:
            return request.use_json_single_action_strategy
        strategy_value = getattr(request.provider, "json_agent_strategy", None)
        if callable(strategy_value):
            strategy = str(strategy_value())
        elif isinstance(strategy_value, str):
            strategy = strategy_value
        else:
            strategy = os.environ.get(_JSON_AGENT_STRATEGY_ENV, "single_action")
        return strategy.strip().lower() not in {"staged", "split", "decision"}

    @staticmethod
    def _can_fallback_to_prompt_tools(provider: object, exc: Exception) -> bool:
        if _is_context_limit_error(exc):
            return False
        run_text = getattr(provider, "run_text", None)
        if not callable(run_text):
            return False
        can_fallback = getattr(provider, "can_fallback_to_prompt_tools", None)
        if not callable(can_fallback):
            return False
        return bool(can_fallback(exc))

    @staticmethod
    def _structured_provider(provider: object) -> object:
        run_structured = getattr(provider, "run_structured", None)
        if not callable(run_structured):
            raise RuntimeError(
                "Staged schema protocol requires a provider that supports run_structured()."
            )
        return provider

    @classmethod
    def _uses_prompt_tool_category_strategy(cls, tool_specs: list[ToolSpec]) -> bool:
        if os.environ.get(_PROMPT_TOOL_STRATEGY_ENV, "").strip().lower() != "category":
            return False
        if len(tool_specs) < cls._prompt_tool_category_threshold():
            return False
        return len(PromptToolAdapter.derive_tool_categories(tool_specs)) > 1

    @staticmethod
    def _uses_prompt_tool_single_action_strategy() -> bool:
        strategy = os.environ.get(_PROMPT_TOOL_STRATEGY_ENV, "").strip().lower()
        if not strategy:
            return True
        if strategy in _PROMPT_TOOL_SPLIT_STRATEGIES or strategy == "category":
            return False
        return strategy in _PROMPT_TOOL_SINGLE_ACTION_STRATEGIES

    @staticmethod
    def _prompt_tool_category_threshold() -> int:
        raw_value = os.environ.get(_PROMPT_TOOL_CATEGORY_THRESHOLD_ENV)
        if raw_value is None:
            return _DEFAULT_PROMPT_TOOL_CATEGORY_THRESHOLD
        try:
            return max(int(raw_value), 1)
        except ValueError:
            return _DEFAULT_PROMPT_TOOL_CATEGORY_THRESHOLD

    @staticmethod
    def _native_tool_round_messages(
        *,
        messages: list[dict[str, Any]],
        decision_context: str | None,
    ) -> list[dict[str, Any]]:
        if not decision_context:
            return messages
        return [
            *messages,
            {
                "role": "system",
                "content": (
                    "Current native-tool step.\n"
                    f"{decision_context}\n\n"
                    "Use another native tool call only if it is necessary and within budget. "
                    "If available evidence is sufficient, return the final answer now."
                ),
            },
        ]

    @staticmethod
    def _prompt_tool_base_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            *messages,
            {
                "role": "system",
                "content": (
                    "Switching to prompt-tool protocol. Ignore earlier native tool-call "
                    "or structured JSON response instructions. Follow only the fenced "
                    "prompt-tool instructions in the next system message."
                ),
            },
        ]


def _redacted_error_summary(
    error: Exception,
    redaction_config: RedactionConfig | None,
) -> str:
    payload = _redact_payload(
        {"error": str(error).strip() or type(error).__name__},
        redaction_config,
    )
    error_value = payload.get("error")
    return str(error_value)


def _redact_payload(
    payload: dict[str, object],
    redaction_config: RedactionConfig | None,
) -> dict[str, object]:
    if redaction_config is None:
        return dict(payload)
    redactor = Redactor(redaction_config, tool_name="model_turn_protocol")
    redacted = redactor.redact_structured(payload, target=RedactionTarget.ERROR_DETAILS)
    return dict(redacted) if isinstance(redacted, dict) else dict(payload)


def _parsed_response_summary(parsed: ParsedModelResponse) -> dict[str, object]:
    if parsed.final_response is not None:
        return {
            "mode": "final_response",
            "final_response_type": type(parsed.final_response).__name__,
            "invocation_count": 0,
        }
    return {
        "mode": "tool_invocations",
        "invocation_count": len(parsed.invocations),
        "tool_names": [invocation.tool_name for invocation in parsed.invocations],
    }


def _final_response_payload(payload: object, final_response_model: object) -> object:
    if isinstance(payload, BaseModel):
        return payload.model_dump(mode="json")
    if final_response_model is str:
        if isinstance(payload, str):
            return payload
        return str(payload)
    if isinstance(final_response_model, type) and issubclass(
        final_response_model,
        BaseModel,
    ):
        return final_response_model.model_validate(payload).model_dump(mode="json")
    return payload


def _safe_final_response_payload(message: str, final_response_model: object) -> object:
    if final_response_model is str:
        return message
    if isinstance(final_response_model, type) and issubclass(
        final_response_model,
        BaseModel,
    ):
        try:
            return final_response_model(answer=message).model_dump(mode="json")
        except Exception:
            return message
    return message


def _is_context_limit_error(exc: Exception) -> bool:
    for candidate in _iter_exception_chain(exc):
        message = str(candidate).lower()
        if any(marker in message for marker in _CONTEXT_LIMIT_ERROR_MARKERS):
            return True
        status_code = getattr(candidate, "status_code", None)
        if status_code == 400 and any(
            marker in message for marker in ("token", "context", "prompt")
        ):
            return True
    return False


def _iter_exception_chain(error: Exception) -> list[BaseException]:
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


__all__ = [
    "ModelTurnProtocolEvent",
    "ModelTurnProtocolEventKind",
    "ModelTurnProtocolKind",
    "ModelTurnProtocolMode",
    "ModelTurnProtocolObserver",
    "ModelTurnProtocolProvider",
    "ModelTurnProtocolRequest",
    "ModelTurnProtocolRunner",
    "ModelTurnProtectionContext",
]
