"""Research-session provider helpers for the assistant app."""

from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from typing import TypeVar, cast

from pydantic import BaseModel

from llm_tools.apps.assistant_config import StreamlitAssistantConfig
from llm_tools.apps.assistant_prompts import build_research_system_prompt
from llm_tools.apps.chat_config import ChatLLMConfig
from llm_tools.apps.chat_runtime import create_provider
from llm_tools.apps.protection_runtime import (
    build_protection_controller,
    build_protection_environment,
)
from llm_tools.harness_api import collect_state_provenance
from llm_tools.llm_adapters import (
    ActionEnvelopeAdapter,
    ParsedModelResponse,
    PromptToolAdapter,
)
from llm_tools.llm_providers import OpenAICompatibleProvider, ProviderModeStrategy
from llm_tools.tool_api import ToolContext, ToolRegistry, ToolSpec
from llm_tools.workflow_api.executor import PreparedModelInteraction
from llm_tools.workflow_api.protection import ProtectionController

_StagedStepT = TypeVar("_StagedStepT")


class AssistantHarnessTurnProvider:
    """Live harness provider wrapper backed by the OpenAI-compatible client."""

    def __init__(
        self,
        *,
        provider: OpenAICompatibleProvider,
        temperature: float,
        system_prompt: str,
        protection_controller: ProtectionController | None = None,
    ) -> None:
        self._provider = provider
        self._temperature = temperature
        self._system_prompt = system_prompt
        self._protection_controller = protection_controller
        self._prompt_tool_adapter = PromptToolAdapter()

    def prefers_simplified_json_schema_contract(self) -> bool:
        """Return whether research turns should use the simplified JSON contract."""
        return self._provider.prefers_simplified_json_schema_contract()

    def uses_staged_schema_protocol(self) -> bool:
        """Return whether research turns should use staged strict schemas."""
        preference = getattr(self._provider, "uses_staged_schema_protocol", None)
        if not callable(preference):
            return False
        return bool(preference())

    def uses_prompt_tool_protocol(self) -> bool:
        """Return whether research turns should use prompt-emitted tool calls."""
        preference = getattr(self._provider, "uses_prompt_tool_protocol", None)
        if not callable(preference):
            return False
        return bool(preference())

    def run(
        self,
        *,
        state: object,
        selected_task_ids: Sequence[str],
        context: ToolContext,
        adapter: ActionEnvelopeAdapter,
        prepared_interaction: PreparedModelInteraction,
    ) -> ParsedModelResponse:
        provenance = collect_state_provenance(state)
        messages = _build_research_messages(
            system_prompt=self._system_prompt,
            selected_task_ids=selected_task_ids,
            context=context,
        )
        if self._protection_controller is not None:
            prompt_decision = self._protection_controller.assess_prompt(
                messages=messages,
                provenance=provenance,
            )
            if (
                prompt_decision.action.value == "constrain"
                and prompt_decision.guard_text
            ):
                messages = [
                    {"role": "system", "content": prompt_decision.guard_text},
                    *messages,
                ]
            elif prompt_decision.action.value in {"challenge", "block"}:
                safe_message = (
                    prompt_decision.challenge_message
                    or "The request was withheld for this environment."
                )
                context.metadata["protection_review"] = {
                    "purge_requested": True,
                    "safe_message": safe_message,
                }
                return ParsedModelResponse(final_response=safe_message)
        if self.uses_prompt_tool_protocol():
            parsed = self._run_prompt_tool(
                messages=messages,
                prepared_interaction=prepared_interaction,
            )
        elif self.uses_staged_schema_protocol():
            try:
                parsed = self._run_staged(
                    messages=messages,
                    prepared_interaction=prepared_interaction,
                    adapter=adapter,
                )
            except Exception as exc:
                if not self._can_fallback_to_prompt_tools(exc):
                    raise
                parsed = self._run_prompt_tool(
                    messages=messages,
                    prepared_interaction=prepared_interaction,
                )
        else:
            try:
                parsed = self._provider.run(
                    adapter=adapter,
                    messages=messages,
                    response_model=prepared_interaction.response_model,
                    request_params={"temperature": self._temperature},
                )
            except Exception as exc:
                if not self._can_fallback_to_prompt_tools(exc):
                    raise
                parsed = self._run_prompt_tool(
                    messages=messages,
                    prepared_interaction=prepared_interaction,
                )
        if self._protection_controller is None or parsed.final_response is None:
            return parsed
        response_decision = self._protection_controller.review_response(
            response_payload=parsed.final_response,
            provenance=provenance,
        )
        if (
            response_decision.action.value == "sanitize"
            and response_decision.sanitized_payload is not None
        ):
            context.metadata["protection_review"] = {
                "purge_requested": response_decision.should_purge,
                "safe_message": response_decision.sanitized_payload,
            }
            return parsed.model_copy(
                update={"final_response": response_decision.sanitized_payload}
            )
        if response_decision.action.value == "block":
            safe_message = (
                response_decision.safe_message
                or "The response was withheld for this environment."
            )
            context.metadata["protection_review"] = {
                "purge_requested": response_decision.should_purge,
                "safe_message": safe_message,
            }
            return parsed.model_copy(update={"final_response": safe_message})
        return parsed

    async def run_async(
        self,
        *,
        state: object,
        selected_task_ids: Sequence[str],
        context: ToolContext,
        adapter: ActionEnvelopeAdapter,
        prepared_interaction: PreparedModelInteraction,
    ) -> ParsedModelResponse:
        provenance = collect_state_provenance(state)
        messages = _build_research_messages(
            system_prompt=self._system_prompt,
            selected_task_ids=selected_task_ids,
            context=context,
        )
        if self._protection_controller is not None:
            prompt_decision = await self._protection_controller.assess_prompt_async(
                messages=messages,
                provenance=provenance,
            )
            if (
                prompt_decision.action.value == "constrain"
                and prompt_decision.guard_text
            ):
                messages = [
                    {"role": "system", "content": prompt_decision.guard_text},
                    *messages,
                ]
            elif prompt_decision.action.value in {"challenge", "block"}:
                safe_message = (
                    prompt_decision.challenge_message
                    or "The request was withheld for this environment."
                )
                context.metadata["protection_review"] = {
                    "purge_requested": True,
                    "safe_message": safe_message,
                }
                return ParsedModelResponse(final_response=safe_message)
        if self.uses_prompt_tool_protocol():
            parsed = await self._run_prompt_tool_async(
                messages=messages,
                prepared_interaction=prepared_interaction,
            )
        elif self.uses_staged_schema_protocol():
            try:
                parsed = await self._run_staged_async(
                    messages=messages,
                    prepared_interaction=prepared_interaction,
                    adapter=adapter,
                )
            except Exception as exc:
                if not self._can_fallback_to_prompt_tools(exc):
                    raise
                parsed = await self._run_prompt_tool_async(
                    messages=messages,
                    prepared_interaction=prepared_interaction,
                )
        else:
            try:
                parsed = await self._provider.run_async(
                    adapter=adapter,
                    messages=messages,
                    response_model=prepared_interaction.response_model,
                    request_params={"temperature": self._temperature},
                )
            except Exception as exc:
                if not self._can_fallback_to_prompt_tools(exc):
                    raise
                parsed = await self._run_prompt_tool_async(
                    messages=messages,
                    prepared_interaction=prepared_interaction,
                )
        if self._protection_controller is None or parsed.final_response is None:
            return parsed
        response_decision = await self._protection_controller.review_response_async(
            response_payload=parsed.final_response,
            provenance=provenance,
        )
        if (
            response_decision.action.value == "sanitize"
            and response_decision.sanitized_payload is not None
        ):
            context.metadata["protection_review"] = {
                "purge_requested": response_decision.should_purge,
                "safe_message": response_decision.sanitized_payload,
            }
            return parsed.model_copy(
                update={"final_response": response_decision.sanitized_payload}
            )
        if response_decision.action.value == "block":
            safe_message = (
                response_decision.safe_message
                or "The response was withheld for this environment."
            )
            context.metadata["protection_review"] = {
                "purge_requested": response_decision.should_purge,
                "safe_message": safe_message,
            }
            return parsed.model_copy(update={"final_response": safe_message})
        return parsed

    def _can_fallback_to_prompt_tools(self, exc: Exception) -> bool:
        run_text = getattr(self._provider, "run_text", None)
        if not callable(run_text):
            return False
        can_fallback = getattr(self._provider, "can_fallback_to_prompt_tools", None)
        if not callable(can_fallback):
            return False
        return bool(can_fallback(exc))

    def _run_staged(
        self,
        *,
        messages: list[dict[str, str]],
        prepared_interaction: PreparedModelInteraction,
        adapter: ActionEnvelopeAdapter,
    ) -> ParsedModelResponse:
        decision_model = adapter.build_decision_step_model(
            prepared_interaction.tool_specs
        )
        decision = self._run_staged_step(
            stage_name="decision",
            messages=[
                *messages,
                {
                    "role": "system",
                    "content": (
                        "Current step: choose the next action.\n"
                        "Return mode='tool' with exactly one tool_name, or mode='finalize'.\n"
                        "Do not provide tool arguments or a final response in this step.\n"
                        f"Available tools:\n{json.dumps(adapter.export_compact_tool_catalog(prepared_interaction.tool_specs), indent=2, sort_keys=True)}"
                    ),
                },
            ],
            response_model=decision_model,
            parser=lambda payload: decision_model.model_validate(payload),
        )
        if getattr(decision, "mode", None) == "finalize":
            final_model = adapter.build_final_response_step_model(
                final_response_model=str
            )
            return self._run_staged_step(
                stage_name="final_response",
                messages=[
                    *messages,
                    {
                        "role": "system",
                        "content": (
                            "Current step: finalize the research response.\n"
                            "Return only final_response and no tool invocation.\n"
                            f"Schema:\n{json.dumps(adapter.export_schema(final_model), indent=2, sort_keys=True)}"
                        ),
                    },
                ],
                response_model=final_model,
                parser=lambda payload: adapter.parse_final_response_step(
                    payload,
                    response_model=final_model,
                ),
            )
        tool_name = getattr(decision, "tool_name", None)
        if not isinstance(tool_name, str) or tool_name.strip() == "":
            raise ValueError("Decision stage did not select a valid tool name.")
        tool_spec = self._tool_spec(prepared_interaction.tool_specs, tool_name)
        tool_input_model = prepared_interaction.input_models.get(tool_name)
        if tool_input_model is None:
            raise ValueError(
                f"Selected tool '{tool_name}' was not prepared for this interaction."
            )
        tool_model = adapter.build_tool_invocation_step_model(
            tool_name=tool_name,
            input_model=tool_input_model,
        )
        return self._run_staged_step(
            stage_name=f"tool:{tool_name}",
            messages=[
                *messages,
                {
                    "role": "system",
                    "content": (
                        f"Current step: invoke the selected tool '{tool_spec.name}'.\n"
                        f"Tool description: {tool_spec.description}\n"
                        "Return exactly one invocation for this tool.\n"
                        "Do not choose another tool and do not finalize in this step.\n"
                        f"Schema:\n{json.dumps(adapter.export_schema(tool_model), indent=2, sort_keys=True)}"
                    ),
                },
            ],
            response_model=tool_model,
            parser=lambda payload: adapter.parse_tool_invocation_step(
                payload,
                response_model=tool_model,
            ),
        )

    async def _run_staged_async(
        self,
        *,
        messages: list[dict[str, str]],
        prepared_interaction: PreparedModelInteraction,
        adapter: ActionEnvelopeAdapter,
    ) -> ParsedModelResponse:
        decision_model = adapter.build_decision_step_model(
            prepared_interaction.tool_specs
        )
        decision = await self._run_staged_step_async(
            stage_name="decision",
            messages=[
                *messages,
                {
                    "role": "system",
                    "content": (
                        "Current step: choose the next action.\n"
                        "Return mode='tool' with exactly one tool_name, or mode='finalize'.\n"
                        "Do not provide tool arguments or a final response in this step.\n"
                        f"Available tools:\n{json.dumps(adapter.export_compact_tool_catalog(prepared_interaction.tool_specs), indent=2, sort_keys=True)}"
                    ),
                },
            ],
            response_model=decision_model,
            parser=lambda payload: decision_model.model_validate(payload),
        )
        if getattr(decision, "mode", None) == "finalize":
            final_model = adapter.build_final_response_step_model(
                final_response_model=str
            )
            return await self._run_staged_step_async(
                stage_name="final_response",
                messages=[
                    *messages,
                    {
                        "role": "system",
                        "content": (
                            "Current step: finalize the research response.\n"
                            "Return only final_response and no tool invocation.\n"
                            f"Schema:\n{json.dumps(adapter.export_schema(final_model), indent=2, sort_keys=True)}"
                        ),
                    },
                ],
                response_model=final_model,
                parser=lambda payload: adapter.parse_final_response_step(
                    payload,
                    response_model=final_model,
                ),
            )
        tool_name = getattr(decision, "tool_name", None)
        if not isinstance(tool_name, str) or tool_name.strip() == "":
            raise ValueError("Decision stage did not select a valid tool name.")
        tool_spec = self._tool_spec(prepared_interaction.tool_specs, tool_name)
        tool_input_model = prepared_interaction.input_models.get(tool_name)
        if tool_input_model is None:
            raise ValueError(
                f"Selected tool '{tool_name}' was not prepared for this interaction."
            )
        tool_model = adapter.build_tool_invocation_step_model(
            tool_name=tool_name,
            input_model=tool_input_model,
        )
        return await self._run_staged_step_async(
            stage_name=f"tool:{tool_name}",
            messages=[
                *messages,
                {
                    "role": "system",
                    "content": (
                        f"Current step: invoke the selected tool '{tool_spec.name}'.\n"
                        f"Tool description: {tool_spec.description}\n"
                        "Return exactly one invocation for this tool.\n"
                        "Do not choose another tool and do not finalize in this step.\n"
                        f"Schema:\n{json.dumps(adapter.export_schema(tool_model), indent=2, sort_keys=True)}"
                    ),
                },
            ],
            response_model=tool_model,
            parser=lambda payload: adapter.parse_tool_invocation_step(
                payload,
                response_model=tool_model,
            ),
        )

    def _run_prompt_tool(
        self,
        *,
        messages: list[dict[str, str]],
        prepared_interaction: PreparedModelInteraction,
    ) -> ParsedModelResponse:
        decision = self._run_prompt_tool_step(
            stage_name="decision",
            messages=self._prompt_tool_adapter.decision_stage_messages(
                base_messages=messages,
                tool_specs=prepared_interaction.tool_specs,
            ),
            parser=lambda text: self._prompt_tool_adapter.parse_decision(
                text,
                tool_specs=prepared_interaction.tool_specs,
            ),
            repair_context={"tool_specs": prepared_interaction.tool_specs},
        )
        if decision.mode == "finalize":
            return self._run_prompt_tool_step(
                stage_name="final_response",
                messages=self._prompt_tool_adapter.final_response_stage_messages(
                    base_messages=messages,
                    final_response_model=str,
                ),
                parser=lambda text: self._prompt_tool_adapter.parse_final_response(
                    text,
                    final_response_model=str,
                ),
                repair_context={"final_response_model": str},
            )

        tool_name = decision.tool_name
        if not isinstance(tool_name, str) or tool_name.strip() == "":
            raise ValueError("Prompt-tool decision did not select a valid tool name.")
        tool_spec = self._tool_spec(prepared_interaction.tool_specs, tool_name)
        tool_input_model = prepared_interaction.input_models.get(tool_name)
        if tool_input_model is None:
            raise ValueError(
                f"Selected tool '{tool_name}' was not prepared for this interaction."
            )
        return self._run_prompt_tool_step(
            stage_name=f"tool:{tool_name}",
            messages=self._prompt_tool_adapter.tool_invocation_stage_messages(
                base_messages=messages,
                tool_spec=tool_spec,
                input_model=tool_input_model,
            ),
            parser=lambda text: self._prompt_tool_adapter.parse_tool_invocation(
                text,
                tool_name=tool_name,
                input_model=tool_input_model,
            ),
            repair_context={
                "selected_tool": tool_spec,
                "input_model": tool_input_model,
            },
        )

    async def _run_prompt_tool_async(
        self,
        *,
        messages: list[dict[str, str]],
        prepared_interaction: PreparedModelInteraction,
    ) -> ParsedModelResponse:
        decision = await self._run_prompt_tool_step_async(
            stage_name="decision",
            messages=self._prompt_tool_adapter.decision_stage_messages(
                base_messages=messages,
                tool_specs=prepared_interaction.tool_specs,
            ),
            parser=lambda text: self._prompt_tool_adapter.parse_decision(
                text,
                tool_specs=prepared_interaction.tool_specs,
            ),
            repair_context={"tool_specs": prepared_interaction.tool_specs},
        )
        if decision.mode == "finalize":
            return await self._run_prompt_tool_step_async(
                stage_name="final_response",
                messages=self._prompt_tool_adapter.final_response_stage_messages(
                    base_messages=messages,
                    final_response_model=str,
                ),
                parser=lambda text: self._prompt_tool_adapter.parse_final_response(
                    text,
                    final_response_model=str,
                ),
                repair_context={"final_response_model": str},
            )

        tool_name = decision.tool_name
        if not isinstance(tool_name, str) or tool_name.strip() == "":
            raise ValueError("Prompt-tool decision did not select a valid tool name.")
        tool_spec = self._tool_spec(prepared_interaction.tool_specs, tool_name)
        tool_input_model = prepared_interaction.input_models.get(tool_name)
        if tool_input_model is None:
            raise ValueError(
                f"Selected tool '{tool_name}' was not prepared for this interaction."
            )
        return await self._run_prompt_tool_step_async(
            stage_name=f"tool:{tool_name}",
            messages=self._prompt_tool_adapter.tool_invocation_stage_messages(
                base_messages=messages,
                tool_spec=tool_spec,
                input_model=tool_input_model,
            ),
            parser=lambda text: self._prompt_tool_adapter.parse_tool_invocation(
                text,
                tool_name=tool_name,
                input_model=tool_input_model,
            ),
            repair_context={
                "selected_tool": tool_spec,
                "input_model": tool_input_model,
            },
        )

    def _run_prompt_tool_step(
        self,
        *,
        stage_name: str,
        messages: list[dict[str, str]],
        parser: Callable[[str], _StagedStepT],
        repair_context: dict[str, object],
    ) -> _StagedStepT:
        run_text = getattr(self._provider, "run_text", None)
        if not callable(run_text):
            raise RuntimeError(
                "Prompt-tool protocol requires a provider that supports run_text()."
            )
        attempt_messages = list(messages)
        repair_attempted = False
        while True:
            text: str | None = None
            try:
                text = run_text(
                    messages=attempt_messages,
                    request_params={"temperature": self._temperature},
                )
                return parser(text)
            except Exception as exc:
                if repair_attempted:
                    raise
                repair_attempted = True
                invalid_payload: object | None = text
                if invalid_payload is None:
                    invalid_payload = getattr(exc, "invalid_payload", None)
                attempt_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": self._prompt_tool_repair_message(
                            stage_name=stage_name,
                            error=exc,
                            invalid_payload=invalid_payload,
                            repair_context=repair_context,
                        ),
                    },
                ]

    async def _run_prompt_tool_step_async(
        self,
        *,
        stage_name: str,
        messages: list[dict[str, str]],
        parser: Callable[[str], _StagedStepT],
        repair_context: dict[str, object],
    ) -> _StagedStepT:
        run_text_async = getattr(self._provider, "run_text_async", None)
        if not callable(run_text_async):
            raise RuntimeError(
                "Prompt-tool protocol requires a provider that supports run_text_async()."
            )
        attempt_messages = list(messages)
        repair_attempted = False
        while True:
            text: str | None = None
            try:
                text = await run_text_async(
                    messages=attempt_messages,
                    request_params={"temperature": self._temperature},
                )
                return parser(text)
            except Exception as exc:
                if repair_attempted:
                    raise
                repair_attempted = True
                invalid_payload: object | None = text
                if invalid_payload is None:
                    invalid_payload = getattr(exc, "invalid_payload", None)
                attempt_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": self._prompt_tool_repair_message(
                            stage_name=stage_name,
                            error=exc,
                            invalid_payload=invalid_payload,
                            repair_context=repair_context,
                        ),
                    },
                ]

    def _prompt_tool_repair_message(
        self,
        *,
        stage_name: str,
        error: Exception,
        invalid_payload: object | None,
        repair_context: dict[str, object],
    ) -> str:
        tool_specs_value = repair_context.get("tool_specs")
        selected_tool_value = repair_context.get("selected_tool")
        input_model_value = repair_context.get("input_model")
        return self._prompt_tool_adapter.repair_stage_message(
            stage_name=stage_name,
            error=error,
            invalid_payload=invalid_payload,
            tool_specs=(
                cast(list[ToolSpec], tool_specs_value)
                if isinstance(tool_specs_value, list)
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

    def _run_staged_step(
        self,
        *,
        stage_name: str,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        parser: Callable[[object], _StagedStepT],
    ) -> _StagedStepT:
        attempt_messages = list(messages)
        repair_attempted = False
        while True:
            payload: object | None = None
            try:
                payload = self._provider.run_structured(
                    messages=attempt_messages,
                    response_model=response_model,
                    request_params={"temperature": self._temperature},
                )
                return parser(payload)
            except Exception as exc:
                if repair_attempted:
                    raise
                repair_attempted = True
                invalid_payload = payload
                if invalid_payload is None:
                    invalid_payload = getattr(exc, "invalid_payload", None)
                attempt_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": self._repair_stage_message(
                            stage_name=stage_name,
                            response_model=response_model,
                            error=exc,
                            invalid_payload=invalid_payload,
                        ),
                    },
                ]

    async def _run_staged_step_async(
        self,
        *,
        stage_name: str,
        messages: list[dict[str, str]],
        response_model: type[BaseModel],
        parser: Callable[[object], _StagedStepT],
    ) -> _StagedStepT:
        attempt_messages = list(messages)
        repair_attempted = False
        while True:
            payload: object | None = None
            try:
                payload = await self._provider.run_structured_async(
                    messages=attempt_messages,
                    response_model=response_model,
                    request_params={"temperature": self._temperature},
                )
                return parser(payload)
            except Exception as exc:
                if repair_attempted:
                    raise
                repair_attempted = True
                invalid_payload = payload
                if invalid_payload is None:
                    invalid_payload = getattr(exc, "invalid_payload", None)
                attempt_messages = [
                    *messages,
                    {
                        "role": "system",
                        "content": self._repair_stage_message(
                            stage_name=stage_name,
                            response_model=response_model,
                            error=exc,
                            invalid_payload=invalid_payload,
                        ),
                    },
                ]

    @staticmethod
    def _provider_schema(response_model: type[BaseModel]) -> dict[str, object]:
        return response_model.model_json_schema()

    def _repair_stage_message(
        self,
        *,
        stage_name: str,
        response_model: type[BaseModel],
        error: Exception,
        invalid_payload: object | None,
    ) -> str:
        return (
            f"The previous {stage_name} response was invalid.\n"
            "Correct the response for the same stage only.\n"
            f"{self._repair_stage_guidance(stage_name)}\n"
            f"Validation summary: {self._validation_error_summary(error)}\n"
            "Previous invalid payload:\n"
            f"{self._format_invalid_payload(invalid_payload)}\n"
            "Return a corrected payload matching this schema exactly:\n"
            f"{json.dumps(self._provider_schema(response_model), indent=2, sort_keys=True)}"
        )

    @staticmethod
    def _validation_error_summary(error: Exception) -> str:
        message = str(error).strip()
        return message or type(error).__name__

    @staticmethod
    def _repair_stage_guidance(stage_name: str) -> str:
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

    @staticmethod
    def _format_invalid_payload(invalid_payload: object | None) -> str:
        if invalid_payload is None:
            return "(unavailable)"
        if isinstance(invalid_payload, str):
            return invalid_payload
        try:
            return json.dumps(invalid_payload, indent=2, sort_keys=True, default=str)
        except TypeError:
            return str(invalid_payload)

    @staticmethod
    def _tool_spec(tool_specs: list[ToolSpec], tool_name: str) -> ToolSpec:
        for spec in tool_specs:
            if spec.name == tool_name:
                return spec
        raise ValueError(
            f"Unknown tool selected during staged interaction: {tool_name}"
        )


def build_live_harness_provider(
    *,
    config: StreamlitAssistantConfig,
    provider_config: ChatLLMConfig,
    model_name: str,
    api_key: str | None,
    mode_strategy: ProviderModeStrategy,
    tool_registry: ToolRegistry,
    enabled_tool_names: set[str],
    workspace_enabled: bool,
    workspace: str | None,
    allow_network: bool,
    allow_filesystem: bool,
    allow_subprocess: bool,
) -> AssistantHarnessTurnProvider:
    """Create the live provider wrapper used by research sessions."""
    provider = create_provider(
        provider_config,
        api_key=api_key,
        model_name=model_name,
        mode_strategy=mode_strategy,
    )
    protection_controller = build_protection_controller(
        config=config.protection,
        provider=provider,
        environment=build_protection_environment(
            app_name="streamlit_assistant_research",
            model_name=model_name,
            workspace=workspace,
            enabled_tools=enabled_tool_names,
            allow_network=allow_network,
            allow_filesystem=allow_filesystem,
            allow_subprocess=allow_subprocess,
        ),
    )
    return AssistantHarnessTurnProvider(
        provider=provider,
        temperature=config.llm.temperature,
        system_prompt=build_research_system_prompt(
            tool_registry=tool_registry,
            tool_limits=config.tool_limits,
            enabled_tool_names=enabled_tool_names,
            workspace_enabled=workspace_enabled,
            staged_schema_protocol=_uses_staged_schema_protocol(provider),
        ),
        protection_controller=protection_controller,
    )


def _uses_staged_schema_protocol(provider: object) -> bool:
    preference = getattr(provider, "uses_staged_schema_protocol", None)
    if not callable(preference):
        return False
    return bool(preference())


def _build_research_messages(
    *,
    system_prompt: str,
    selected_task_ids: Sequence[str],
    context: ToolContext,
) -> list[dict[str, str]]:
    projection = context.metadata.get("harness_turn_context", {})
    user_payload = {
        "selected_task_ids": list(selected_task_ids),
        "turn_context": projection,
    }
    return [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": json.dumps(user_payload, indent=2, sort_keys=True, default=str),
        },
    ]


__all__ = ["AssistantHarnessTurnProvider", "build_live_harness_provider"]
