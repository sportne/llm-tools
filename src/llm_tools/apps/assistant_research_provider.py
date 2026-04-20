"""Research-session provider helpers for the assistant app."""

from __future__ import annotations

import json
from collections.abc import Sequence

from llm_tools.apps.assistant_config import StreamlitAssistantConfig
from llm_tools.apps.assistant_prompts import build_research_system_prompt
from llm_tools.apps.chat_config import ChatLLMConfig
from llm_tools.apps.chat_runtime import create_provider
from llm_tools.apps.protection_runtime import (
    build_protection_controller,
    build_protection_environment,
)
from llm_tools.harness_api import collect_state_provenance
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.llm_providers import OpenAICompatibleProvider, ProviderModeStrategy
from llm_tools.tool_api import ToolContext, ToolRegistry
from llm_tools.workflow_api.executor import PreparedModelInteraction
from llm_tools.workflow_api.protection import ProtectionController


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
        parsed = self._provider.run(
            adapter=adapter,
            messages=messages,
            response_model=prepared_interaction.response_model,
            request_params={"temperature": self._temperature},
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
        parsed = await self._provider.run_async(
            adapter=adapter,
            messages=messages,
            response_model=prepared_interaction.response_model,
            request_params={"temperature": self._temperature},
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
        ),
        protection_controller=protection_controller,
    )


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
