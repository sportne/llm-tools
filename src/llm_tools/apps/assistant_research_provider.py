"""Research-session provider helpers for the assistant app."""

from __future__ import annotations

import json
from collections.abc import Sequence

from llm_tools.apps.assistant_config import AssistantConfig
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
)
from llm_tools.llm_providers import OpenAICompatibleProvider, ResponseModeStrategy
from llm_tools.tool_api import ToolContext, ToolRegistry
from llm_tools.workflow_api.executor import PreparedModelInteraction
from llm_tools.workflow_api.model_turn_protocol import (
    ModelTurnProtectionContext,
    ModelTurnProtocolRequest,
    ModelTurnProtocolRunner,
)
from llm_tools.workflow_api.protection import ProtectionController
from llm_tools.workflow_api.staged_structured import (
    format_invalid_payload,
    validation_error_summary,
)


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
        return ModelTurnProtocolRunner().run(
            ModelTurnProtocolRequest(
                provider=self._provider,
                messages=messages,
                prepared_interaction=prepared_interaction,
                adapter=adapter,
                final_response_model=str,
                temperature=self._temperature,
                protection=(
                    ModelTurnProtectionContext(
                        controller=self._protection_controller,
                        provenance=provenance,
                        metadata_sink=context.metadata,
                    )
                    if self._protection_controller is not None
                    else None
                ),
                use_json_single_action_strategy=False,
            )
        )

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
        return await ModelTurnProtocolRunner().run_async(
            ModelTurnProtocolRequest(
                provider=self._provider,
                messages=messages,
                prepared_interaction=prepared_interaction,
                adapter=adapter,
                final_response_model=str,
                temperature=self._temperature,
                protection=(
                    ModelTurnProtectionContext(
                        controller=self._protection_controller,
                        provenance=provenance,
                        metadata_sink=context.metadata,
                    )
                    if self._protection_controller is not None
                    else None
                ),
                use_json_single_action_strategy=False,
            )
        )

    @staticmethod
    def _validation_error_summary(error: Exception) -> str:
        return validation_error_summary(error)

    @staticmethod
    def _format_invalid_payload(invalid_payload: object | None) -> str:
        return format_invalid_payload(invalid_payload)


def build_live_harness_provider(
    *,
    config: AssistantConfig,
    provider_config: ChatLLMConfig,
    selected_model: str,
    api_key: str | None,
    response_mode_strategy: ResponseModeStrategy,
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
        provider_protocol=provider_config.provider_protocol,
        provider_connection=provider_config.provider_connection,
        api_key=api_key,
        selected_model=selected_model,
        response_mode_strategy=response_mode_strategy,
        timeout_seconds=provider_config.timeout_seconds,
    )
    protection_controller = build_protection_controller(
        config=config.protection,
        provider=provider,
        environment=build_protection_environment(
            app_name="assistant_app_research",
            model_name=selected_model,
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
