"""Pure-Python controller logic for the Textual workbench."""

from __future__ import annotations

import json
import os
from typing import Any

from llm_tools.apps.textual_workbench.models import (
    DirectExecutionResult,
    ModelTurnExecutionResult,
    ProviderPreset,
    WorkbenchConfigState,
    WorkbenchMode,
)
from llm_tools.llm_adapters import (
    LLMAdapter,
    NativeToolCallingAdapter,
    PromptSchemaAdapter,
    StructuredOutputAdapter,
)
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import (
    SideEffectClass,
    ToolContext,
    ToolInvocationRequest,
    ToolPolicy,
    ToolRegistry,
    ToolRuntime,
)
from llm_tools.tools import (
    register_atlassian_tools,
    register_filesystem_tools,
    register_git_tools,
    register_text_tools,
)
from llm_tools.workflow_api import WorkflowExecutor


class WorkbenchController:
    """Build registries, providers, and execution flows for the workbench."""

    def __init__(
        self,
        *,
        provider_factory: type[OpenAICompatibleProvider] = OpenAICompatibleProvider,
    ) -> None:
        self._provider_factory = provider_factory

    def default_config(self) -> WorkbenchConfigState:
        """Return the default ephemeral configuration for a new launch."""
        return WorkbenchConfigState()

    def cycle_provider_preset(
        self,
        config: WorkbenchConfigState,
    ) -> WorkbenchConfigState:
        """Return a config with the next provider preset selected."""
        ordered = [
            ProviderPreset.OPENAI,
            ProviderPreset.OLLAMA,
            ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
        ]
        current_index = ordered.index(config.provider_preset)
        next_preset = ordered[(current_index + 1) % len(ordered)]
        return self.apply_provider_preset(config, next_preset)

    def apply_provider_preset(
        self,
        config: WorkbenchConfigState,
        preset: ProviderPreset,
    ) -> WorkbenchConfigState:
        """Return a config updated with preset-specific defaults."""
        if preset is ProviderPreset.OPENAI:
            return config.model_copy(
                update={
                    "provider_preset": preset,
                    "base_url": "",
                    "model": "gpt-4.1-mini",
                    "api_key": "",
                }
            )

        if preset is ProviderPreset.OLLAMA:
            return config.model_copy(
                update={
                    "provider_preset": preset,
                    "base_url": "http://localhost:11434/v1",
                    "model": "gemma4:26b",
                    "api_key": "ollama",
                }
            )

        return config.model_copy(
            update={
                "provider_preset": preset,
                "base_url": "",
                "model": "",
                "api_key": "",
            }
        )

    def cycle_mode(self, config: WorkbenchConfigState) -> WorkbenchConfigState:
        """Return a config with the next interaction mode selected."""
        ordered = [
            WorkbenchMode.NATIVE_TOOL_CALLING,
            WorkbenchMode.STRUCTURED_OUTPUT,
            WorkbenchMode.PROMPT_SCHEMA,
        ]
        current_index = ordered.index(config.mode)
        next_mode = ordered[(current_index + 1) % len(ordered)]
        return config.model_copy(update={"mode": next_mode})

    def build_registry(self, config: WorkbenchConfigState) -> ToolRegistry:
        """Build a fresh registry from the current UI configuration."""
        registry = ToolRegistry()
        if config.enable_filesystem_tools:
            register_filesystem_tools(registry)
        if config.enable_git_tools:
            register_git_tools(registry)
        if config.enable_text_tools:
            register_text_tools(registry)
        if config.enable_atlassian_tools:
            register_atlassian_tools(registry)
        return registry

    def list_tool_names(self, config: WorkbenchConfigState) -> list[str]:
        """Return registered tool names in deterministic order."""
        return [
            tool.spec.name
            for tool in self.build_registry(config).list_registered_tools()
        ]

    def get_tool_details(
        self,
        config: WorkbenchConfigState,
        tool_name: str,
    ) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
        """Return the selected tool spec and input schema, when available."""
        normalized_name = tool_name.strip()
        if normalized_name == "":
            return None, None

        registry = self.build_registry(config)
        try:
            tool = registry.get(normalized_name)
        except Exception:
            return None, None

        return (
            tool.spec.model_dump(mode="json"),
            tool.input_model.model_json_schema(),
        )

    def export_tools(self, config: WorkbenchConfigState) -> Any:
        """Export tools for the currently selected interaction mode."""
        registry = self.build_registry(config)
        executor = WorkflowExecutor(registry, policy=self.build_policy(config))
        return executor.export_tools(
            self._build_adapter(config.mode),
            context=self._make_context(config, invocation_id="workbench-export-tools"),
        )

    def execute_direct_tool(
        self,
        config: WorkbenchConfigState,
        *,
        tool_name: str,
        arguments_text: str,
    ) -> DirectExecutionResult:
        """Execute one tool directly through ToolRuntime."""
        normalized_tool_name = tool_name.strip()
        if normalized_tool_name == "":
            raise ValueError("A tool name is required for direct execution.")

        arguments = self._parse_json_object(
            arguments_text,
            error_prefix="Direct tool arguments",
        )
        registry = self.build_registry(config)
        runtime = ToolRuntime(registry, policy=self.build_policy(config))
        result = runtime.execute(
            ToolInvocationRequest(tool_name=normalized_tool_name, arguments=arguments),
            self._make_context(config, invocation_id="workbench-direct-tool"),
        )
        return DirectExecutionResult(tool_result=result)

    def run_model_turn(
        self,
        config: WorkbenchConfigState,
        *,
        prompt: str,
    ) -> ModelTurnExecutionResult:
        """Run one provider-backed model turn through the selected mode."""
        normalized_prompt = prompt.strip()
        if normalized_prompt == "":
            raise ValueError("A prompt is required to run a model turn.")

        registry = self.build_registry(config)
        policy = self.build_policy(config)
        executor = WorkflowExecutor(registry, policy=policy)
        adapter = self._build_adapter(config.mode)
        model_turn_context = self._make_context(
            config, invocation_id="workbench-model-turn"
        )
        exported_tools = executor.export_tools(
            adapter,
            context=model_turn_context,
        )
        provider = self._build_provider(config)
        messages = [{"role": "user", "content": normalized_prompt}]

        if config.mode is WorkbenchMode.NATIVE_TOOL_CALLING:
            parsed = provider.run_native_tool_calling(
                adapter=adapter,  # type: ignore[arg-type]
                messages=messages,
                tool_descriptions=exported_tools,
            )
        elif config.mode is WorkbenchMode.STRUCTURED_OUTPUT:
            parsed = provider.run_structured_output(
                adapter=adapter,  # type: ignore[arg-type]
                messages=messages,
                tool_descriptions=exported_tools,
            )
        else:
            parsed = provider.run_prompt_schema(
                adapter=adapter,  # type: ignore[arg-type]
                messages=messages,
                tool_descriptions=exported_tools,
            )

        workflow_result = None
        if config.execute_after_parse:
            workflow_result = executor.execute_parsed_response(
                parsed, model_turn_context
            )

        return ModelTurnExecutionResult(
            exported_tools=exported_tools,
            parsed_response=parsed,
            workflow_result=workflow_result,
        )

    def build_policy(self, config: WorkbenchConfigState) -> ToolPolicy:
        """Build a policy from the current UI toggles."""
        allowed_side_effects = {
            SideEffectClass.NONE,
            SideEffectClass.LOCAL_READ,
        }
        if config.allow_local_write:
            allowed_side_effects.add(SideEffectClass.LOCAL_WRITE)
        if config.allow_external_read:
            allowed_side_effects.add(SideEffectClass.EXTERNAL_READ)
        if config.allow_external_write:
            allowed_side_effects.add(SideEffectClass.EXTERNAL_WRITE)

        return ToolPolicy(
            allowed_side_effects=allowed_side_effects,
            allow_network=config.allow_network,
            allow_filesystem=config.allow_filesystem,
            allow_subprocess=config.allow_subprocess,
        )

    def _build_adapter(self, mode: WorkbenchMode) -> LLMAdapter:
        if mode is WorkbenchMode.NATIVE_TOOL_CALLING:
            return NativeToolCallingAdapter()
        if mode is WorkbenchMode.STRUCTURED_OUTPUT:
            return StructuredOutputAdapter()
        return PromptSchemaAdapter()

    def _build_provider(
        self,
        config: WorkbenchConfigState,
    ) -> OpenAICompatibleProvider:
        model = config.model.strip()
        if model == "":
            raise ValueError("A model name is required for provider-backed execution.")

        if config.provider_preset is ProviderPreset.OPENAI:
            return self._provider_factory.for_openai(
                model=model,
                api_key=config.api_key.strip() or None,
            )

        if config.provider_preset is ProviderPreset.OLLAMA:
            base_url = config.base_url.strip()
            if base_url == "":
                raise ValueError("Ollama requires a base URL.")
            return self._provider_factory.for_ollama(
                model=model,
                base_url=base_url,
                api_key=config.api_key.strip() or "ollama",
            )

        base_url = config.base_url.strip()
        if base_url == "":
            raise ValueError("Custom OpenAI-compatible providers require a base URL.")

        return self._provider_factory(
            model=model,
            base_url=base_url,
            api_key=config.api_key.strip() or None,
        )

    @staticmethod
    def _parse_json_object(
        arguments_text: str,
        *,
        error_prefix: str,
    ) -> dict[str, Any]:
        try:
            parsed = json.loads(arguments_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{error_prefix} must be valid JSON.") from exc

        if not isinstance(parsed, dict):
            raise ValueError(f"{error_prefix} must decode to a JSON object.")

        return parsed

    @staticmethod
    def _make_context(
        config: WorkbenchConfigState,
        *,
        invocation_id: str,
    ) -> ToolContext:
        return ToolContext(
            invocation_id=invocation_id,
            workspace=config.workspace.strip() or None,
            env=dict(os.environ),
        )
