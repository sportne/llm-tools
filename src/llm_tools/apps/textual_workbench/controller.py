"""Pure-Python controller logic for the Textual workbench."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from llm_tools.apps.textual_workbench.models import (
    ApprovalFinalizeResult,
    ApprovalResolutionResult,
    DirectExecutionResult,
    ExportToolsResult,
    ModelTurnExecutionResult,
    ProviderModeStrategy,
    ProviderPreset,
    WorkbenchConfigState,
)
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.llm_providers import ProviderModeStrategy as ProviderRunMode
from llm_tools.tool_api import (
    SideEffectClass,
    ToolContext,
    ToolInvocationRequest,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
)
from llm_tools.tools import (
    register_atlassian_tools,
    register_filesystem_tools,
    register_git_tools,
    register_text_tools,
)
from llm_tools.workflow_api import (
    ApprovalRequest,
    WorkflowExecutor,
    WorkflowInvocationStatus,
    WorkflowTurnResult,
)


@dataclass(slots=True)
class _WorkbenchSession:
    """In-memory executor session tied to one config signature."""

    signature: str
    executor: WorkflowExecutor


class WorkbenchController:
    """Build registries, providers, and execution flows for the workbench."""

    def __init__(
        self,
        *,
        provider_factory: type[OpenAICompatibleProvider] = OpenAICompatibleProvider,
    ) -> None:
        self._provider_factory = provider_factory
        self._active_session: _WorkbenchSession | None = None

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

    def cycle_provider_mode_strategy(
        self, config: WorkbenchConfigState
    ) -> WorkbenchConfigState:
        """Return a config with the next provider mode strategy selected."""
        ordered = [
            ProviderModeStrategy.AUTO,
            ProviderModeStrategy.TOOLS,
            ProviderModeStrategy.JSON,
            ProviderModeStrategy.MD_JSON,
        ]
        current_index = ordered.index(config.provider_mode_strategy)
        next_mode = ordered[(current_index + 1) % len(ordered)]
        return config.model_copy(update={"provider_mode_strategy": next_mode})

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

    def list_pending_approvals(
        self, config: WorkbenchConfigState
    ) -> list[ApprovalRequest]:
        """Return pending approvals from the active executor session."""
        executor, _ = self._get_or_rebuild_session(config)
        return executor.list_pending_approvals()

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

    def export_tools(self, config: WorkbenchConfigState) -> ExportToolsResult:
        """Export tools for the canonical action-envelope adapter."""
        adapter = self._build_adapter()
        context = self._make_context(config, invocation_id="workbench-export-tools")
        executor, session_rebuilt = self._get_or_rebuild_session(config)
        prepared = executor.prepare_model_interaction(adapter, context=context)
        return ExportToolsResult(
            session_rebuilt=session_rebuilt,
            exported_tools=prepared.schema,
        )

    def execute_direct_tool(
        self,
        config: WorkbenchConfigState,
        *,
        tool_name: str,
        arguments_text: str,
    ) -> DirectExecutionResult:
        """Execute one tool through workflow execution to support approvals."""
        normalized_tool_name = tool_name.strip()
        if normalized_tool_name == "":
            raise ValueError("A tool name is required for direct execution.")

        arguments = self._parse_json_object(
            arguments_text,
            error_prefix="Direct tool arguments",
        )
        context = self._make_context(config, invocation_id="workbench-direct-tool")
        executor, session_rebuilt = self._get_or_rebuild_session(config)
        workflow_result = executor.execute_parsed_response(
            ParsedModelResponse(
                invocations=[
                    ToolInvocationRequest(
                        tool_name=normalized_tool_name,
                        arguments=arguments,
                    )
                ]
            ),
            context,
        )
        return DirectExecutionResult(
            session_rebuilt=session_rebuilt,
            workflow_result=workflow_result,
            tool_result=self._latest_tool_result(workflow_result),
        )

    async def execute_direct_tool_async(
        self,
        config: WorkbenchConfigState,
        *,
        tool_name: str,
        arguments_text: str,
    ) -> DirectExecutionResult:
        """Asynchronously execute one tool through workflow execution."""
        normalized_tool_name = tool_name.strip()
        if normalized_tool_name == "":
            raise ValueError("A tool name is required for direct execution.")

        arguments = self._parse_json_object(
            arguments_text,
            error_prefix="Direct tool arguments",
        )
        context = self._make_context(config, invocation_id="workbench-direct-tool")
        executor, session_rebuilt = self._get_or_rebuild_session(config)
        workflow_result = await executor.execute_parsed_response_async(
            ParsedModelResponse(
                invocations=[
                    ToolInvocationRequest(
                        tool_name=normalized_tool_name,
                        arguments=arguments,
                    )
                ]
            ),
            context,
        )
        return DirectExecutionResult(
            session_rebuilt=session_rebuilt,
            workflow_result=workflow_result,
            tool_result=self._latest_tool_result(workflow_result),
        )

    def run_model_turn(
        self,
        config: WorkbenchConfigState,
        *,
        prompt: str,
    ) -> ModelTurnExecutionResult:
        """Run one provider-backed model turn through the action envelope."""
        normalized_prompt = prompt.strip()
        if normalized_prompt == "":
            raise ValueError("A prompt is required to run a model turn.")

        adapter = self._build_adapter()
        policy_context = self._make_context(
            config, invocation_id="workbench-model-turn"
        )
        executor, session_rebuilt = self._get_or_rebuild_session(config)
        prepared = executor.prepare_model_interaction(
            adapter,
            context=policy_context,
            include_requires_approval=True,
        )
        provider = self._build_provider(config)
        parsed = provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": normalized_prompt}],
            response_model=prepared.response_model,
        )

        workflow_result = None
        if config.execute_after_parse:
            workflow_result = executor.execute_parsed_response(parsed, policy_context)

        return ModelTurnExecutionResult(
            session_rebuilt=session_rebuilt,
            exported_tools=prepared.schema,
            parsed_response=parsed,
            workflow_result=workflow_result,
        )

    async def run_model_turn_async(
        self,
        config: WorkbenchConfigState,
        *,
        prompt: str,
    ) -> ModelTurnExecutionResult:
        """Asynchronously run one provider-backed model turn."""
        normalized_prompt = prompt.strip()
        if normalized_prompt == "":
            raise ValueError("A prompt is required to run a model turn.")

        adapter = self._build_adapter()
        policy_context = self._make_context(
            config, invocation_id="workbench-model-turn"
        )
        executor, session_rebuilt = self._get_or_rebuild_session(config)
        prepared = executor.prepare_model_interaction(
            adapter,
            context=policy_context,
            include_requires_approval=True,
        )
        provider = self._build_provider(config)
        parsed = await provider.run_async(
            adapter=adapter,
            messages=[{"role": "user", "content": normalized_prompt}],
            response_model=prepared.response_model,
        )

        workflow_result = None
        if config.execute_after_parse:
            workflow_result = await executor.execute_parsed_response_async(
                parsed, policy_context
            )

        return ModelTurnExecutionResult(
            session_rebuilt=session_rebuilt,
            exported_tools=prepared.schema,
            parsed_response=parsed,
            workflow_result=workflow_result,
        )

    def resolve_pending_approval(
        self,
        config: WorkbenchConfigState,
        *,
        approval_id: str,
        approved: bool,
    ) -> ApprovalResolutionResult:
        """Resolve one pending approval and continue execution."""
        executor, session_rebuilt = self._get_or_rebuild_session(config)
        workflow_result = executor.resolve_pending_approval(
            approval_id=approval_id,
            approved=approved,
        )
        return ApprovalResolutionResult(
            session_rebuilt=session_rebuilt,
            workflow_result=workflow_result,
        )

    async def resolve_pending_approval_async(
        self,
        config: WorkbenchConfigState,
        *,
        approval_id: str,
        approved: bool,
    ) -> ApprovalResolutionResult:
        """Asynchronously resolve one pending approval and continue execution."""
        executor, session_rebuilt = self._get_or_rebuild_session(config)
        workflow_result = await executor.resolve_pending_approval_async(
            approval_id=approval_id,
            approved=approved,
        )
        return ApprovalResolutionResult(
            session_rebuilt=session_rebuilt,
            workflow_result=workflow_result,
        )

    def finalize_expired_approvals(
        self,
        config: WorkbenchConfigState,
    ) -> ApprovalFinalizeResult:
        """Finalize approvals past their configured expiration time."""
        executor, session_rebuilt = self._get_or_rebuild_session(config)
        return ApprovalFinalizeResult(
            session_rebuilt=session_rebuilt,
            workflow_results=executor.finalize_expired_approvals(),
        )

    async def finalize_expired_approvals_async(
        self,
        config: WorkbenchConfigState,
    ) -> ApprovalFinalizeResult:
        """Asynchronously finalize approvals past their expiration."""
        executor, session_rebuilt = self._get_or_rebuild_session(config)
        return ApprovalFinalizeResult(
            session_rebuilt=session_rebuilt,
            workflow_results=await executor.finalize_expired_approvals_async(),
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

        require_approval_for: set[SideEffectClass] = set()
        if config.require_approval_for_local_read:
            require_approval_for.add(SideEffectClass.LOCAL_READ)
        if config.require_approval_for_local_write:
            require_approval_for.add(SideEffectClass.LOCAL_WRITE)
        if config.require_approval_for_external_read:
            require_approval_for.add(SideEffectClass.EXTERNAL_READ)
        if config.require_approval_for_external_write:
            require_approval_for.add(SideEffectClass.EXTERNAL_WRITE)

        return ToolPolicy(
            allowed_side_effects=allowed_side_effects,
            require_approval_for=require_approval_for,
            allow_network=config.allow_network,
            allow_filesystem=config.allow_filesystem,
            allow_subprocess=config.allow_subprocess,
        )

    @staticmethod
    def _build_adapter() -> ActionEnvelopeAdapter:
        return ActionEnvelopeAdapter()

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
                mode_strategy=ProviderRunMode(config.provider_mode_strategy.value),
            )

        if config.provider_preset is ProviderPreset.OLLAMA:
            base_url = config.base_url.strip()
            if base_url == "":
                raise ValueError("Ollama requires a base URL.")
            return self._provider_factory.for_ollama(
                model=model,
                base_url=base_url,
                api_key=config.api_key.strip() or "ollama",
                mode_strategy=ProviderRunMode(config.provider_mode_strategy.value),
            )

        base_url = config.base_url.strip()
        if base_url == "":
            raise ValueError("Custom OpenAI-compatible providers require a base URL.")

        return self._provider_factory(
            model=model,
            base_url=base_url,
            api_key=config.api_key.strip() or None,
            mode_strategy=ProviderRunMode(config.provider_mode_strategy.value),
        )

    def _get_or_rebuild_session(
        self,
        config: WorkbenchConfigState,
    ) -> tuple[WorkflowExecutor, bool]:
        signature = self._session_signature(config)
        if (
            self._active_session is not None
            and self._active_session.signature == signature
        ):
            return self._active_session.executor, False

        executor = WorkflowExecutor(
            registry=self.build_registry(config),
            policy=self.build_policy(config),
        )
        session_rebuilt = self._active_session is not None
        self._active_session = _WorkbenchSession(signature=signature, executor=executor)
        return executor, session_rebuilt

    @staticmethod
    def _session_signature(config: WorkbenchConfigState) -> str:
        return json.dumps(config.model_dump(mode="json"), sort_keys=True)

    @staticmethod
    def _latest_tool_result(workflow_result: WorkflowTurnResult) -> ToolResult | None:
        for outcome in reversed(workflow_result.outcomes):
            if (
                outcome.status is WorkflowInvocationStatus.EXECUTED
                and outcome.tool_result is not None
            ):
                return outcome.tool_result
        return None

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
