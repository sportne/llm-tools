"""Additional assistant runtime coverage for protection-aware branches."""

from __future__ import annotations

import asyncio
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_tools.apps.assistant_config import AssistantConfig
from llm_tools.apps.assistant_runtime import (
    AssistantHarnessTurnProvider,
    build_assistant_available_tool_specs,
    build_assistant_policy,
    build_assistant_registry,
    build_live_harness_provider,
    build_tool_capabilities,
    resolve_assistant_default_enabled_tools,
)
from llm_tools.apps.chat_config import ChatPolicyConfig
from llm_tools.harness_api import (
    BudgetPolicy,
    HarnessSession,
    HarnessState,
    HarnessTurn,
    TaskOrigin,
    TaskRecord,
)
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import (
    SideEffectClass,
    SourceProvenanceRef,
    ToolContext,
    ToolResult,
)
from llm_tools.workflow_api import (
    PromptProtectionDecision,
    ProtectionAction,
    ProtectionConfig,
    ResponseProtectionDecision,
    WorkflowInvocationOutcome,
    WorkflowInvocationStatus,
    WorkflowTurnResult,
)
from llm_tools.workflow_api.executor import PreparedModelInteraction
from llm_tools.workflow_api.staged_structured import repair_stage_guidance


class _Provider:
    def __init__(self, response: ParsedModelResponse) -> None:
        self.response = response
        self.messages: list[dict[str, str]] | None = None
        self.run_calls = 0
        self.async_calls = 0

    def run(self, **kwargs: object) -> ParsedModelResponse:
        self.run_calls += 1
        self.messages = list(kwargs["messages"])  # type: ignore[index]
        return self.response

    async def run_async(self, **kwargs: object) -> ParsedModelResponse:
        self.async_calls += 1
        self.messages = list(kwargs["messages"])  # type: ignore[index]
        return self.response


class _ProtectionController:
    def __init__(
        self,
        *,
        prompt_decision: PromptProtectionDecision,
        response_decision: ResponseProtectionDecision,
    ) -> None:
        self.prompt_decision = prompt_decision
        self.response_decision = response_decision
        self.prompt_calls: list[dict[str, object]] = []
        self.response_calls: list[dict[str, object]] = []

    def assess_prompt(self, **kwargs) -> PromptProtectionDecision:
        self.prompt_calls.append(dict(kwargs))
        return self.prompt_decision

    async def assess_prompt_async(self, **kwargs) -> PromptProtectionDecision:
        self.prompt_calls.append(dict(kwargs))
        return self.prompt_decision

    def review_response(self, **kwargs) -> ResponseProtectionDecision:
        self.response_calls.append(dict(kwargs))
        return self.response_decision

    async def review_response_async(self, **kwargs) -> ResponseProtectionDecision:
        self.response_calls.append(dict(kwargs))
        return self.response_decision


class _StagedProvider:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.sync_messages: list[list[dict[str, str]]] = []
        self.async_messages: list[list[dict[str, str]]] = []
        self.response_model_names: list[str] = []

    def prefers_simplified_json_schema_contract(self) -> bool:
        return False

    def uses_staged_schema_protocol(self) -> bool:
        return True

    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        raise AssertionError("staged provider should use run_structured()")

    async def run_async(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        raise AssertionError("staged provider should use run_structured_async()")

    def run_structured(self, **kwargs: object) -> object:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.sync_messages.append([dict(message) for message in messages])
        response_model = kwargs["response_model"]
        assert isinstance(response_model, type)
        self.response_model_names.append(response_model.__name__)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def run_structured_async(self, **kwargs: object) -> object:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.async_messages.append([dict(message) for message in messages])
        response_model = kwargs["response_model"]
        assert isinstance(response_model, type)
        self.response_model_names.append(response_model.__name__)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class _PromptToolProvider:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.sync_messages: list[list[dict[str, str]]] = []
        self.async_messages: list[list[dict[str, str]]] = []

    def prefers_simplified_json_schema_contract(self) -> bool:
        return False

    def uses_prompt_tool_protocol(self) -> bool:
        return True

    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        raise AssertionError("prompt-tool provider should use run_text()")

    async def run_async(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        raise AssertionError("prompt-tool provider should use run_text_async()")

    def run_text(self, **kwargs: object) -> str:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.sync_messages.append([dict(message) for message in messages])
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, str)
        return response

    async def run_text_async(self, **kwargs: object) -> str:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.async_messages.append([dict(message) for message in messages])
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, str)
        return response


class _FallbackPromptToolProvider(_PromptToolProvider):
    def __init__(self, responses: list[object], *, staged: bool) -> None:
        super().__init__(responses)
        self._staged = staged
        self.run_calls = 0
        self.async_run_calls = 0
        self.structured_calls = 0
        self.async_structured_calls = 0

    def uses_prompt_tool_protocol(self) -> bool:
        return False

    def uses_staged_schema_protocol(self) -> bool:
        return self._staged

    def can_fallback_to_prompt_tools(self, exc: Exception) -> bool:
        del exc
        return True

    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        self.run_calls += 1
        raise RuntimeError("native parse failed")

    async def run_async(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        self.async_run_calls += 1
        raise RuntimeError("native async parse failed")

    def run_structured(self, **kwargs: object) -> object:
        del kwargs
        self.structured_calls += 1
        raise RuntimeError("staged parse failed")

    async def run_structured_async(self, **kwargs: object) -> object:
        del kwargs
        self.async_structured_calls += 1
        raise RuntimeError("staged async parse failed")


def _prepared_research_interaction(
    *, tool_name: str = "read_file"
) -> PreparedModelInteraction:
    registry = build_assistant_registry()
    binding = next(
        item for item in registry.list_bindings() if item.spec.name == tool_name
    )
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_final_response_step_model(final_response_model=str)
    return PreparedModelInteraction(
        response_model=response_model,
        schema=adapter.export_schema(response_model),
        tool_names=[binding.spec.name],
        tool_specs=[binding.spec],
        input_models={binding.spec.name: binding.input_model},
    )


def _prepared_research_interaction_for_tools(
    tool_names: list[str],
) -> PreparedModelInteraction:
    registry = build_assistant_registry()
    bindings = [
        item for item in registry.list_bindings() if item.spec.name in set(tool_names)
    ]
    bindings.sort(key=lambda item: tool_names.index(item.spec.name))
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_final_response_step_model(final_response_model=str)
    return PreparedModelInteraction(
        response_model=response_model,
        schema=adapter.export_schema(response_model),
        tool_names=[binding.spec.name for binding in bindings],
        tool_specs=[binding.spec for binding in bindings],
        input_models={binding.spec.name: binding.input_model for binding in bindings},
    )


def _state_with_provenance() -> HarnessState:
    source = SourceProvenanceRef(
        source_kind="local_file",
        source_id="/workspace/secret.txt",
        content_hash="abc123",
        whole_source_reproduction_allowed=True,
    )
    return HarnessState(
        schema_version="3",
        session=HarnessSession(
            session_id="session-1",
            root_task_id="task-1",
            budget_policy=BudgetPolicy(max_turns=3),
            started_at="2026-01-01T00:00:00Z",
            current_turn_index=1,
        ),
        tasks=[
            TaskRecord(
                task_id="task-1",
                title="Task",
                intent="Do work",
                origin=TaskOrigin.USER_REQUESTED,
            )
        ],
        turns=[
            HarnessTurn(
                turn_index=1,
                started_at="2026-01-01T00:00:00Z",
                selected_task_ids=["task-1"],
                workflow_result=WorkflowTurnResult(
                    parsed_response=ParsedModelResponse(
                        invocations=[
                            {"tool_name": "read_file", "arguments": {"path": "."}}
                        ]
                    ),
                    outcomes=[
                        WorkflowInvocationOutcome(
                            invocation_index=1,
                            request={
                                "tool_name": "read_file",
                                "arguments": {"path": "."},
                            },
                            status=WorkflowInvocationStatus.EXECUTED,
                            tool_result=ToolResult(
                                ok=True,
                                tool_name="read_file",
                                tool_version="0.1.0",
                                output={"path": "secret.txt"},
                                source_provenance=[source],
                            ),
                        )
                    ],
                ),
                decision={
                    "action": "continue",
                    "selected_task_ids": ["task-1"],
                },
                ended_at="2026-01-01T00:00:01Z",
            )
        ],
    )


def test_assistant_runtime_helper_edges_cover_intersection_permissions_and_missing_specs() -> (
    None
):
    import llm_tools.apps.assistant_runtime as assistant_runtime

    config = AssistantConfig(
        policy=ChatPolicyConfig(enabled_tools=["read_file", "missing_tool"])
    )
    tool_specs = build_assistant_available_tool_specs()
    other_spec = tool_specs["read_file"].model_copy(
        update={"name": "other_tool", "tags": ["misc"]}
    )
    capabilities = build_tool_capabilities(
        tool_specs={"other_tool": other_spec},
        enabled_tools={"other_tool"},
        root_path=".",
        env={},
        allow_network=True,
        allow_filesystem=False,
        allow_subprocess=True,
        require_approval_for=set(),
    )
    policy = build_assistant_policy(
        enabled_tools={"read_file", "missing_tool"},
        tool_specs=tool_specs,
        require_approval_for=set(),
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=False,
        redaction_config=config.policy.redaction,
    )

    assert resolve_assistant_default_enabled_tools(config) == {"read_file"}
    assert assistant_runtime.assistant_tool_group(other_spec) == "Other"
    assert capabilities["Other"][0].status == "permission_blocked"
    assert SideEffectClass.LOCAL_READ in policy.allowed_side_effects


def test_assistant_harness_turn_provider_returns_safe_message_for_sync_challenge() -> (
    None
):
    provider = _Provider(ParsedModelResponse(final_response="unused"))
    harness_provider = AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
        protection_controller=_ProtectionController(
            prompt_decision=PromptProtectionDecision(
                action=ProtectionAction.CHALLENGE,
                challenge_message="Please confirm the sensitivity analysis.",
            ),
            response_decision=ResponseProtectionDecision(action=ProtectionAction.ALLOW),
        ),
    )
    context = ToolContext(invocation_id="turn-1", metadata={})

    response = harness_provider.run(
        state=object(),
        selected_task_ids=["task-1"],
        context=context,
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=SimpleNamespace(response_model=str),
    )

    assert response.final_response == "Please confirm the sensitivity analysis."
    assert provider.run_calls == 0
    assert context.metadata["protection_review"]["purge_requested"] is True


def test_assistant_harness_turn_provider_sync_constrains_sanitizes_and_uses_prior_provenance() -> (
    None
):
    provider = _Provider(ParsedModelResponse(final_response="secret"))
    protection_controller = _ProtectionController(
        prompt_decision=PromptProtectionDecision(
            action=ProtectionAction.CONSTRAIN,
            guard_text="stay within policy",
        ),
        response_decision=ResponseProtectionDecision(
            action=ProtectionAction.SANITIZE,
            sanitized_payload="clean",
            should_purge=True,
        ),
    )
    harness_provider = AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
        protection_controller=protection_controller,
    )
    context = ToolContext(
        invocation_id="turn-2",
        metadata={"harness_turn_context": {"turn_index": 2}},
    )

    response = harness_provider.run(
        state=_state_with_provenance(),
        selected_task_ids=["task-2"],
        context=context,
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=SimpleNamespace(response_model=str),
    )

    assert response.final_response == "clean"
    assert provider.messages is not None
    assert provider.messages[0]["content"] == "stay within policy"
    assert '"task-2"' in provider.messages[2]["content"]
    assert (
        protection_controller.prompt_calls[0]["provenance"].sources[0].source_id
        == "/workspace/secret.txt"
    )
    assert (
        protection_controller.response_calls[0]["provenance"].sources[0].source_id
        == "/workspace/secret.txt"
    )
    assert context.metadata["protection_review"] == {
        "purge_requested": True,
        "safe_message": "clean",
    }


def test_assistant_harness_turn_provider_sync_blocks_response() -> None:
    provider = _Provider(ParsedModelResponse(final_response="secret"))
    harness_provider = AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
        protection_controller=_ProtectionController(
            prompt_decision=PromptProtectionDecision(action=ProtectionAction.ALLOW),
            response_decision=ResponseProtectionDecision(
                action=ProtectionAction.BLOCK,
                safe_message="Blocked for this environment.",
                should_purge=True,
            ),
        ),
    )
    context = ToolContext(invocation_id="turn-3", metadata={})

    response = harness_provider.run(
        state=object(),
        selected_task_ids=["task-3"],
        context=context,
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=SimpleNamespace(response_model=str),
    )

    assert response.final_response == "Blocked for this environment."
    assert context.metadata["protection_review"] == {
        "purge_requested": True,
        "safe_message": "Blocked for this environment.",
    }


def test_assistant_harness_turn_provider_async_blocks_response() -> None:
    provider = _Provider(ParsedModelResponse(final_response="secret"))
    harness_provider = AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
        protection_controller=_ProtectionController(
            prompt_decision=PromptProtectionDecision(action=ProtectionAction.ALLOW),
            response_decision=ResponseProtectionDecision(
                action=ProtectionAction.BLOCK,
                safe_message="Blocked for this environment.",
                should_purge=True,
            ),
        ),
    )
    context = ToolContext(invocation_id="turn-4", metadata={})

    response = asyncio.run(
        harness_provider.run_async(
            state=object(),
            selected_task_ids=["task-4"],
            context=context,
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=SimpleNamespace(response_model=str),
        )
    )

    assert response.final_response == "Blocked for this environment."
    assert provider.async_calls == 1
    assert context.metadata["protection_review"] == {
        "purge_requested": True,
        "safe_message": "Blocked for this environment.",
    }


def test_build_live_harness_provider_wraps_created_provider_and_wires_protection(
    monkeypatch, tmp_path: Path
) -> None:
    created = _Provider(ParsedModelResponse(final_response="done"))
    protection_controller = object()
    environment_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "llm_tools.apps.assistant_runtime.create_provider",
        lambda *args, **kwargs: created,
    )
    monkeypatch.setattr(
        "llm_tools.apps.assistant_runtime.build_protection_environment",
        lambda **kwargs: (
            environment_calls.append(dict(kwargs)) or {"environment": "ok"}
        ),
    )
    monkeypatch.setattr(
        "llm_tools.apps.assistant_runtime.build_protection_controller",
        lambda **kwargs: protection_controller,
    )

    harness_provider = build_live_harness_provider(
        config=AssistantConfig(
            protection=ProtectionConfig(
                enabled=True, document_paths=[str(tmp_path / "policy.txt")]
            )
        ),
        provider_config=AssistantConfig().llm,
        model_name="demo-model",
        api_key=None,
        mode_strategy="auto",
        tool_registry=build_assistant_registry(),
        enabled_tool_names={"read_file"},
        workspace_enabled=tmp_path.exists(),
        workspace=str(tmp_path),
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=False,
    )

    assert isinstance(harness_provider, AssistantHarnessTurnProvider)
    assert harness_provider._provider is created
    assert harness_provider._protection_controller is protection_controller
    assert environment_calls[0]["workspace"] == str(tmp_path)
    assert environment_calls[0]["allow_filesystem"] is True
    assert "durable research assistant" in harness_provider._system_prompt


def test_assistant_harness_turn_provider_sync_passthrough_without_protection() -> None:
    provider = _Provider(ParsedModelResponse(final_response="plain"))
    harness_provider = AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )
    context = ToolContext(invocation_id="turn-plain-sync", metadata={})

    response = harness_provider.run(
        state=object(),
        selected_task_ids=["task-sync"],
        context=context,
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=SimpleNamespace(response_model=str),
    )

    assert response.final_response == "plain"
    assert context.metadata == {}


def test_assistant_harness_turn_provider_async_challenges_before_provider_call() -> (
    None
):
    provider = _Provider(ParsedModelResponse(final_response="unused"))
    harness_provider = AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
        protection_controller=_ProtectionController(
            prompt_decision=PromptProtectionDecision(
                action=ProtectionAction.CHALLENGE,
                challenge_message="Async sensitivity challenge.",
            ),
            response_decision=ResponseProtectionDecision(action=ProtectionAction.ALLOW),
        ),
    )
    context = ToolContext(invocation_id="turn-async-challenge", metadata={})

    response = asyncio.run(
        harness_provider.run_async(
            state=object(),
            selected_task_ids=["task-async"],
            context=context,
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=SimpleNamespace(response_model=str),
        )
    )

    assert response.final_response == "Async sensitivity challenge."
    assert provider.async_calls == 0
    assert context.metadata["protection_review"]["purge_requested"] is True


def test_assistant_harness_turn_provider_async_constrains_and_sanitizes() -> None:
    provider = _Provider(ParsedModelResponse(final_response="secret"))
    protection_controller = _ProtectionController(
        prompt_decision=PromptProtectionDecision(
            action=ProtectionAction.CONSTRAIN,
            guard_text="async-guard",
        ),
        response_decision=ResponseProtectionDecision(
            action=ProtectionAction.SANITIZE,
            sanitized_payload="async-clean",
            should_purge=True,
        ),
    )
    harness_provider = AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
        protection_controller=protection_controller,
    )
    context = ToolContext(
        invocation_id="turn-async-sanitize",
        metadata={"harness_turn_context": {"turn_index": 3}},
    )

    response = asyncio.run(
        harness_provider.run_async(
            state=_state_with_provenance(),
            selected_task_ids=["task-async-2"],
            context=context,
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=SimpleNamespace(response_model=str),
        )
    )

    assert response.final_response == "async-clean"
    assert provider.messages is not None
    assert provider.messages[0]["content"] == "async-guard"
    assert (
        protection_controller.prompt_calls[0]["provenance"].sources[0].source_id
        == "/workspace/secret.txt"
    )
    assert context.metadata["protection_review"] == {
        "purge_requested": True,
        "safe_message": "async-clean",
    }


def test_assistant_harness_turn_provider_async_allows_passthrough_after_review() -> (
    None
):
    provider = _Provider(ParsedModelResponse(final_response="allowed"))
    harness_provider = AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
        protection_controller=_ProtectionController(
            prompt_decision=PromptProtectionDecision(action=ProtectionAction.ALLOW),
            response_decision=ResponseProtectionDecision(action=ProtectionAction.ALLOW),
        ),
    )
    context = ToolContext(invocation_id="turn-async-allow", metadata={})

    response = asyncio.run(
        harness_provider.run_async(
            state=object(),
            selected_task_ids=["task-allow"],
            context=context,
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=SimpleNamespace(response_model=str),
        )
    )

    assert response.final_response == "allowed"
    assert provider.async_calls == 1
    assert context.metadata == {}


def test_direct_research_provider_builder_module_is_wired(
    monkeypatch, tmp_path: Path
) -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    created = _Provider(ParsedModelResponse(final_response="done"))
    monkeypatch.setattr(
        provider_module, "create_provider", lambda *args, **kwargs: created
    )
    monkeypatch.setattr(
        provider_module,
        "build_protection_environment",
        lambda **kwargs: {"environment": kwargs["workspace"]},
    )
    monkeypatch.setattr(
        provider_module,
        "build_protection_controller",
        lambda **kwargs: {"controller": kwargs["environment"]},
    )
    monkeypatch.setattr(
        provider_module,
        "build_research_system_prompt",
        lambda **kwargs: "patched research system prompt",
    )

    harness_provider = provider_module.build_live_harness_provider(
        config=AssistantConfig(),
        provider_config=AssistantConfig().llm,
        model_name="demo-model",
        api_key=None,
        mode_strategy="auto",
        tool_registry=build_assistant_registry(),
        enabled_tool_names={"read_file"},
        workspace_enabled=True,
        workspace=str(tmp_path),
        allow_network=False,
        allow_filesystem=True,
        allow_subprocess=False,
    )

    assert harness_provider._provider is created
    assert harness_provider._protection_controller == {
        "controller": {"environment": str(tmp_path)}
    }
    assert harness_provider._system_prompt == "patched research system prompt"


def test_direct_research_provider_runs_staged_tool_flow() -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    provider = _StagedProvider(
        [
            {"mode": "tool", "tool_name": "read_file"},
            {
                "mode": "tool",
                "tool_name": "read_file",
                "arguments": {"path": "README.md"},
            },
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.2,
        system_prompt="research-system",
    )
    context = ToolContext(
        invocation_id="research-staged-sync",
        metadata={"harness_turn_context": {"turn_index": 4}},
    )

    parsed = harness_provider.run(
        state=object(),
        selected_task_ids=["task-1"],
        context=context,
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=_prepared_research_interaction(),
    )

    assert parsed.invocations[0].tool_name == "read_file"
    assert parsed.invocations[0].arguments["path"] == "README.md"
    assert provider.response_model_names == [
        "DecisionStep",
        "ReadFileInvocationStep",
    ]
    assert provider.sync_messages[0][-1]["content"].startswith(
        "Current step: choose the next action."
    )
    assert (
        "invoke the selected tool 'read_file'"
        in provider.sync_messages[1][-1]["content"]
    )


def test_direct_research_provider_repairs_invalid_sync_stage_and_exposes_helpers() -> (
    None
):
    import llm_tools.apps.assistant_research_provider as provider_module

    provider = _StagedProvider(
        [
            {"mode": "finalize", "tool_name": "read_file"},
            {"mode": "finalize"},
            {"mode": "finalize", "final_response": "done"},
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )
    context = ToolContext(invocation_id="research-repair-sync", metadata={})

    parsed = harness_provider.run(
        state=object(),
        selected_task_ids=["task-2"],
        context=context,
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=_prepared_research_interaction(),
    )

    assert parsed.final_response == "done"
    repair_message = provider.sync_messages[1][-1]["content"]
    assert "The previous decision response was invalid." in repair_message
    assert (
        "Decision stage rules: return only mode and, when mode='tool', one tool_name."
        in repair_message
    )
    assert provider_module._uses_staged_schema_protocol(object()) is False
    assert provider_module._uses_staged_schema_protocol(provider) is True
    assert (
        provider_module.AssistantHarnessTurnProvider(
            provider=provider,  # type: ignore[arg-type]
            temperature=0.1,
            system_prompt="research-system",
        ).prefers_simplified_json_schema_contract()
        is False
    )
    assert (
        provider_module.AssistantHarnessTurnProvider._validation_error_summary(
            RuntimeError("")
        )
        == "RuntimeError"
    )
    assert (
        provider_module.AssistantHarnessTurnProvider._format_invalid_payload(None)
        == "(unavailable)"
    )
    assert (
        provider_module.AssistantHarnessTurnProvider._format_invalid_payload("payload")
        == "payload"
    )
    bad_keys = {object(): "value"}
    assert provider_module.AssistantHarnessTurnProvider._format_invalid_payload(
        bad_keys
    ) == str(bad_keys)
    assert (
        repair_stage_guidance("other")
        == "Return only the fields required for this stage."
    )
    assert repair_stage_guidance("final_response").startswith(
        "Finalization stage rules:"
    )
    try:
        provider_module.AssistantHarnessTurnProvider._tool_spec([], "missing")
    except ValueError as exc:
        assert "Unknown tool selected during staged interaction" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ValueError for missing staged tool")


def test_direct_research_provider_runs_staged_async_and_repairs_provider_failure() -> (
    None
):
    import llm_tools.apps.assistant_research_provider as provider_module

    error = ValueError("temporary parse failure")
    error.invalid_payload = {"mode": "tool"}  # type: ignore[attr-defined]
    provider = _StagedProvider(
        [
            error,
            {"mode": "tool", "tool_name": "read_file"},
            {
                "mode": "tool",
                "tool_name": "read_file",
                "arguments": {"path": "README.md"},
            },
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )
    context = ToolContext(
        invocation_id="research-staged-async",
        metadata={"harness_turn_context": {"turn_index": 5}},
    )

    parsed = asyncio.run(
        harness_provider.run_async(
            state=object(),
            selected_task_ids=["task-3"],
            context=context,
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=_prepared_research_interaction(),
        )
    )

    assert parsed.invocations[0].tool_name == "read_file"
    assert provider.async_messages[1][-1]["content"].startswith(
        "The previous decision response was invalid."
    )


def test_direct_research_provider_raises_after_two_staged_repairs() -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    provider = _StagedProvider(
        [
            ValueError("schema validation failed"),
            ValueError("schema validation failed again"),
            ValueError("schema validation failed last"),
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    try:
        asyncio.run(
            harness_provider.run_async(
                state=object(),
                selected_task_ids=["task-4"],
                context=ToolContext(invocation_id="research-fail-async", metadata={}),
                adapter=ActionEnvelopeAdapter(),
                prepared_interaction=_prepared_research_interaction(),
            )
        )
    except ValueError as exc:
        assert str(exc) == "schema validation failed last"
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected repeated staged failure to propagate")

    assert len(provider.async_messages) == 3
    assert provider.async_messages[1][-1]["content"].startswith(
        "The previous decision response was invalid."
    )
    assert provider.async_messages[2][-1]["content"].startswith(
        "The previous decision response was invalid."
    )


def test_direct_research_provider_prompt_tools_repairs_decision_and_tool_stage(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    provider = _PromptToolProvider(
        [
            "```decision\nMODE: tool\nTOOL_NAME: read_file\n```",
            "```tool\nTOOL_NAME: read_file\nBEGIN_ARG: path\nREADME.md\n```",
            ("```tool\nTOOL_NAME: read_file\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```"),
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    parsed = harness_provider.run(
        state=object(),
        selected_task_ids=["task-5"],
        context=ToolContext(invocation_id="research-prompt-sync", metadata={}),
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=_prepared_research_interaction(),
    )

    assert parsed.invocations[0].tool_name == "read_file"
    assert parsed.invocations[0].arguments["path"] == "README.md"
    assert parsed.invocations[0].tool_call_id is not None
    assert len(provider.sync_messages) == 3
    assert (
        "The previous tool:read_file response was invalid."
        in (provider.sync_messages[2][-1]["content"])
    )
    tool_repair = provider.sync_messages[2][-1]["content"]
    assert "Selected tool schema:" in tool_repair


def test_direct_research_provider_prompt_tools_async_repairs_final_response(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    provider = _PromptToolProvider(
        [
            "```decision\nMODE: finalize\n```",
            "```final\nANSWER:\n```",
            "```final\nANSWER:\nDone async.\n```",
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    parsed = asyncio.run(
        harness_provider.run_async(
            state=object(),
            selected_task_ids=["task-6"],
            context=ToolContext(invocation_id="research-prompt-async", metadata={}),
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=_prepared_research_interaction(),
        )
    )

    assert parsed.final_response == "Done async."
    assert len(provider.async_messages) == 3
    assert "Finalization rules:" in provider.async_messages[2][-1]["content"]


def test_direct_research_provider_prompt_tools_single_action_env_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "single_action")
    provider = _PromptToolProvider(
        [
            "```tool\nTOOL_NAME: read_file\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```",
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    parsed = harness_provider.run(
        state=object(),
        selected_task_ids=["task-single-action"],
        context=ToolContext(invocation_id="research-single-action", metadata={}),
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=_prepared_research_interaction(),
    )

    assert parsed.invocations[0].tool_name == "read_file"
    assert "Prompt-tool output contract:" in provider.sync_messages[0][-1]["content"]


def test_direct_research_provider_prompt_tools_async_split_tool_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    provider = _PromptToolProvider(
        [
            "```decision\nMODE: tool\nTOOL_NAME: read_file\n```",
            "```tool\nTOOL_NAME: read_file\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```",
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    parsed = asyncio.run(
        harness_provider.run_async(
            state=object(),
            selected_task_ids=["task-split-async-tool"],
            context=ToolContext(invocation_id="research-split-async-tool", metadata={}),
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=_prepared_research_interaction(),
        )
    )

    assert parsed.invocations[0].tool_name == "read_file"
    assert len(provider.async_messages) == 2
    assert (
        "The selected tool is fixed: read_file"
        in provider.async_messages[1][-1]["content"]
    )


def test_direct_research_provider_prompt_tools_category_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "category")
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_CATEGORY_THRESHOLD", "1")
    provider = _PromptToolProvider(
        [
            "```category\nMODE: category\nCATEGORY: filesystem\n```",
            (
                "```tool\n"
                "TOOL_NAME: search_text\n"
                "BEGIN_ARG: path\n.\nEND_ARG\n"
                "BEGIN_ARG: query\nneedle\nEND_ARG\n"
                "```"
            ),
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    parsed = harness_provider.run(
        state=object(),
        selected_task_ids=["task-category"],
        context=ToolContext(invocation_id="research-category", metadata={}),
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=_prepared_research_interaction_for_tools(
            ["read_file", "search_text", "run_git_status"]
        ),
    )

    assert parsed.invocations[0].tool_name == "search_text"
    assert parsed.invocations[0].arguments == {
        "path": ".",
        "query": "needle",
        "include_hidden": False,
    }
    assert "Available categories:" in provider.sync_messages[0][-1]["content"]
    assert (
        "Current tool category: filesystem" in provider.sync_messages[1][-1]["content"]
    )


def test_direct_research_provider_prompt_tools_category_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "category")
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_CATEGORY_THRESHOLD", "1")
    provider = _PromptToolProvider(
        [
            "```category\nMODE: finalize\n```",
            "```final\nANSWER:\nDone category.\n```",
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    parsed = harness_provider.run(
        state=object(),
        selected_task_ids=["task-category-final"],
        context=ToolContext(invocation_id="research-category-final", metadata={}),
        adapter=ActionEnvelopeAdapter(),
        prepared_interaction=_prepared_research_interaction_for_tools(
            ["read_file", "search_text", "run_git_status"]
        ),
    )

    assert parsed.final_response == "Done category."
    assert "```category" in provider.sync_messages[0][-1]["content"]
    assert "```final" in provider.sync_messages[1][-1]["content"]


def test_direct_research_provider_prompt_tools_category_async_flow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "category")
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_CATEGORY_THRESHOLD", "bad")
    assert (
        provider_module.AssistantHarnessTurnProvider._prompt_tool_category_threshold()
        == 7
    )
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_CATEGORY_THRESHOLD", "1")
    provider = _PromptToolProvider(
        [
            "```category\nMODE: category\nCATEGORY: filesystem\n```",
            ("```tool\nTOOL_NAME: read_file\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```"),
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    parsed = asyncio.run(
        harness_provider.run_async(
            state=object(),
            selected_task_ids=["task-category-async"],
            context=ToolContext(invocation_id="research-category-async", metadata={}),
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=_prepared_research_interaction_for_tools(
                ["read_file", "search_text", "run_git_status"]
            ),
        )
    )

    assert parsed.invocations[0].tool_name == "read_file"
    assert parsed.invocations[0].arguments["path"] == "README.md"
    assert len(provider.async_messages) == 2


def test_direct_research_provider_prompt_tools_category_async_finalize(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "category")
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_CATEGORY_THRESHOLD", "1")
    provider = _PromptToolProvider(
        [
            "```category\nMODE: finalize\n```",
            "```final\nANSWER:\nDone async category.\n```",
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    parsed = asyncio.run(
        harness_provider.run_async(
            state=object(),
            selected_task_ids=["task-category-async-final"],
            context=ToolContext(
                invocation_id="research-category-async-final",
                metadata={},
            ),
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=_prepared_research_interaction_for_tools(
                ["read_file", "search_text", "run_git_status"]
            ),
        )
    )

    assert parsed.final_response == "Done async category."
    assert "MODE is not a category name" in provider.async_messages[0][-1]["content"]


def test_direct_research_provider_prompt_tools_requires_text_transport() -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    class _MissingTextTransport:
        def prefers_simplified_json_schema_contract(self) -> bool:
            return False

        def uses_prompt_tool_protocol(self) -> bool:
            return True

    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=_MissingTextTransport(),  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    try:
        harness_provider.run(
            state=object(),
            selected_task_ids=["task-7"],
            context=ToolContext(invocation_id="research-prompt-missing", metadata={}),
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=_prepared_research_interaction(),
        )
    except RuntimeError as exc:
        assert "supports run_text" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected prompt-tool text transport error")


def test_direct_research_provider_prompt_tools_requires_async_text_transport() -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    class _MissingAsyncTextTransport:
        def prefers_simplified_json_schema_contract(self) -> bool:
            return False

        def uses_prompt_tool_protocol(self) -> bool:
            return True

        def run_text(self, **kwargs: object) -> str:
            del kwargs
            return "unused"

    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=_MissingAsyncTextTransport(),  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    try:
        asyncio.run(
            harness_provider.run_async(
                state=object(),
                selected_task_ids=["task-8"],
                context=ToolContext(
                    invocation_id="research-prompt-missing-async",
                    metadata={},
                ),
                adapter=ActionEnvelopeAdapter(),
                prepared_interaction=_prepared_research_interaction(),
            )
        )
    except RuntimeError as exc:
        assert "supports run_text_async" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected prompt-tool async text transport error")


def test_direct_research_provider_falls_back_to_prompt_tools_from_sync_modes() -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    for staged in [True, False]:
        provider = _FallbackPromptToolProvider(
            [
                "```decision\nMODE: finalize\n```",
                f"```final\nANSWER:\nFallback staged={staged}.\n```",
            ],
            staged=staged,
        )
        harness_provider = provider_module.AssistantHarnessTurnProvider(
            provider=provider,  # type: ignore[arg-type]
            temperature=0.1,
            system_prompt="research-system",
        )

        parsed = harness_provider.run(
            state=object(),
            selected_task_ids=["task-fallback"],
            context=ToolContext(invocation_id=f"fallback-{staged}", metadata={}),
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=_prepared_research_interaction(),
        )

        assert parsed.final_response == f"Fallback staged={staged}."
        assert len(provider.sync_messages) == 2
        assert provider.structured_calls == (1 if staged else 0)
        assert provider.run_calls == int(not staged)


def test_direct_research_provider_falls_back_to_prompt_tools_from_async_modes() -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    for staged in [True, False]:
        provider = _FallbackPromptToolProvider(
            [
                "```decision\nMODE: finalize\n```",
                f"```final\nANSWER:\nAsync fallback staged={staged}.\n```",
            ],
            staged=staged,
        )
        harness_provider = provider_module.AssistantHarnessTurnProvider(
            provider=provider,  # type: ignore[arg-type]
            temperature=0.1,
            system_prompt="research-system",
        )

        parsed = asyncio.run(
            harness_provider.run_async(
                state=object(),
                selected_task_ids=["task-fallback-async"],
                context=ToolContext(
                    invocation_id=f"fallback-async-{staged}",
                    metadata={},
                ),
                adapter=ActionEnvelopeAdapter(),
                prepared_interaction=_prepared_research_interaction(),
            )
        )

        assert parsed.final_response == f"Async fallback staged={staged}."
        assert len(provider.async_messages) == 2
        assert provider.async_structured_calls == (1 if staged else 0)
        assert provider.async_run_calls == int(not staged)


def test_direct_research_provider_rejects_unprepared_staged_and_prompt_tools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    staged_provider = _StagedProvider([{"mode": "tool", "tool_name": "read_file"}])
    prompt_provider = _PromptToolProvider(
        [
            "```decision\nMODE: tool\nTOOL_NAME: read_file\n```",
        ]
    )
    prepared = _prepared_research_interaction()
    prepared_without_input = replace(prepared, input_models={})

    staged_harness = provider_module.AssistantHarnessTurnProvider(
        provider=staged_provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )
    prompt_harness = provider_module.AssistantHarnessTurnProvider(
        provider=prompt_provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    try:
        staged_harness.run(
            state=object(),
            selected_task_ids=["task-staged-missing"],
            context=ToolContext(invocation_id="staged-missing", metadata={}),
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=prepared_without_input,
        )
    except ValueError as exc:
        assert "was not prepared" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected staged missing input model error")

    try:
        prompt_harness.run(
            state=object(),
            selected_task_ids=["task-prompt-missing"],
            context=ToolContext(invocation_id="prompt-missing", metadata={}),
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=prepared_without_input,
        )
    except ValueError as exc:
        assert "was not prepared" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected prompt-tool missing input model error")


def test_direct_research_provider_prompt_tools_raise_after_two_repairs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import llm_tools.apps.assistant_research_provider as provider_module

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    provider = _PromptToolProvider(
        [
            "not an action block",
            "still wrong",
            "wrong again",
        ]
    )
    harness_provider = provider_module.AssistantHarnessTurnProvider(
        provider=provider,  # type: ignore[arg-type]
        temperature=0.1,
        system_prompt="research-system",
    )

    try:
        harness_provider.run(
            state=object(),
            selected_task_ids=["task-prompt-fail"],
            context=ToolContext(invocation_id="prompt-fail", metadata={}),
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=_prepared_research_interaction(),
        )
    except ValueError as exc:
        assert "Missing fenced decision block" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected prompt-tool repair exhaustion")
