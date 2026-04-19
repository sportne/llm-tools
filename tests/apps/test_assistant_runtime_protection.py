"""Additional assistant runtime coverage for protection-aware branches."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from llm_tools.apps.assistant_config import StreamlitAssistantConfig
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

    config = StreamlitAssistantConfig(
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
        config=StreamlitAssistantConfig(
            protection=ProtectionConfig(
                enabled=True, document_paths=[str(tmp_path / "policy.txt")]
            )
        ),
        provider_config=StreamlitAssistantConfig().llm,
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
