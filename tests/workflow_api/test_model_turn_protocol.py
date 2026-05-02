"""Tests for the shared workflow model-turn protocol runner."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import BaseModel

from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import ProtectionProvenanceSnapshot, ToolSpec
from llm_tools.workflow_api.executor import PreparedModelInteraction
from llm_tools.workflow_api.model_turn_protocol import (
    AsyncModelTurnProtocolProvider,
    AsyncStructuredModelTurnProtocolProvider,
    AsyncTextModelTurnProtocolProvider,
    ModelTurnJsonStrategyProvider,
    ModelTurnProtectionContext,
    ModelTurnProtocolEvent,
    ModelTurnProtocolPreferenceProvider,
    ModelTurnProtocolProvider,
    ModelTurnProtocolRequest,
    ModelTurnProtocolRunner,
    StructuredModelTurnProtocolProvider,
    TextModelTurnProtocolProvider,
)
from llm_tools.workflow_api.protection import (
    PromptProtectionDecision,
    ProtectionAction,
    ResponseProtectionDecision,
)


class _LookupInput(BaseModel):
    path: str


class _FinalAnswer(BaseModel):
    answer: str


class _ProtocolProvider:
    def __init__(
        self,
        *,
        native: list[object] | None = None,
        structured: list[object] | None = None,
        text: list[object] | None = None,
        staged: bool = False,
        prompt_tools: bool = False,
        fallback: bool = False,
        json_agent_strategy: str | None = None,
    ) -> None:
        self.native = list(native or [])
        self.structured = list(structured or [])
        self.text = list(text or [])
        self.staged = staged
        self.prompt_tools = prompt_tools
        self.fallback = fallback
        self.json_agent_strategy = json_agent_strategy
        self.native_messages: list[list[dict[str, object]]] = []
        self.structured_messages: list[list[dict[str, object]]] = []
        self.text_messages: list[list[dict[str, object]]] = []

    def run(self, **kwargs: object) -> ParsedModelResponse:
        self.native_messages.append(_copied_messages(kwargs))
        response = self.native.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, ParsedModelResponse)
        return response

    async def run_async(self, **kwargs: object) -> ParsedModelResponse:
        return self.run(**kwargs)

    def run_structured(self, **kwargs: object) -> object:
        self.structured_messages.append(_copied_messages(kwargs))
        response = self.structured.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    async def run_structured_async(self, **kwargs: object) -> object:
        return self.run_structured(**kwargs)

    def run_text(self, **kwargs: object) -> str:
        self.text_messages.append(_copied_messages(kwargs))
        response = self.text.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, str)
        return response

    async def run_text_async(self, **kwargs: object) -> str:
        return self.run_text(**kwargs)

    def uses_staged_schema_protocol(self) -> bool:
        return self.staged

    def uses_prompt_tool_protocol(self) -> bool:
        return self.prompt_tools

    def can_fallback_to_prompt_tools(self, exc: Exception) -> bool:
        del exc
        return self.fallback


class _MissingAsyncTextProvider(_ProtocolProvider):
    run_text_async = None


class _MissingSyncTextProvider(_ProtocolProvider):
    run_text = None


class _AsyncTextOnlyFinalProvider(_ProtocolProvider):
    run_structured_async = None


class _MissingAsyncFinalTransportProvider(_ProtocolProvider):
    run_structured_async = None
    run_text_async = None


class _AsyncPromptFallbackProvider(_ProtocolProvider):
    run_text = None

    async def run_text_async(self, **kwargs: object) -> str:
        self.text_messages.append(_copied_messages(kwargs))
        response = self.text.pop(0)
        if isinstance(response, Exception):
            raise response
        assert isinstance(response, str)
        return response


class _ProtectionController:
    def __init__(
        self,
        *,
        prompt: PromptProtectionDecision,
        response: ResponseProtectionDecision,
    ) -> None:
        self.prompt = prompt
        self.response = response

    def assess_prompt(self, **kwargs: object) -> PromptProtectionDecision:
        del kwargs
        return self.prompt

    async def assess_prompt_async(self, **kwargs: object) -> PromptProtectionDecision:
        return self.assess_prompt(**kwargs)

    def review_response(self, **kwargs: object) -> ResponseProtectionDecision:
        del kwargs
        return self.response

    async def review_response_async(
        self, **kwargs: object
    ) -> ResponseProtectionDecision:
        return self.review_response(**kwargs)

    def build_pending_prompt(self, **kwargs: object) -> object:
        return {"pending": True, "session_id": kwargs.get("session_id")}


def _copied_messages(kwargs: dict[str, object]) -> list[dict[str, object]]:
    messages = kwargs["messages"]
    assert isinstance(messages, list)
    return [dict(message) for message in messages]


def _parsed_event(events: list[ModelTurnProtocolEvent]) -> ModelTurnProtocolEvent:
    parsed_events = [event for event in events if event.kind == "parsed_response"]
    assert len(parsed_events) == 1
    return parsed_events[0]


def _prepared(adapter: ActionEnvelopeAdapter) -> PreparedModelInteraction:
    tool_specs = [
        ToolSpec(
            name="search_text",
            description="Search filesystem text.",
            tags=["filesystem"],
        ),
        ToolSpec(
            name="run_git_status",
            description="Read git status.",
            tags=["git"],
        ),
    ]
    input_models = {
        "search_text": _LookupInput,
        "run_git_status": _LookupInput,
    }
    response_model = adapter.build_response_model(
        tool_specs,
        input_models,
        final_response_model=_FinalAnswer,
    )
    return PreparedModelInteraction(
        response_model=response_model,
        schema=adapter.export_schema(response_model),
        tool_names=[spec.name for spec in tool_specs],
        tool_specs=tool_specs,
        input_models=input_models,
    )


def _request(
    provider: object,
    *,
    adapter: ActionEnvelopeAdapter | None = None,
    mode: str = "auto",
    protection: ModelTurnProtectionContext | None = None,
    use_json_single_action_strategy: bool | None = None,
) -> ModelTurnProtocolRequest:
    adapter = adapter or ActionEnvelopeAdapter()
    return ModelTurnProtocolRequest(
        provider=provider,
        messages=[{"role": "user", "content": "Find the answer."}],
        prepared_interaction=_prepared(adapter),
        adapter=adapter,
        final_response_model=_FinalAnswer,
        temperature=0,
        decision_context="Use one tool at most.",
        protection=protection,
        mode=mode,  # type: ignore[arg-type]
        use_json_single_action_strategy=use_json_single_action_strategy,
    )


def test_protocol_runs_native_sync_and_async() -> None:
    sync_provider = _ProtocolProvider(
        native=[ParsedModelResponse(final_response={"answer": "sync"})]
    )
    parsed = ModelTurnProtocolRunner().run(_request(sync_provider))
    assert parsed.final_response == {"answer": "sync"}
    assert (
        "Current native-tool step." in sync_provider.native_messages[0][-1]["content"]
    )

    async_provider = _ProtocolProvider(
        native=[ParsedModelResponse(final_response={"answer": "async"})]
    )
    parsed_async = asyncio.run(
        ModelTurnProtocolRunner().run_async(_request(async_provider))
    )
    assert parsed_async.final_response == {"answer": "async"}


def test_protocol_runs_native_forced_final_async_structured_and_text() -> None:
    structured_provider = _ProtocolProvider(
        structured=[{"answer": "structured async final"}]
    )
    parsed = asyncio.run(
        ModelTurnProtocolRunner().run_async(
            _request(structured_provider, mode="final_response")
        )
    )
    assert parsed.final_response == {"answer": "structured async final"}

    text_provider = _AsyncTextOnlyFinalProvider(
        structured=[ValueError("schema validation failed")],
        text=["text async final"],
    )
    parsed_text = asyncio.run(
        ModelTurnProtocolRunner().run_async(
            _request(text_provider, mode="final_response")
        )
    )
    assert parsed_text.final_response == {"answer": "text async final"}

    missing_transport = _MissingAsyncFinalTransportProvider(
        structured=[ValueError("schema validation failed")]
    )
    with pytest.raises(RuntimeError, match="run_text_async"):
        asyncio.run(
            ModelTurnProtocolRunner().run_async(
                _request(missing_transport, mode="final_response")
            )
        )


def test_protocol_runs_staged_split_single_action_final_and_async() -> None:
    adapter = ActionEnvelopeAdapter()
    split_provider = _ProtocolProvider(
        structured=[
            {"mode": "tool", "tool_name": "search_text"},
            {
                "mode": "tool",
                "tool_name": "search_text",
                "arguments": {"path": "README.md"},
            },
        ],
        staged=True,
    )
    split = ModelTurnProtocolRunner().run(
        _request(
            split_provider,
            adapter=adapter,
            use_json_single_action_strategy=False,
        )
    )
    assert split.invocations[0].tool_name == "search_text"

    single_provider = _ProtocolProvider(
        structured=[
            {
                "mode": "tool",
                "tool_name": "search_text",
                "arguments": {"path": "README.md"},
            }
        ],
        staged=True,
    )
    single = ModelTurnProtocolRunner().run(
        _request(single_provider, use_json_single_action_strategy=True)
    )
    assert single.invocations[0].arguments == {"path": "README.md"}

    final_provider = _ProtocolProvider(
        structured=[{"mode": "finalize", "final_response": {"answer": "done"}}],
        staged=True,
    )
    final = ModelTurnProtocolRunner().run(
        _request(final_provider, mode="final_response")
    )
    assert final.final_response == {"answer": "done"}

    async_provider = _ProtocolProvider(
        structured=[
            {"mode": "finalize"},
            {"mode": "finalize", "final_response": {"answer": "async"}},
        ],
        staged=True,
    )
    parsed_async = asyncio.run(
        ModelTurnProtocolRunner().run_async(
            _request(async_provider, use_json_single_action_strategy=False)
        )
    )
    assert parsed_async.final_response == {"answer": "async"}


def test_protocol_runs_staged_single_action_async() -> None:
    provider = _ProtocolProvider(
        structured=[
            {
                "mode": "tool",
                "tool_name": "search_text",
                "arguments": {"path": "README.md"},
            }
        ],
        staged=True,
    )

    parsed = asyncio.run(
        ModelTurnProtocolRunner().run_async(
            _request(provider, use_json_single_action_strategy=True)
        )
    )

    assert parsed.invocations[0].tool_name == "search_text"
    assert parsed.invocations[0].arguments == {"path": "README.md"}


def test_protocol_runs_prompt_tool_strategies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    split_provider = _ProtocolProvider(
        text=[
            "```decision\nMODE: tool\nTOOL_NAME: search_text\n```",
            "```tool\nTOOL_NAME: search_text\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```",
        ],
        prompt_tools=True,
    )
    split = ModelTurnProtocolRunner().run(_request(split_provider))
    assert split.invocations[0].tool_name == "search_text"

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "single_action")
    single_provider = _ProtocolProvider(
        text=[
            "```final\nANSWER:\nSingle action final.\n```",
        ],
        prompt_tools=True,
    )
    single = ModelTurnProtocolRunner().run(_request(single_provider))
    assert single.final_response == {"answer": "Single action final."}

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "category")
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_CATEGORY_THRESHOLD", "1")
    category_provider = _ProtocolProvider(
        text=[
            "```category\nMODE: category\nCATEGORY: filesystem\n```",
            "```final\nANSWER:\nCategory final.\n```",
        ],
        prompt_tools=True,
    )
    category = ModelTurnProtocolRunner().run(_request(category_provider))
    assert category.final_response == {"answer": "Category final."}


def test_protocol_runs_prompt_tool_async_strategies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    split_provider = _ProtocolProvider(
        text=[
            "```decision\nMODE: tool\nTOOL_NAME: search_text\n```",
            "```tool\nTOOL_NAME: search_text\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```",
        ],
        prompt_tools=True,
    )
    split = asyncio.run(ModelTurnProtocolRunner().run_async(_request(split_provider)))
    assert split.invocations[0].arguments == {"path": "README.md"}

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "single_action")
    single_provider = _ProtocolProvider(
        text=[
            "```tool\nTOOL_NAME: search_text\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```"
        ],
        prompt_tools=True,
    )
    single = asyncio.run(ModelTurnProtocolRunner().run_async(_request(single_provider)))
    assert single.invocations[0].tool_name == "search_text"

    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "category")
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_CATEGORY_THRESHOLD", "1")
    category_provider = _ProtocolProvider(
        text=[
            "```category\nMODE: category\nCATEGORY: filesystem\n```",
            "```tool\nTOOL_NAME: search_text\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```",
        ],
        prompt_tools=True,
    )
    category = asyncio.run(
        ModelTurnProtocolRunner().run_async(_request(category_provider))
    )
    assert category.invocations[0].arguments == {"path": "README.md"}


def test_protocol_reports_missing_sync_prompt_tool_transport() -> None:
    provider = _MissingSyncTextProvider(prompt_tools=True, text=["unused"])

    with pytest.raises(RuntimeError, match="run_text"):
        ModelTurnProtocolRunner().run(_request(provider))


def test_protocol_reports_missing_async_prompt_tool_transport() -> None:
    provider = _MissingAsyncTextProvider(prompt_tools=True, text=["unused"])

    with pytest.raises(RuntimeError, match="run_text_async"):
        asyncio.run(ModelTurnProtocolRunner().run_async(_request(provider)))


def test_protocol_repairs_and_exhausts_prompt_tool_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    provider = _ProtocolProvider(
        text=[
            "not protocol",
            "```decision\nMODE: finalize\n```",
            "```final\nANSWER:\nRepaired.\n```",
        ],
        prompt_tools=True,
    )
    events: list[ModelTurnProtocolEvent] = []
    parsed = ModelTurnProtocolRunner().run(_request(provider), observer=events.append)
    assert parsed.final_response == {"answer": "Repaired."}
    assert any(event.kind == "stage_repair" for event in events)

    failing = _ProtocolProvider(
        text=["bad", "still bad", "wrong again"],
        prompt_tools=True,
    )
    with pytest.raises(Exception, match="Missing fenced decision block"):
        ModelTurnProtocolRunner().run(_request(failing))


def test_protocol_does_not_repair_transport_failure() -> None:
    provider = _ProtocolProvider(
        structured=[RuntimeError("connection refused")],
        staged=True,
    )

    with pytest.raises(RuntimeError, match="connection refused"):
        ModelTurnProtocolRunner().run(
            _request(provider, use_json_single_action_strategy=False)
        )

    assert len(provider.structured_messages) == 1


def test_protocol_falls_back_and_skips_context_limit_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    provider = _ProtocolProvider(
        native=[RuntimeError("unsupported parameter: tools")],
        text=[
            "```decision\nMODE: finalize\n```",
            "```final\nANSWER:\nFallback.\n```",
        ],
        fallback=True,
    )
    events: list[ModelTurnProtocolEvent] = []
    parsed = ModelTurnProtocolRunner().run(_request(provider), observer=events.append)
    assert parsed.final_response == {"answer": "Fallback."}
    assert any(event.kind == "fallback" for event in events)

    context_limit = _ProtocolProvider(
        native=[RuntimeError("maximum context length exceeded")],
        text=["unused"],
        fallback=True,
    )
    with pytest.raises(RuntimeError, match="maximum context length"):
        ModelTurnProtocolRunner().run(_request(context_limit))
    assert context_limit.text_messages == []


def test_protocol_async_fallback_uses_async_text_transport(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    provider = _AsyncPromptFallbackProvider(
        native=[RuntimeError("unsupported parameter: tools")],
        text=[
            "```decision\nMODE: finalize\n```",
            "```final\nANSWER:\nAsync fallback.\n```",
        ],
        fallback=True,
    )

    parsed = asyncio.run(ModelTurnProtocolRunner().run_async(_request(provider)))

    assert parsed.final_response == {"answer": "Async fallback."}
    assert len(provider.text_messages) == 2


def test_protocol_parsed_response_events_use_actual_protocol(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "single_action")
    prompt_provider = _ProtocolProvider(
        text=["```final\nANSWER:\nPrompt.\n```"],
        prompt_tools=True,
    )
    prompt_events: list[ModelTurnProtocolEvent] = []
    ModelTurnProtocolRunner().run(
        _request(prompt_provider), observer=prompt_events.append
    )
    assert _parsed_event(prompt_events).protocol == "prompt_tools"

    staged_provider = _ProtocolProvider(
        structured=[
            {"mode": "finalize", "final_response": {"answer": "staged"}},
        ],
        staged=True,
    )
    staged_events: list[ModelTurnProtocolEvent] = []
    ModelTurnProtocolRunner().run(
        _request(staged_provider, use_json_single_action_strategy=True),
        observer=staged_events.append,
    )
    assert _parsed_event(staged_events).protocol == "staged"

    protection_provider = _ProtocolProvider(
        native=[ParsedModelResponse(final_response={"answer": "unused"})]
    )
    protection = ModelTurnProtectionContext(
        controller=_ProtectionController(
            prompt=PromptProtectionDecision(
                action=ProtectionAction.BLOCK,
                challenge_message="Blocked.",
            ),
            response=ResponseProtectionDecision(action=ProtectionAction.ALLOW),
        ),  # type: ignore[arg-type]
        provenance=ProtectionProvenanceSnapshot(),
    )
    protection_events: list[ModelTurnProtocolEvent] = []
    ModelTurnProtocolRunner().run(
        _request(protection_provider, protection=protection),
        observer=protection_events.append,
    )
    assert _parsed_event(protection_events).protocol == "protection"


def test_protocol_provider_protocol_surface_is_split_by_capability() -> None:
    assert callable(getattr(ModelTurnProtocolProvider, "run", None))
    optional_methods = {
        "run_async",
        "run_structured",
        "run_structured_async",
        "run_text",
        "run_text_async",
        "uses_staged_schema_protocol",
        "uses_prompt_tool_protocol",
        "can_fallback_to_prompt_tools",
        "json_agent_strategy",
    }
    for method_name in optional_methods:
        assert getattr(ModelTurnProtocolProvider, method_name, None) is None

    capability_methods = {
        AsyncModelTurnProtocolProvider: {"run_async"},
        StructuredModelTurnProtocolProvider: {"run_structured"},
        AsyncStructuredModelTurnProtocolProvider: {"run_structured_async"},
        TextModelTurnProtocolProvider: {"run_text"},
        AsyncTextModelTurnProtocolProvider: {"run_text_async"},
        ModelTurnProtocolPreferenceProvider: {
            "uses_staged_schema_protocol",
            "uses_prompt_tool_protocol",
            "can_fallback_to_prompt_tools",
        },
        ModelTurnJsonStrategyProvider: {"json_agent_strategy"},
    }

    for protocol, expected_methods in capability_methods.items():
        for method_name in expected_methods:
            assert callable(getattr(protocol, method_name, None))


def test_workflow_api_exports_model_turn_protocol_capability_types() -> None:
    import llm_tools.workflow_api as workflow_api

    expected_names = {
        "AsyncModelTurnProtocolProvider",
        "AsyncStructuredModelTurnProtocolProvider",
        "AsyncTextModelTurnProtocolProvider",
        "ModelTurnJsonStrategyProvider",
        "ModelTurnProtocolPreferenceProvider",
        "StructuredModelTurnProtocolProvider",
        "TextModelTurnProtocolProvider",
    }

    assert workflow_api.ModelTurnProtocolProvider is ModelTurnProtocolProvider
    for name in expected_names:
        assert getattr(workflow_api, name) is not None


def test_protocol_events_do_not_expose_raw_provider_payloads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "single_action")
    provider = _ProtocolProvider(
        text=["```final\nANSWER:\nsecret provider text\n```"],
        prompt_tools=True,
    )
    events: list[ModelTurnProtocolEvent] = []

    parsed = ModelTurnProtocolRunner().run(_request(provider), observer=events.append)

    assert parsed.final_response == {"answer": "secret provider text"}
    assert "secret provider text" not in str(events)
    assert any(event.kind == "parsed_response" for event in events)


def test_protocol_applies_prompt_and_response_protection() -> None:
    provider = _ProtocolProvider(
        native=[ParsedModelResponse(final_response={"answer": "raw"})]
    )
    metadata: dict[str, object] = {}
    protection = ModelTurnProtectionContext(
        controller=_ProtectionController(
            prompt=PromptProtectionDecision(
                action=ProtectionAction.CONSTRAIN,
                guard_text="Use only public facts.",
            ),
            response=ResponseProtectionDecision(
                action=ProtectionAction.SANITIZE,
                sanitized_payload={"answer": "clean"},
                should_purge=True,
            ),
        ),  # type: ignore[arg-type]
        provenance=ProtectionProvenanceSnapshot(),
        metadata_sink=metadata,
    )

    parsed = ModelTurnProtocolRunner().run(_request(provider, protection=protection))

    assert provider.native_messages[0][0]["content"] == "Use only public facts."
    assert parsed.final_response == {"answer": "clean"}
    assert metadata["protection_review"] == {
        "purge_requested": True,
        "safe_message": {"answer": "clean"},
    }


def test_protocol_blocks_prompt_before_provider_call() -> None:
    provider = _ProtocolProvider(
        native=[ParsedModelResponse(final_response={"answer": "unused"})]
    )
    metadata: dict[str, object] = {}
    protection = ModelTurnProtectionContext(
        controller=_ProtectionController(
            prompt=PromptProtectionDecision(
                action=ProtectionAction.BLOCK,
                challenge_message="Blocked.",
            ),
            response=ResponseProtectionDecision(action=ProtectionAction.ALLOW),
        ),  # type: ignore[arg-type]
        provenance=ProtectionProvenanceSnapshot(),
        metadata_sink=metadata,
    )

    parsed = ModelTurnProtocolRunner().run(_request(provider, protection=protection))

    assert parsed.final_response == {"answer": "Blocked."}
    assert provider.native_messages == []
    assert metadata["protection_review"] == {
        "purge_requested": True,
        "safe_message": "Blocked.",
    }


def test_protocol_applies_async_prompt_challenge_and_response_block() -> None:
    prompt_block_provider = _ProtocolProvider(
        native=[ParsedModelResponse(final_response={"answer": "unused"})]
    )
    prompt_metadata: dict[str, object] = {}
    prompt_protection = ModelTurnProtectionContext(
        controller=_ProtectionController(
            prompt=PromptProtectionDecision(
                action=ProtectionAction.CHALLENGE,
                challenge_message="Need review.",
            ),
            response=ResponseProtectionDecision(action=ProtectionAction.ALLOW),
        ),  # type: ignore[arg-type]
        provenance=ProtectionProvenanceSnapshot(),
        metadata_sink=prompt_metadata,
        original_user_message="sensitive prompt",
        session_id="session-1",
    )

    prompt_parsed = asyncio.run(
        ModelTurnProtocolRunner().run_async(
            _request(prompt_block_provider, protection=prompt_protection)
        )
    )

    assert prompt_parsed.final_response == {"answer": "Need review."}
    assert prompt_block_provider.native_messages == []
    assert prompt_metadata["pending_protection_prompt"] == {
        "pending": True,
        "session_id": "session-1",
    }

    response_block_provider = _ProtocolProvider(
        native=[ParsedModelResponse(final_response={"answer": "raw"})]
    )
    response_metadata: dict[str, object] = {}
    response_protection = ModelTurnProtectionContext(
        controller=_ProtectionController(
            prompt=PromptProtectionDecision(action=ProtectionAction.ALLOW),
            response=ResponseProtectionDecision(
                action=ProtectionAction.BLOCK,
                safe_message="Withheld.",
                should_purge=True,
            ),
        ),  # type: ignore[arg-type]
        provenance=ProtectionProvenanceSnapshot(),
        metadata_sink=response_metadata,
    )

    response_parsed = asyncio.run(
        ModelTurnProtocolRunner().run_async(
            _request(response_block_provider, protection=response_protection)
        )
    )

    assert response_parsed.final_response == {"answer": "Withheld."}
    assert response_metadata["protection_review"] == {
        "purge_requested": True,
        "safe_message": "Withheld.",
    }
