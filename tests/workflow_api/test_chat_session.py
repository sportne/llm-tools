"""Tests for interactive repository chat session orchestration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.llm_adapters.prompt_tools import PromptToolProtocolError
from llm_tools.tool_api import (
    ErrorCode,
    SideEffectClass,
    ToolContext,
    ToolError,
    ToolInvocationRequest,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
)
from llm_tools.tool_api.redaction import RedactionConfig
from llm_tools.tools import register_filesystem_tools, register_text_tools
from llm_tools.tools.filesystem import ToolLimits
from llm_tools.workflow_api import (
    ChatContextSummary,
    ChatFinalResponse,
    ChatMessage,
    ChatSessionConfig,
    ChatSessionState,
    ChatSessionTurnRecord,
    ChatWorkflowApprovalEvent,
    ChatWorkflowApprovalResolvedEvent,
    ChatWorkflowInspectorEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ProtectionAction,
    ProtectionAssessment,
    ProtectionConfig,
    ProtectionController,
    WorkflowExecutor,
    run_interactive_chat_session_turn,
)
from llm_tools.workflow_api.chat_runner import ChatSessionTurnRunner
from llm_tools.workflow_api.models import (
    ApprovalRequest,
    WorkflowInvocationOutcome,
    WorkflowInvocationStatus,
    WorkflowTurnResult,
)


class _FakeProvider:
    def __init__(self, responses: list[ParsedModelResponse]) -> None:
        self._responses = list(responses)
        self.calls = 0

    def run(self, **kwargs) -> ParsedModelResponse:
        del kwargs
        self.calls += 1
        return self._responses.pop(0)


class _FakeNativeLimitProvider:
    def __init__(self, first_response: ParsedModelResponse, final_answer: str) -> None:
        self._first_response = first_response
        self._final_answer = final_answer
        self.run_messages: list[list[dict[str, object]]] = []
        self.structured_messages: list[list[dict[str, object]]] = []
        self.response_model_names: list[str] = []

    def run(self, **kwargs) -> ParsedModelResponse:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.run_messages.append([dict(message) for message in messages])
        return self._first_response

    def run_structured(self, **kwargs) -> ChatFinalResponse:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.structured_messages.append([dict(message) for message in messages])
        self.response_model_names.append(kwargs["response_model"].__name__)
        return ChatFinalResponse(answer=self._final_answer)


class _FakeNativeTextFinalProvider(_FakeNativeLimitProvider):
    def __init__(self, first_response: ParsedModelResponse, final_answer: str) -> None:
        super().__init__(first_response, final_answer)
        self.text_messages: list[list[dict[str, object]]] = []

    def run_structured(self, **kwargs) -> object:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.structured_messages.append([dict(message) for message in messages])
        self.response_model_names.append(kwargs["response_model"].__name__)
        raise ValueError("invalid json")

    def run_text(self, **kwargs) -> str:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.text_messages.append([dict(message) for message in messages])
        return self._final_answer


class _FakeStagedProvider:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, object]]] = []
        self.response_model_names: list[str] = []
        self.json_agent_strategy = "staged"

    def run(self, **kwargs) -> ParsedModelResponse:
        del kwargs
        raise AssertionError("staged provider should use run_structured()")

    def run_structured(self, **kwargs) -> object:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.calls.append([dict(message) for message in messages])
        self.response_model_names.append(kwargs["response_model"].__name__)
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def uses_staged_schema_protocol(self) -> bool:
        return True


class _FakeSingleActionStagedProvider(_FakeStagedProvider):
    def __init__(self, responses: list[object]) -> None:
        super().__init__(responses)
        self.json_agent_strategy = "single_action"


class _FakePromptToolProvider:
    def __init__(self, responses: list[str | Exception]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, object]]] = []

    def run(self, **kwargs) -> ParsedModelResponse:
        del kwargs
        raise AssertionError("prompt-tool provider should use run_text()")

    def run_structured(self, **kwargs) -> object:
        del kwargs
        raise AssertionError("prompt-tool provider should use run_text()")

    def run_text(self, **kwargs) -> str:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.calls.append([dict(message) for message in messages])
        response = self._responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    def uses_prompt_tool_protocol(self) -> bool:
        return True


class _FallbackPromptToolProvider(_FakePromptToolProvider):
    def __init__(self, responses: list[str], *, failure: Exception) -> None:
        super().__init__(responses)
        self.failure = failure
        self.native_calls = 0

    def run(self, **kwargs) -> ParsedModelResponse:
        del kwargs
        self.native_calls += 1
        raise self.failure

    def uses_prompt_tool_protocol(self) -> bool:
        return False

    def can_fallback_to_prompt_tools(self, exc: Exception) -> bool:
        return exc is self.failure


class _FallbackStagedPromptToolProvider(_FakePromptToolProvider):
    def __init__(self, responses: list[str], *, failure: Exception) -> None:
        super().__init__(responses)
        self.failure = failure
        self.structured_calls = 0

    def run_structured(self, **kwargs) -> object:
        del kwargs
        self.structured_calls += 1
        raise self.failure

    def uses_prompt_tool_protocol(self) -> bool:
        return False

    def uses_staged_schema_protocol(self) -> bool:
        return True

    def can_fallback_to_prompt_tools(self, exc: Exception) -> bool:
        return exc is self.failure


class _FailingNativeProvider:
    def run(self, **kwargs) -> ParsedModelResponse:
        del kwargs
        raise RuntimeError("auth failed")


class _CancelOnRunProvider:
    def __init__(self, response: ParsedModelResponse) -> None:
        self._response = response
        self.runner = None

    def run(self, **kwargs) -> ParsedModelResponse:
        del kwargs
        assert self.runner is not None
        self.runner.cancel()
        return self._response


class _CallableJsonStrategyProvider(_FakeStagedProvider):
    def json_agent_strategy(self) -> str:
        return "decision"


def _executor() -> WorkflowExecutor:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)
    return WorkflowExecutor(
        registry=registry,
        policy=ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ}
        ),
    )


def _empty_executor() -> WorkflowExecutor:
    return WorkflowExecutor(
        registry=ToolRegistry(),
        policy=ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ}
        ),
    )


def _context(tmp_path: Path) -> ToolContext:
    return ToolContext(
        invocation_id="chat-session",
        workspace=str(tmp_path),
        metadata={
            "source_filters": {"include_hidden": False},
            "tool_limits": ToolLimits(max_tool_result_chars=500).model_dump(
                mode="json"
            ),
        },
    )


def _approval_request(
    request: ToolInvocationRequest, *, expires_at: str
) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id="approval-1",
        invocation_index=1,
        request=request,
        tool_name=request.tool_name,
        tool_version="0.1.0",
        policy_reason="approval required",
        policy_metadata={},
        requested_at="2026-01-01T00:00:00Z",
        expires_at=expires_at,
    )


def test_chat_session_runner_summarizes_model_visible_assistant_history(
    tmp_path: Path,
) -> None:
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=_FakePromptToolProvider(["```final\nANSWER:\nDone.\n```"]),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )
    tool_message = ChatMessage(
        role="assistant",
        content=(
            '{"actions": ['
            '{"tool_name": "read_file", "arguments": {"path": "README.md"}},'
            '{"tool_name": 3, "arguments": []}'
            "]}"
        ),
    )
    final_message = ChatMessage(
        role="assistant",
        content='{"final_response": {"answer": "Done."}}',
    )

    serialized = runner._serialize_messages_for_model([tool_message, final_message])

    assert serialized[0]["content"] == (
        "Tool call audit metadata, not evidence and not an answer: "
        'read_file({"path":"README.md"}).'
    )
    assert serialized[1]["content"] == "Assistant final answer: Done."
    assert ChatSessionTurnRunner._assistant_message_summary("not json") is None
    assert ChatSessionTurnRunner._assistant_message_summary("[1, 2]") is None
    assert (
        ChatSessionTurnRunner._assistant_message_summary(
            '{"actions": [{"arguments": {}}]}'
        )
        is None
    )
    assert (
        ChatSessionTurnRunner._assistant_message_summary(
            '{"final_response": "Plain final."}'
        )
        == "Assistant final answer: Plain final."
    )


def test_chat_session_runner_compacts_old_context_before_provider_call(
    tmp_path: Path,
) -> None:
    old_turn = ChatSessionTurnRecord(
        status="completed",
        new_messages=[
            ChatMessage(role="user", content="old question " * 30),
            ChatMessage(role="assistant", content="old answer " * 30),
        ],
        final_response=ChatFinalResponse(answer="old answer"),
    )
    provider = _FakePromptToolProvider(
        ["durable summary of old context", "```final\nANSWER:\nDone.\n```"]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(turns=[old_turn]),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(max_context_tokens=40),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)
    result = next(
        event for event in events if isinstance(event, ChatWorkflowResultEvent)
    )

    assert result.result.context_warning is not None
    assert result.result.session_state is not None
    assert result.result.session_state.context_summary is not None
    assert result.result.session_state.context_summary.content == (
        "durable summary of old context"
    )
    assert result.result.session_state.context_summary.covered_turn_count == 1
    final_call_messages = provider.calls[-1]
    assert any(
        "Earlier conversation summary:\ndurable summary of old context"
        in str(message["content"])
        for message in final_call_messages
    )


def test_chat_session_runner_retries_context_limit_after_compaction(
    tmp_path: Path,
) -> None:
    old_turn = ChatSessionTurnRecord(
        status="completed",
        new_messages=[
            ChatMessage(role="user", content="remember alpha"),
            ChatMessage(role="assistant", content="alpha is remembered"),
        ],
        final_response=ChatFinalResponse(answer="alpha is remembered"),
    )
    provider = _FakePromptToolProvider(
        [
            RuntimeError("maximum context length exceeded"),
            "summary after context error",
            "```final\nANSWER:\nDone after retry.\n```",
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(turns=[old_turn]),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(max_context_tokens=1000),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)
    result = next(
        event for event in events if isinstance(event, ChatWorkflowResultEvent)
    )

    assert len(provider.calls) == 3
    assert result.result.final_response is not None
    assert result.result.final_response.answer == "Done after retry."
    assert result.result.session_state is not None
    assert result.result.session_state.context_summary is not None
    assert result.result.session_state.context_summary.content == (
        "summary after context error"
    )


def test_chat_session_runner_does_not_retry_repeated_context_limit(
    tmp_path: Path,
) -> None:
    old_turn = ChatSessionTurnRecord(
        status="completed",
        new_messages=[
            ChatMessage(role="user", content="remember alpha"),
            ChatMessage(role="assistant", content="alpha is remembered"),
        ],
        final_response=ChatFinalResponse(answer="alpha is remembered"),
    )
    provider = _FakePromptToolProvider(
        [
            RuntimeError("maximum context length exceeded"),
            "summary after context error",
            RuntimeError("maximum context length exceeded"),
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(turns=[old_turn]),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(max_context_tokens=1000),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    with pytest.raises(RuntimeError, match="maximum context length"):
        list(runner)


def test_chat_session_runner_uses_deterministic_context_summary_fallback(
    tmp_path: Path,
) -> None:
    old_turn = ChatSessionTurnRecord(
        status="completed",
        new_messages=[
            ChatMessage(role="user", content="old question"),
            ChatMessage(
                role="assistant",
                content=(
                    '{"actions": [{"tool_name": "read_file", '
                    '"arguments": {"path": "README.md"}}]}'
                ),
            ),
            ChatMessage(role="tool", content='{"content": "tool evidence"}'),
        ],
        final_response=ChatFinalResponse(answer="old answer"),
        tool_results=[
            ToolResult(
                ok=True,
                tool_name="read_file",
                tool_version="0.1.0",
                output={"content": "tool evidence"},
                metadata={
                    "execution_record": {
                        "request": {
                            "tool_name": "read_file",
                            "arguments": {"path": "README.md"},
                        }
                    }
                },
            )
        ],
    )
    provider = _FakePromptToolProvider(
        [RuntimeError("summary failed"), "```final\nANSWER:\nDone.\n```"]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(turns=[old_turn]),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(max_context_tokens=20),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    result = next(
        event for event in list(runner) if isinstance(event, ChatWorkflowResultEvent)
    )

    assert result.result.session_state is not None
    summary = result.result.session_state.context_summary
    assert summary is not None
    assert "Turn 1 (completed):" in summary.content
    assert "Tool result:" in summary.content
    assert 'read_file({"path":"README.md"}) -> success' in summary.content


def test_chat_session_runner_context_summary_helpers_cover_edges(
    tmp_path: Path,
) -> None:
    previous = ChatContextSummary(
        content="prior compact memory",
        covered_turn_count=2,
        created_at="2026-01-01T00:00:00Z",
        updated_at="2026-01-01T00:00:00Z",
        compaction_count=1,
    )
    turn = ChatSessionTurnRecord(
        status="completed",
        new_messages=[ChatMessage(role="assistant", content='{"actions": [3]}')],
        final_response=ChatFinalResponse(answer="new answer"),
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=_FakePromptToolProvider(["```final\nANSWER:\nDone.\n```"]),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    prompt_messages = runner._context_summary_prompt_messages(
        previous_summary=previous,
        turns=[turn],
        starting_turn_number=3,
    )
    fallback = runner._fallback_context_summary(
        previous_summary=previous,
        turns=[turn],
        starting_turn_number=3,
    )

    assert "prior compact memory" in prompt_messages[1]["content"]
    assert "Previous summary:" in fallback
    assert "new answer" in fallback
    assert runner._compact_text("x" * 705).endswith("...(truncated)")
    assert runner._truncate_summary("x" * 12005).endswith("...(truncated)")
    assert runner._is_context_limit_error(
        RuntimeError("Request failed: too many tokens")
    )
    assert not runner._is_context_limit_error(RuntimeError("auth failed"))


def test_chat_session_runner_protects_context_summary(tmp_path: Path) -> None:
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=_FakePromptToolProvider(["```final\nANSWER:\nDone.\n```"]),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    runner._protection_controller = SimpleNamespace(
        review_response=lambda **_kwargs: SimpleNamespace(
            action=ProtectionAction.SANITIZE,
            sanitized_payload={"answer": "sanitized summary"},
        )
    )
    assert runner._protect_context_summary("raw summary") == "sanitized summary"

    runner._protection_controller = SimpleNamespace(
        review_response=lambda **_kwargs: SimpleNamespace(
            action=ProtectionAction.BLOCK,
            safe_message=None,
            sanitized_payload=None,
        )
    )
    assert runner._protect_context_summary("raw summary") == (
        "Earlier context summary was withheld for this environment."
    )


def test_chat_session_runner_strategy_helpers_use_env_defaults(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=_FakeStagedProvider([]),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )
    delattr(runner._provider, "json_agent_strategy")

    monkeypatch.delenv("LLM_TOOLS_JSON_AGENT_STRATEGY", raising=False)
    assert runner._uses_json_single_action_strategy() is True
    monkeypatch.setenv("LLM_TOOLS_JSON_AGENT_STRATEGY", "decision")
    assert runner._uses_json_single_action_strategy() is False
    monkeypatch.delenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", raising=False)
    assert runner._uses_prompt_tool_single_action_strategy() is True
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    assert runner._uses_prompt_tool_single_action_strategy() is False
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "category")
    assert runner._uses_prompt_tool_single_action_strategy() is False
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_CATEGORY_THRESHOLD", "bad")
    assert ChatSessionTurnRunner._prompt_tool_category_threshold() == 7


def test_chat_session_runner_helper_error_branches(tmp_path: Path) -> None:
    native_runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=_FakeProvider([]),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )
    assistant_message = ChatMessage(
        role="assistant",
        content='{"final_response": ""}',
    )

    assert native_runner._serialize_chat_message_for_protocol(
        assistant_message,
        protocol="unknown",
    ) == {"role": "assistant", "content": '{"final_response": ""}'}
    assert native_runner._serialize_chat_message_for_protocol(
        ChatMessage(role="assistant", content="{}"),
        protocol="native_tools",
    ) == {"role": "assistant", "content": "{}"}
    with pytest.raises(RuntimeError, match="run_structured"):
        native_runner._structured_provider()

    prompt_step = native_runner._run_prompt_tool_step(
        round_index=1,
        stage_name="decision",
        messages=[],
        parser=lambda text: text,
        repair_context={},
    )
    with pytest.raises(RuntimeError, match="run_text"):
        next(prompt_step)

    staged_runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=_CallableJsonStrategyProvider([]),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )
    assert staged_runner._uses_json_single_action_strategy() is False

    forced_final = native_runner._force_final_response_at_limit(
        round_index=1,
        messages=[ChatMessage(role="user", content="Answer plainly.")],
        new_messages=[],
        tool_results=[],
    )
    with pytest.raises(StopIteration) as exc_info:
        next(forced_final)
    assert exc_info.value.value is None


def test_chat_session_runner_executes_tool_then_returns_final_response(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    runner = run_interactive_chat_session_turn(
        user_message="Where is needle defined?",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        {
                            "tool_name": "search_text",
                            "arguments": {"path": ".", "query": "needle"},
                        }
                    ]
                ),
                ParsedModelResponse(
                    final_response=ChatFinalResponse(
                        answer="It is defined in src/app.py.",
                        citations=[{"source_path": "src/app.py", "line_start": 1}],
                    ).model_dump(mode="json")
                ),
            ]
        ),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)
    assert isinstance(events[0], ChatWorkflowStatusEvent)
    assert any(isinstance(event, ChatWorkflowInspectorEvent) for event in events)
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "completed"
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "It is defined in src/app.py."
    assert result_event.result.session_state is not None
    assert len(result_event.result.session_state.turns) == 1


def test_chat_session_runner_executes_prompt_tool_rounds(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    provider = _FakePromptToolProvider(
        [
            "```decision\nMODE: tool\nTOOL_NAME: search_text\n```",
            (
                "```tool\n"
                "TOOL_NAME: search_text\n"
                "BEGIN_ARG: path\n.\nEND_ARG\n"
                "BEGIN_ARG: query\nneedle\nEND_ARG\n"
                "```"
            ),
            "```decision\nMODE: finalize\n```",
            "```final\nANSWER:\nIt is defined in src/app.py.\n```",
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Where is needle defined?",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)

    assert len(provider.calls) == 4
    assistant_history = [
        message["content"]
        for message in provider.calls[2]
        if message.get("role") == "assistant"
    ]
    assert assistant_history
    assert all('"actions"' not in str(content) for content in assistant_history)
    assert any(
        "Tool call audit metadata, not evidence and not an answer: search_text"
        in str(content)
        for content in assistant_history
    )
    assert any(
        isinstance(event, ChatWorkflowStatusEvent) and event.status == "searching text"
        for event in events
    )
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "completed"
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "It is defined in src/app.py."


def test_chat_session_runner_prompt_tool_path_honors_approval_gates(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)
    executor = WorkflowExecutor(
        registry=registry,
        policy=ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
            require_approval_for={SideEffectClass.LOCAL_READ},
        ),
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    provider = _FakePromptToolProvider(
        [
            "```decision\nMODE: tool\nTOOL_NAME: search_text\n```",
            (
                "```tool\n"
                "TOOL_NAME: search_text\n"
                "BEGIN_ARG: path\n.\nEND_ARG\n"
                "BEGIN_ARG: query\nneedle\nEND_ARG\n"
                "```"
            ),
            "```decision\nMODE: finalize\n```",
            "```final\nANSWER:\nIt is defined in src/app.py.\n```",
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Where is needle defined?",
        session_state=ChatSessionState(),
        executor=executor,
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    iterator = iter(runner)
    approval_event = next(
        event for event in iterator if isinstance(event, ChatWorkflowApprovalEvent)
    )
    assert approval_event.approval.tool_name == "search_text"
    assert runner.resolve_pending_approval(True) is True
    remaining = list(iterator)

    assert any(
        isinstance(event, ChatWorkflowApprovalResolvedEvent)
        and event.resolution == "approved"
        for event in remaining
    )
    result_event = remaining[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "completed"
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "It is defined in src/app.py."


def test_chat_session_runner_prompt_tool_category_strategy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "category")
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_CATEGORY_THRESHOLD", "1")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    provider = _FakePromptToolProvider(
        [
            "```category\nMODE: category\nCATEGORY: text\n```",
            (
                "```tool\n"
                "TOOL_NAME: search_text\n"
                "BEGIN_ARG: path\n.\nEND_ARG\n"
                "BEGIN_ARG: query\nneedle\nEND_ARG\n"
                "```"
            ),
            "```category\nMODE: finalize\n```",
            "```final\nANSWER:\nIt is defined in src/app.py.\n```",
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Where is needle defined?",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)

    assert len(provider.calls) == 4
    assert "Available categories:" in provider.calls[0][-1]["content"]
    assert "Current tool category: text" in provider.calls[1][-1]["content"]
    assert any(
        isinstance(event, ChatWorkflowStatusEvent)
        and event.status == "using text tools"
        for event in events
    )
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "It is defined in src/app.py."


def test_chat_session_runner_prompt_tool_does_not_repair_transport_failure(
    tmp_path: Path,
) -> None:
    provider = _FakePromptToolProvider([RuntimeError("connection refused")])
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    with pytest.raises(RuntimeError, match="connection refused"):
        list(runner)

    assert len(provider.calls) == 1


def test_chat_session_runner_repairs_prompt_tool_final_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    provider = _FakePromptToolProvider(
        [
            "```decision\nMODE: finalize\n```",
            "```final\nANSWER:\n```",
            "```final\nANSWER:\nDone after repair.\n```",
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)

    assert len(provider.calls) == 3
    assert (
        "previous final_response response was invalid"
        in provider.calls[-1][-1]["content"]
    )
    assert any(
        isinstance(event, ChatWorkflowStatusEvent)
        and event.status == "repairing final_response"
        for event in events
    )
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "Done after repair."


def test_chat_session_runner_repairs_prompt_tool_audit_metadata_final_response(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    provider = _FakePromptToolProvider(
        [
            "```decision\nMODE: finalize\n```",
            (
                "```final\nANSWER:\n"
                "Tool call audit metadata, not evidence and not an answer: "
                'read_file({"path":"README.md"}).\n```'
            ),
            "```final\nANSWER:\nThe README explains the project.\n```",
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer from the gathered evidence.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)

    assert len(provider.calls) == 3
    repair_message = provider.calls[-1][-1]["content"]
    assert isinstance(repair_message, str)
    assert "audit metadata" in repair_message
    assert "tool result content as evidence" in repair_message
    assert any(
        isinstance(event, ChatWorkflowStatusEvent)
        and event.status == "repairing final_response"
        for event in events
    )
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert (
        result_event.result.final_response.answer == "The README explains the project."
    )


def test_chat_session_runner_fails_prompt_tool_stage_after_two_repairs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LLM_TOOLS_PROMPT_TOOL_STRATEGY", "split")
    provider = _FakePromptToolProvider(
        [
            "```decision\nMODE: finalize\n```",
            "```final\nANSWER:\n```",
            "```final\nANSWER:\n```",
            "```final\nANSWER:\n```",
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    with pytest.raises(PromptToolProtocolError):
        list(runner)

    assert len(provider.calls) == 4


def test_chat_session_runner_auto_falls_back_to_prompt_tools(
    tmp_path: Path,
) -> None:
    failure = RuntimeError("unsupported parameter: tools")
    provider = _FallbackPromptToolProvider(
        [
            "```decision\nMODE: finalize\n```",
            "```final\nANSWER:\nFallback answer.\n```",
        ],
        failure=failure,
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)

    assert provider.native_calls == 1
    assert len(provider.calls) == 2
    assert any(
        isinstance(event, ChatWorkflowStatusEvent)
        and event.status == "using prompt tools"
        for event in events
    )
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "Fallback answer."


def test_chat_session_runner_does_not_fallback_for_uncategorized_native_failure(
    tmp_path: Path,
) -> None:
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=_FailingNativeProvider(),  # type: ignore[arg-type]
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    with pytest.raises(RuntimeError, match="auth failed"):
        list(runner)


def test_chat_session_runner_staged_failure_falls_back_to_prompt_tools(
    tmp_path: Path,
) -> None:
    failure = ValueError("invalid json schema response")
    provider = _FallbackStagedPromptToolProvider(
        [
            "```decision\nMODE: finalize\n```",
            "```final\nANSWER:\nFallback answer.\n```",
        ],
        failure=failure,
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)

    assert provider.structured_calls == 3
    assert len(provider.calls) == 2
    assert any(
        isinstance(event, ChatWorkflowStatusEvent)
        and event.status == "using prompt tools"
        for event in events
    )
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "Fallback answer."


def test_chat_session_runner_executes_json_single_action_rounds(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    provider = _FakeSingleActionStagedProvider(
        [
            {
                "mode": "tool",
                "tool_name": "search_text",
                "arguments": {"path": ".", "query": "needle"},
            },
            {
                "mode": "finalize",
                "final_response": {
                    "answer": "It is defined in src/app.py.",
                    "citations": [{"source_path": "src/app.py", "line_start": 1}],
                },
            },
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Where is needle defined?",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)

    assert provider.response_model_names == ["SingleActionStep", "SingleActionStep"]
    assert (
        "Current step: choose exactly one next action."
        in provider.calls[0][-1]["content"]
    )
    assert any(
        isinstance(event, ChatWorkflowStatusEvent)
        and event.status == "preparing search_text"
        for event in events
    )
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "It is defined in src/app.py."


def test_chat_session_runner_retries_ungrounded_repo_answer_with_tools(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    provider = _FakeProvider(
        [
            ParsedModelResponse(
                final_response=ChatFinalResponse(
                    answer="The app probably wires state through the chat controller."
                ).model_dump(mode="json")
            ),
            ParsedModelResponse(
                invocations=[
                    {
                        "tool_name": "search_text",
                        "arguments": {"path": ".", "query": "session_state"},
                    }
                ]
            ),
            ParsedModelResponse(
                final_response=ChatFinalResponse(
                    answer="It wires state through the chat controller.",
                    citations=[{"source_path": "src/app.py", "line_start": 1}],
                ).model_dump(mode="json")
            ),
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message=(
            "Explain how this repository wires chat state together. "
            "You must use local workspace tools before answering."
        ),
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
        enabled_tool_names={"search_text", "read_file", "find_files"},
    )

    events = list(runner)

    assert provider.calls == 3
    assert any(
        isinstance(event, ChatWorkflowStatusEvent)
        and event.status == "gathering evidence"
        for event in events
    )
    assert any(
        isinstance(event, ChatWorkflowStatusEvent) and event.status == "searching text"
        for event in events
    )
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert (
        result_event.result.final_response.answer
        == "It wires state through the chat controller."
    )


def test_chat_session_runner_executes_staged_tool_then_returns_final_response(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    provider = _FakeStagedProvider(
        [
            {"mode": "tool", "tool_name": "search_text"},
            {
                "mode": "tool",
                "tool_name": "search_text",
                "arguments": {"path": ".", "query": "needle"},
            },
            {"mode": "finalize"},
            {
                "mode": "finalize",
                "final_response": {
                    "answer": "It is defined in src/app.py.",
                    "citations": [{"source_path": "src/app.py", "line_start": 1}],
                },
            },
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Where is needle defined?",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(max_tool_result_chars=500),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)

    assert any(
        isinstance(event, ChatWorkflowStatusEvent)
        and event.status == "preparing search_text"
        for event in events
    )
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "It is defined in src/app.py."
    assert len(provider.calls) == 4
    assert provider.response_model_names == [
        "DecisionStep",
        "SearchTextInvocationStep",
        "DecisionStep",
        "FinalResponseStep",
    ]
    assert "Current step: choose the next action." in provider.calls[0][-1]["content"]
    assert "invoke the selected tool 'search_text'" in provider.calls[1][-1]["content"]
    assert "Current step: finalize the answer." in provider.calls[3][-1]["content"]


def test_chat_session_runner_repairs_invalid_staged_final_response(
    tmp_path: Path,
) -> None:
    provider = _FakeStagedProvider(
        [
            {"mode": "finalize"},
            {
                "mode": "finalize",
                "final_response": {
                    "answer": "done",
                    "citations": ["README.md"],
                },
            },
            {
                "mode": "finalize",
                "final_response": {
                    "answer": "done",
                    "citations": [{"source_path": "README.md", "line_start": 1}],
                },
            },
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer with a citation.",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)

    assert any(
        isinstance(event, ChatWorkflowStatusEvent)
        and event.status == "repairing final_response"
        for event in events
    )
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "done"
    assert len(provider.calls) == 3
    repair_message = provider.calls[-1][-1]["content"]
    assert isinstance(repair_message, str)
    assert "The previous final_response response was invalid." in repair_message
    assert "Validation summary:" in repair_message
    assert '"answer": "done"' in repair_message


def test_chat_session_runner_with_no_tools_repairs_invalid_tool_decision(
    tmp_path: Path,
) -> None:
    provider = _FakeStagedProvider(
        [
            {"mode": "tool", "tool_name": "missing"},
            {"mode": "finalize"},
            {
                "mode": "finalize",
                "final_response": {
                    "answer": "done",
                    "citations": [],
                },
            },
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer without tools.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)

    assert any(
        isinstance(event, ChatWorkflowStatusEvent)
        and event.status == "repairing decision"
        for event in events
    )
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "done"
    repair_message = provider.calls[1][-1]["content"]
    assert isinstance(repair_message, str)
    assert "The previous decision response was invalid." in repair_message
    assert '"const": "finalize"' in repair_message


def test_chat_session_runner_fails_staged_stage_after_two_repairs(
    tmp_path: Path,
) -> None:
    provider = _FakeStagedProvider(
        [
            {"mode": "finalize"},
            {"mode": "finalize", "final_response": {"answer": ""}},
            {"mode": "finalize", "final_response": {"answer": ""}},
            {"mode": "finalize", "final_response": {"answer": ""}},
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    with pytest.raises(ValueError, match="Invalid staged final-response payload"):
        list(runner)

    assert len(provider.calls) == 4
    assert provider.response_model_names == [
        "DecisionStep",
        "FinalResponseStep",
        "FinalResponseStep",
        "FinalResponseStep",
    ]
    assert "The previous final_response response was invalid." in str(
        provider.calls[2][-1]["content"]
    )
    assert "The previous final_response response was invalid." in str(
        provider.calls[3][-1]["content"]
    )


def test_chat_session_runner_does_not_repair_staged_transport_failure(
    tmp_path: Path,
) -> None:
    provider = _FakeStagedProvider([RuntimeError("connection refused")])
    runner = run_interactive_chat_session_turn(
        user_message="Answer plainly.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    with pytest.raises(RuntimeError, match="connection refused"):
        list(runner)

    assert len(provider.calls) == 1


def test_chat_session_runner_stage_helpers_cover_remaining_edges(
    tmp_path: Path,
) -> None:
    runner = run_interactive_chat_session_turn(
        user_message="Show git status for this repo.",
        session_state=ChatSessionState(),
        executor=_empty_executor(),
        provider=_FakeProvider(
            [ParsedModelResponse(final_response={"answer": "unused"})]
        ),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
        enabled_tool_names={"run_git_status"},
    )

    assert runner._validation_error_summary(RuntimeError("")) == "RuntimeError"
    assert runner._repair_stage_guidance("tool:read_file").startswith(
        "Tool stage rules:"
    )
    assert (
        runner._repair_stage_guidance("mystery")
        == "Return only the fields required for this stage."
    )
    assert runner._format_invalid_payload(None) == "(unavailable)"
    assert runner._format_invalid_payload("payload") == "payload"
    bad_keys = {object(): "value"}
    assert runner._format_invalid_payload(bad_keys) == str(bad_keys)
    assert runner._should_retry_ungrounded_final_response(
        parsed=ParsedModelResponse(final_response={"answer": "draft"}),
        tool_results=[],
    )
    decision_context = runner._decision_tool_use_context(
        round_count=2,
        executed_tool_call_count=1,
        tool_results=[
            ToolResult(
                ok=True,
                tool_name="read_file",
                tool_version="0.1.0",
                metadata={
                    "execution_record": {
                        "request": {
                            "tool_name": "read_file",
                            "arguments": {"path": "README.md", "limit": 20},
                        }
                    }
                },
            )
        ],
    )
    assert "Tool rounds used: 2 of 8." in decision_context
    assert "Tool rounds remaining after choosing a tool now: 5." in decision_context
    assert "Total tool calls used: 1 of 12." in decision_context
    assert 'read_file({"limit":20,"path":"README.md"}) -> success' in decision_context
    assert "same tool again with different arguments" in decision_context
    assert (
        runner._tool_call_summary(
            ToolResult(
                ok=False,
                tool_name="read_file",
                tool_version="0.1.0",
                error=ToolError(
                    code=ErrorCode.RUNTIME_ERROR,
                    message="failed",
                ),
                metadata={
                    "execution_record": {
                        "request": {"tool_name": "", "arguments": "invalid"}
                    }
                },
            )
        )
        == "read_file({}) -> error:runtime_error"
    )
    assert runner._compact_json({"path": "a" * 100}, max_chars=20).endswith(
        "...(truncated)"
    )
    try:
        runner._tool_spec([], "missing")
    except ValueError as exc:
        assert "Unknown tool selected during staged interaction" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("Expected ValueError for missing staged tool")


def test_chat_session_runner_returns_continuation_for_tool_budget(
    tmp_path: Path,
) -> None:
    config = ChatSessionConfig(max_tool_calls_per_round=1)
    runner = run_interactive_chat_session_turn(
        user_message="Need more tools",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        {"tool_name": "list_directory", "arguments": {"path": "."}},
                        {
                            "tool_name": "find_files",
                            "arguments": {"path": ".", "pattern": "**/*.py"},
                        },
                    ]
                )
            ]
        ),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=config,
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    result_event = list(runner)[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "needs_continuation"
    assert "more tool calls" in (result_event.result.continuation_reason or "")


def test_chat_session_runner_interrupts_before_provider_call(tmp_path: Path) -> None:
    runner = run_interactive_chat_session_turn(
        user_message="Stop me",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=_FakeProvider([]),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )
    runner.cancel()
    result_event = list(runner)[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "interrupted"


def test_chat_session_runner_pauses_for_approval_and_resumes_same_turn(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)
    executor = WorkflowExecutor(
        registry=registry,
        policy=ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
            require_approval_for={SideEffectClass.LOCAL_READ},
        ),
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    runner = run_interactive_chat_session_turn(
        user_message="Where is needle defined?",
        session_state=ChatSessionState(),
        executor=executor,
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        {
                            "tool_name": "search_text",
                            "arguments": {"path": ".", "query": "needle"},
                        }
                    ]
                ),
                ParsedModelResponse(
                    final_response=ChatFinalResponse(
                        answer="It is defined in src/app.py."
                    ).model_dump(mode="json")
                ),
            ]
        ),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    iterator = iter(runner)
    assert isinstance(next(iterator), ChatWorkflowStatusEvent)
    assert isinstance(next(iterator), ChatWorkflowInspectorEvent)
    assert isinstance(next(iterator), ChatWorkflowInspectorEvent)
    approval_event = next(iterator)
    assert isinstance(approval_event, ChatWorkflowApprovalEvent)
    assert runner.resolve_pending_approval(True) is True
    resolved_event = next(iterator)
    assert isinstance(resolved_event, ChatWorkflowApprovalResolvedEvent)
    assert resolved_event.resolution == "approved"
    remaining = list(iterator)
    assert isinstance(remaining[-1], ChatWorkflowResultEvent)
    assert remaining[-1].result.status == "completed"


def test_chat_session_runner_denied_approval_continues_turn(tmp_path: Path) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)
    executor = WorkflowExecutor(
        registry=registry,
        policy=ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
            require_approval_for={SideEffectClass.LOCAL_READ},
        ),
    )
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    runner = run_interactive_chat_session_turn(
        user_message="Where is needle defined?",
        session_state=ChatSessionState(),
        executor=executor,
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        {
                            "tool_name": "search_text",
                            "arguments": {"path": ".", "query": "needle"},
                        }
                    ]
                ),
                ParsedModelResponse(
                    final_response=ChatFinalResponse(
                        answer="It is still defined in src/app.py."
                    ).model_dump(mode="json")
                ),
            ]
        ),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    iterator = iter(runner)
    assert isinstance(next(iterator), ChatWorkflowStatusEvent)
    assert isinstance(next(iterator), ChatWorkflowInspectorEvent)
    assert isinstance(next(iterator), ChatWorkflowInspectorEvent)
    approval_event = next(iterator)
    assert isinstance(approval_event, ChatWorkflowApprovalEvent)
    assert runner.resolve_pending_approval(False) is True
    resolved_event = next(iterator)
    assert isinstance(resolved_event, ChatWorkflowApprovalResolvedEvent)
    assert resolved_event.resolution == "denied"
    remaining = list(iterator)
    assert isinstance(remaining[-1], ChatWorkflowResultEvent)
    assert remaining[-1].result.status == "completed"


def test_chat_session_runner_cancelled_after_provider_call(tmp_path: Path) -> None:
    provider = _CancelOnRunProvider(
        ParsedModelResponse(
            final_response=ChatFinalResponse(answer="Should not be shown").model_dump(
                mode="json"
            )
        )
    )
    runner = run_interactive_chat_session_turn(
        user_message="Cancel me",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )
    provider.runner = runner

    events = list(runner)
    assert isinstance(events[-1], ChatWorkflowResultEvent)
    assert events[-1].result.status == "interrupted"
    assert events[-1].result.interruption_reason == "Interrupted by user."


def test_chat_session_runner_returns_continuation_for_total_tool_budget(
    tmp_path: Path,
) -> None:
    config = ChatSessionConfig(max_total_tool_calls_per_turn=1)
    runner = run_interactive_chat_session_turn(
        user_message="Need one tool",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        {"tool_name": "list_directory", "arguments": {"path": "."}},
                        {"tool_name": "find_files", "arguments": {"path": "."}},
                    ]
                )
            ]
        ),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=config,
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    result_event = list(runner)[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "needs_continuation"
    assert "tool-call budget" in (result_event.result.continuation_reason or "")


def test_chat_session_runner_forces_native_final_response_at_round_budget(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    provider = _FakeNativeLimitProvider(
        ParsedModelResponse(
            invocations=[{"tool_name": "list_directory", "arguments": {"path": "."}}]
        ),
        "The workspace contains src/app.py.",
    )
    runner = run_interactive_chat_session_turn(
        user_message="Inspect once",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(max_tool_round_trips=1),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "completed"
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == (
        "The workspace contains src/app.py."
    )
    assert (
        "Tool rounds remaining after choosing a tool now: 0."
        in provider.run_messages[0][-1]["content"]
    )
    assert provider.response_model_names == ["ChatFinalResponse"]
    assert "tool round budget" in provider.structured_messages[0][-1]["content"]


def test_chat_session_runner_forced_native_final_response_can_fall_back_to_text(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    provider = _FakeNativeTextFinalProvider(
        ParsedModelResponse(
            invocations=[{"tool_name": "list_directory", "arguments": {"path": "."}}]
        ),
        "The workspace contains src/app.py.",
    )
    runner = run_interactive_chat_session_turn(
        user_message="Inspect once",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(max_tool_round_trips=1),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)

    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "completed"
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == (
        "The workspace contains src/app.py."
    )
    assert provider.response_model_names == ["ChatFinalResponse"]
    assert len(provider.text_messages) == 1


def test_chat_session_runner_forces_staged_final_response_at_round_budget(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    provider = _FakeStagedProvider(
        [
            {"mode": "tool", "tool_name": "list_directory"},
            {
                "mode": "tool",
                "tool_name": "list_directory",
                "arguments": {"path": "."},
            },
            {
                "mode": "finalize",
                "final_response": {
                    "answer": "The workspace contains src/app.py.",
                    "citations": [{"source_path": "src/app.py"}],
                },
            },
        ]
    )
    runner = run_interactive_chat_session_turn(
        user_message="Inspect once",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=provider,
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(max_tool_round_trips=1),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    events = list(runner)

    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "completed"
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == (
        "The workspace contains src/app.py."
    )
    assert provider.response_model_names == [
        "DecisionStep",
        "ListDirectoryInvocationStep",
        "FinalResponseStep",
    ]
    assert "tool round budget" in provider.calls[-1][-2]["content"]
    assert "Current step: finalize the answer." in provider.calls[-1][-1]["content"]


def test_chat_session_runner_cancelled_while_waiting_for_approval(
    tmp_path: Path,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)
    executor = WorkflowExecutor(
        registry=registry,
        policy=ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
            require_approval_for={SideEffectClass.LOCAL_READ},
        ),
    )
    runner = run_interactive_chat_session_turn(
        user_message="Need approval",
        session_state=ChatSessionState(),
        executor=executor,
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        {"tool_name": "list_directory", "arguments": {"path": "."}}
                    ]
                )
            ]
        ),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    iterator = iter(runner)
    assert isinstance(next(iterator), ChatWorkflowStatusEvent)
    assert isinstance(next(iterator), ChatWorkflowInspectorEvent)
    assert isinstance(next(iterator), ChatWorkflowInspectorEvent)
    approval_event = next(iterator)
    assert isinstance(approval_event, ChatWorkflowApprovalEvent)
    runner.cancel()
    resolved = next(iterator)
    assert isinstance(resolved, ChatWorkflowApprovalResolvedEvent)
    assert resolved.resolution == "cancelled"
    result_event = next(iterator)
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "interrupted"
    assert (
        result_event.result.interruption_reason
        == "Interrupted while waiting for approval."
    )


def test_chat_session_runner_internal_approval_paths(tmp_path: Path) -> None:
    runner = run_interactive_chat_session_turn(
        user_message="internal",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=_FakeProvider([]),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )
    assert runner.resolve_pending_approval(True) is False
    runner._pending_approval = object()  # type: ignore[assignment]
    runner._pending_approval_decision = True
    assert runner.resolve_pending_approval(False) is False
    runner._pending_approval = None
    runner._pending_approval_decision = None

    request = ToolInvocationRequest(tool_name="list_directory", arguments={"path": "."})
    executed = WorkflowInvocationOutcome(
        invocation_index=1,
        request=request,
        status=WorkflowInvocationStatus.EXECUTED,
        tool_result=ToolResult(
            ok=True,
            tool_name="list_directory",
            tool_version="0.1.0",
            output={"entries": []},
        ),
    )
    workflow_result = WorkflowTurnResult(
        parsed_response=ParsedModelResponse(
            invocations=[request.model_dump(mode="json")]
        ),
        outcomes=[executed],
    )
    runner.cancel()
    interrupted_events = list(
        runner._consume_workflow_result(
            workflow_result=workflow_result,
            round_index=1,
            messages=[],
            new_messages=[],
            tool_results=[],
            executed_tool_call_count_ref=[0],
        )
    )
    assert isinstance(interrupted_events[-1], ChatWorkflowResultEvent)
    assert interrupted_events[-1].result.status == "interrupted"

    runner = run_interactive_chat_session_turn(
        user_message="timeout",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=_FakeProvider([]),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )
    future_request = _approval_request(
        request,
        expires_at=(datetime.now(UTC) + timedelta(minutes=1)).isoformat(),
    )
    expired_request = _approval_request(
        request,
        expires_at=(datetime.now(UTC) - timedelta(seconds=1)).isoformat(),
    )
    runner.cancel()
    assert runner._wait_for_approval_resolution(future_request) == "cancelled"

    runner = run_interactive_chat_session_turn(
        user_message="timeout",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=_FakeProvider([]),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )
    assert runner._wait_for_approval_resolution(expired_request) == "timed_out"

    approval_outcome = WorkflowInvocationOutcome(
        invocation_index=1,
        request=request,
        status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
        approval_request=future_request,
    )
    approval_workflow = WorkflowTurnResult(
        parsed_response=ParsedModelResponse(
            invocations=[request.model_dump(mode="json")]
        ),
        outcomes=[approval_outcome],
    )
    finalized = WorkflowTurnResult(
        parsed_response=ParsedModelResponse(
            invocations=[request.model_dump(mode="json")]
        ),
        outcomes=[executed],
    )
    runner._wait_for_approval_resolution = lambda approval_request: "timed_out"  # type: ignore[method-assign]
    runner._executor.finalize_expired_approvals = lambda now: [finalized]  # type: ignore[method-assign]
    timeout_events = list(
        runner._consume_workflow_result(
            workflow_result=approval_workflow,
            round_index=1,
            messages=[],
            new_messages=[],
            tool_results=[],
            executed_tool_call_count_ref=[0],
        )
    )
    assert isinstance(timeout_events[0], ChatWorkflowApprovalEvent)
    assert isinstance(timeout_events[1], ChatWorkflowApprovalResolvedEvent)
    assert timeout_events[1].resolution == "timed_out"
    assert any(
        isinstance(event, ChatWorkflowStatusEvent) and event.status == "listing files"
        for event in timeout_events
    )


class _ProtectionClassifier:
    def __init__(self, *, prompt: ProtectionAssessment, response: ProtectionAssessment):
        self._prompt = prompt
        self._response = response

    def assess_prompt(self, **kwargs) -> ProtectionAssessment:
        del kwargs
        return self._prompt

    def assess_response(self, **kwargs) -> ProtectionAssessment:
        del kwargs
        return self._response


class _UnexpectedProvider:
    def run(self, **kwargs) -> ParsedModelResponse:
        del kwargs
        raise AssertionError("provider should not be called")


def test_chat_session_runner_challenges_prompt_before_provider_call(
    tmp_path: Path,
) -> None:
    protection_controller = ProtectionController(
        config=ProtectionConfig(enabled=True),
        classifier=_ProtectionClassifier(
            prompt=ProtectionAssessment(
                reasoning="This may exceed the allowed sensitivity.",
                recommended_action=ProtectionAction.CHALLENGE,
            ),
            response=ProtectionAssessment(reasoning="unused"),
        ),
    )
    runner = run_interactive_chat_session_turn(
        user_message="Tell me the secret plan",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=_UnexpectedProvider(),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
        protection_controller=protection_controller,
    )

    events = list(runner)
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert "Potential sensitivity issue" in result_event.result.final_response.answer
    assert result_event.result.session_state is not None
    assert result_event.result.session_state.pending_protection_prompt is not None


def test_chat_session_runner_requires_label_for_incorrect_protection_feedback(
    tmp_path: Path,
) -> None:
    protection_controller = ProtectionController(
        config=ProtectionConfig(enabled=True),
        classifier=_ProtectionClassifier(
            prompt=ProtectionAssessment(reasoning="unused"),
            response=ProtectionAssessment(reasoning="unused"),
        ),
    )
    runner = run_interactive_chat_session_turn(
        user_message="analysis_is_correct: false",
        session_state=ChatSessionState(
            pending_protection_prompt=protection_controller.build_pending_prompt(
                original_user_message="Tell me the secret plan",
                serialized_messages=[
                    {"role": "user", "content": "Tell me the secret plan"}
                ],
                decision=protection_controller.assess_prompt(
                    messages=[{"role": "user", "content": "x"}]
                ),
                session_id="chat-session",
            )
        ),
        executor=_executor(),
        provider=_UnexpectedProvider(),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
        protection_controller=protection_controller,
    )

    result_event = list(runner)[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert "expected_sensitivity_label" in result_event.result.final_response.answer
    assert result_event.result.session_state is not None
    assert result_event.result.session_state.pending_protection_prompt is not None


def test_chat_session_runner_records_feedback_and_sanitizes_final_response(
    tmp_path: Path,
) -> None:
    corrections_path = tmp_path / "corrections.json"
    challenge_controller = ProtectionController(
        config=ProtectionConfig(enabled=True, corrections_path=str(corrections_path)),
        classifier=_ProtectionClassifier(
            prompt=ProtectionAssessment(
                reasoning="Potential issue.",
                recommended_action=ProtectionAction.CHALLENGE,
            ),
            response=ProtectionAssessment(reasoning="unused"),
        ),
    )
    pending_prompt = challenge_controller.build_pending_prompt(
        original_user_message="Tell me the secret plan",
        serialized_messages=[{"role": "user", "content": "Tell me the secret plan"}],
        decision=challenge_controller.assess_prompt(
            messages=[{"role": "user", "content": "Tell me the secret plan"}]
        ),
        session_id="chat-session",
    )
    feedback_runner = run_interactive_chat_session_turn(
        user_message="analysis_is_correct: false\nexpected_sensitivity_label: public",
        session_state=ChatSessionState(pending_protection_prompt=pending_prompt),
        executor=_executor(),
        provider=_UnexpectedProvider(),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
        protection_controller=challenge_controller,
    )
    feedback_result = list(feedback_runner)[-1]
    assert isinstance(feedback_result, ChatWorkflowResultEvent)
    assert corrections_path.exists()
    assert feedback_result.result.session_state is not None
    assert feedback_result.result.session_state.pending_protection_prompt is None

    review_controller = ProtectionController(
        config=ProtectionConfig(enabled=True),
        classifier=_ProtectionClassifier(
            prompt=ProtectionAssessment(reasoning="safe prompt"),
            response=ProtectionAssessment(
                reasoning="Sanitize before showing.",
                recommended_action=ProtectionAction.SANITIZE,
                sanitized_text="Safe replacement",
            ),
        ),
    )
    runner = run_interactive_chat_session_turn(
        user_message="Now answer safely",
        session_state=ChatSessionState(),
        executor=_executor(),
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    final_response=ChatFinalResponse(
                        answer="Sensitive raw answer"
                    ).model_dump(mode="json")
                )
            ]
        ),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=ChatSessionConfig(),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
        protection_controller=review_controller,
    )

    events = list(runner)
    parsed_events = [
        event
        for event in events
        if isinstance(event, ChatWorkflowInspectorEvent)
        and event.kind == "parsed_response"
    ]
    assert parsed_events
    assert "Sensitive raw answer" not in str(parsed_events[-1].payload)
    assert "Safe replacement" in str(parsed_events[-1].payload)
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "Safe replacement"
