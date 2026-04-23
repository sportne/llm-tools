"""Tests for interactive repository chat session orchestration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import (
    SideEffectClass,
    ToolContext,
    ToolInvocationRequest,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
)
from llm_tools.tool_api.redaction import RedactionConfig
from llm_tools.tools import register_filesystem_tools, register_text_tools
from llm_tools.tools.filesystem import ToolLimits
from llm_tools.workflow_api import (
    ChatFinalResponse,
    ChatSessionConfig,
    ChatSessionState,
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


class _FakeStagedProvider:
    def __init__(self, responses: list[object]) -> None:
        self._responses = list(responses)
        self.calls: list[list[dict[str, object]]] = []

    def run(self, **kwargs) -> ParsedModelResponse:
        del kwargs
        raise AssertionError("staged provider should use run_structured()")

    def run_structured(self, **kwargs) -> object:
        messages = kwargs["messages"]
        assert isinstance(messages, list)
        self.calls.append([dict(message) for message in messages])
        return self._responses.pop(0)

    def uses_staged_schema_protocol(self) -> bool:
        return True


class _CancelOnRunProvider:
    def __init__(self, response: ParsedModelResponse) -> None:
        self._response = response
        self.runner = None

    def run(self, **kwargs) -> ParsedModelResponse:
        del kwargs
        assert self.runner is not None
        self.runner.cancel()
        return self._response


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


def test_chat_session_runner_retries_ungrounded_repo_answer_with_tools(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    provider = _FakeProvider(
        [
            ParsedModelResponse(
                final_response=ChatFinalResponse(
                    answer="The app probably wires state through Streamlit session state."
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
                    answer="It wires state through Streamlit session state.",
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
        isinstance(event, ChatWorkflowStatusEvent)
        and event.status == "searching text"
        for event in events
    )
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.final_response is not None
    assert (
        result_event.result.final_response.answer
        == "It wires state through Streamlit session state."
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
    assert '"citations": [' in repair_message


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


def test_chat_session_runner_returns_continuation_for_round_trip_budget(
    tmp_path: Path,
) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("print('ok')\n", encoding="utf-8")
    runner = run_interactive_chat_session_turn(
        user_message="Inspect once",
        session_state=ChatSessionState(),
        executor=_executor(),
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
        session_config=ChatSessionConfig(max_tool_round_trips=1),
        tool_limits=ToolLimits(),
        redaction_config=RedactionConfig(),
        temperature=0.1,
    )

    result_event = list(runner)[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "needs_continuation"
    assert "more tool rounds" in (result_event.result.continuation_reason or "")


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
