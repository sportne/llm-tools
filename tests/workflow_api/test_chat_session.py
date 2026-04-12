"""Tests for interactive repository chat session orchestration."""

from __future__ import annotations

from pathlib import Path

from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolRegistry
from llm_tools.tools.chat import ChatSessionConfig, ChatToolLimits, register_chat_tools
from llm_tools.workflow_api import (
    ChatFinalResponse,
    ChatSessionState,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    WorkflowExecutor,
    run_interactive_chat_session_turn,
)


class _FakeProvider:
    def __init__(self, responses: list[ParsedModelResponse]) -> None:
        self._responses = list(responses)

    def run(self, **kwargs) -> ParsedModelResponse:
        del kwargs
        return self._responses.pop(0)


def _executor() -> WorkflowExecutor:
    registry = ToolRegistry()
    register_chat_tools(registry)
    return WorkflowExecutor(
        registry=registry,
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
            "session_config": ChatSessionConfig().model_dump(mode="json"),
            "tool_limits": ChatToolLimits(max_tool_result_chars=500).model_dump(mode="json"),
        },
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
        tool_limits=ChatToolLimits(max_tool_result_chars=500),
        temperature=0.1,
    )

    events = list(runner)
    assert isinstance(events[0], ChatWorkflowStatusEvent)
    result_event = events[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "completed"
    assert result_event.result.final_response is not None
    assert result_event.result.final_response.answer == "It is defined in src/app.py."
    assert result_event.result.session_state is not None
    assert len(result_event.result.session_state.turns) == 1


def test_chat_session_runner_returns_continuation_for_tool_budget(tmp_path: Path) -> None:
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
                        {"tool_name": "find_files", "arguments": {"path": ".", "pattern": "**/*.py"}},
                    ]
                )
            ]
        ),
        system_prompt="You are helpful.",
        base_context=_context(tmp_path),
        session_config=config,
        tool_limits=ChatToolLimits(),
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
        tool_limits=ChatToolLimits(),
        temperature=0.1,
    )
    runner.cancel()
    result_event = list(runner)[-1]
    assert isinstance(result_event, ChatWorkflowResultEvent)
    assert result_event.result.status == "interrupted"
