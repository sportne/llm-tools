"""Additional tests for chat workflow models and helper functions."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from llm_tools.tool_api import ToolResult
from llm_tools.workflow_api import (
    ChatCitation,
    ChatFinalResponse,
    ChatMessage,
    ChatSessionConfig,
    ChatSessionState,
    ChatTokenUsage,
    ChatWorkflowTurnResult,
)
from llm_tools.workflow_api.chat_models import ChatSessionTurnRecord
from llm_tools.workflow_api.chat_session import (
    _build_continuation_result,
    _build_interrupted_result,
    _estimate_messages_tokens,
    _finalize_session_turn_result,
    _prepare_session_context,
    _sanitize_tool_result_message,
    _serialize_chat_message,
    _summarize_session_token_usage,
    _tool_status_label,
)


def test_chat_model_validators_cover_error_paths() -> None:
    with pytest.raises(ValidationError):
        ChatCitation(source_path=" ", line_start=2)
    with pytest.raises(ValidationError):
        ChatCitation(source_path="a.py", line_start=3, line_end=2)

    with pytest.raises(ValidationError):
        ChatFinalResponse(answer=" ")
    with pytest.raises(ValidationError):
        ChatFinalResponse(answer="ok", uncertainty=[" "])

    assert ChatTokenUsage(input_tokens=2, output_tokens=3).total_tokens == 5
    with pytest.raises(ValidationError):
        ChatTokenUsage(input_tokens=2, output_tokens=3, total_tokens=4)

    with pytest.raises(ValidationError):
        ChatMessage(role="user", content=" ", completion_state="complete")
    with pytest.raises(ValidationError):
        ChatMessage(role="user", content="x", completion_state="interrupted")

    final = ChatFinalResponse(answer="done")
    with pytest.raises(ValidationError):
        ChatSessionTurnRecord(status="completed", new_messages=[], final_response=None)
    with pytest.raises(ValidationError):
        ChatSessionTurnRecord(
            status="needs_continuation",
            continuation_reason=None,
            new_messages=[],
        )
    with pytest.raises(ValidationError):
        ChatSessionTurnRecord(
            status="interrupted",
            interruption_reason=None,
            new_messages=[],
        )
    with pytest.raises(ValidationError):
        ChatSessionState(turns=[], active_context_start_turn=1)
    with pytest.raises(ValidationError):
        ChatWorkflowTurnResult(status="completed", new_messages=[], final_response=None)
    with pytest.raises(ValidationError):
        ChatWorkflowTurnResult(status="needs_continuation", new_messages=[])
    with pytest.raises(ValidationError):
        ChatWorkflowTurnResult(status="interrupted", new_messages=[])
    assert ChatSessionTurnRecord(
        status="completed", new_messages=[], final_response=final
    )


def test_chat_session_internal_helpers_cover_branches() -> None:
    tool_message = ChatMessage(role="tool", content='{"ok":true}')
    assert _serialize_chat_message(tool_message)["role"] == "user"
    assert _tool_status_label("find_files") == "listing files"
    assert _tool_status_label("search_text") == "searching text"
    assert _tool_status_label("read_file") == "reading file"
    assert _tool_status_label("other") == "thinking"

    ok_tool = ToolResult(
        ok=True, tool_name="echo", tool_version="0.1.0", output={"x": 1}
    )
    err_tool = ToolResult(
        ok=False,
        tool_name="echo",
        tool_version="0.1.0",
        error={"code": "execution_failed", "message": "boom"},
    )
    err_tool_no_error = ToolResult(ok=False, tool_name="echo", tool_version="0.1.0")
    assert _sanitize_tool_result_message(ok_tool)["status"] == "ok"
    assert _sanitize_tool_result_message(err_tool)["status"] == "error"
    assert _sanitize_tool_result_message(err_tool_no_error)["error"] == {
        "message": "Unknown tool error"
    }

    continuation = _build_continuation_result(
        new_messages=[ChatMessage(role="user", content="q")],
        tool_results=[ok_tool],
        token_usage=None,
        reason="need more",
    )
    interrupted = _build_interrupted_result(
        new_messages=[ChatMessage(role="user", content="q")],
        tool_results=[],
        token_usage=None,
        reason="stop",
    )
    assert continuation.status == "needs_continuation"
    assert interrupted.status == "interrupted"

    prior_turn = ChatSessionTurnRecord(
        status="completed",
        new_messages=[
            ChatMessage(role="user", content="older question"),
            ChatMessage(role="assistant", content="older answer"),
        ],
        final_response=ChatFinalResponse(answer="older answer"),
        token_usage=ChatTokenUsage(total_tokens=20),
    )
    session_state = ChatSessionState(turns=[prior_turn], active_context_start_turn=0)
    prepared = _prepare_session_context(
        user_message="new question",
        session_state=session_state,
        system_prompt="sys",
        session_config=ChatSessionConfig(max_context_tokens=1),
    )
    assert prepared[4] is not None
    assert prepared[3] == 1
    assert _estimate_messages_tokens([ChatMessage(role="user", content="one two")]) == 2

    finalized = _finalize_session_turn_result(
        turn_result=ChatWorkflowTurnResult(
            status="completed",
            new_messages=[ChatMessage(role="user", content="new question")],
            final_response=ChatFinalResponse(answer="new answer"),
            token_usage=None,
        ),
        session_state=session_state,
        active_context_start_turn=0,
        context_warning="trimmed",
        system_message=ChatMessage(role="system", content="sys"),
    )
    assert finalized.session_state is not None
    assert finalized.context_warning == "trimmed"

    summarized = _summarize_session_token_usage(
        base_usage=ChatTokenUsage(input_tokens=1, output_tokens=2),
        session_state=finalized.session_state,
        active_context_messages=[ChatMessage(role="system", content="sys")],
    )
    assert summarized.session_tokens is not None
    assert summarized.active_context_tokens is not None
