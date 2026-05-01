"""Additional tests for chat workflow models and helper functions."""

from __future__ import annotations

from datetime import UTC
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolResult
from llm_tools.tool_api.redaction import RedactionConfig, RedactionRule, RedactionTarget
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
    _format_tool_result_for_model,
    _parse_timestamp,
    _prepare_session_context,
    _redact_tool_payload,
    _sanitize_execution_record,
    _sanitize_invocation_payload,
    _sanitize_parsed_response_for_inspector,
    _sanitize_tool_result_for_chat,
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
    assert _estimate_messages_tokens([ChatMessage(role="user", content="one two")]) == 6
    assert (
        _estimate_messages_tokens([ChatMessage(role="user", content="x" * 100)]) == 29
    )

    finalized = _finalize_session_turn_result(
        turn_result=ChatWorkflowTurnResult(
            status="completed",
            new_messages=[ChatMessage(role="user", content="new question")],
            final_response=ChatFinalResponse(answer="new answer"),
            token_usage=None,
        ),
        session_state=session_state,
        active_context_start_turn=0,
        context_summary=None,
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


def test_chat_session_sanitizer_helpers_cover_edge_cases() -> None:
    no_trim = _prepare_session_context(
        user_message="new question",
        session_state=ChatSessionState(),
        system_prompt="sys",
        session_config=ChatSessionConfig(max_context_tokens=100),
    )
    assert no_trim[3] == 0
    assert no_trim[4] is None

    rendered = _format_tool_result_for_model({"value": "x" * 80}, max_chars=20)
    assert rendered.endswith("...(truncated)")

    execution_record = {
        "request": {"arguments": {"secret": "plain"}},
        "redacted_input": {"secret": "[REDACTED]"},
        "validated_input": {"secret": "plain"},
        "validated_output": {"ok": True},
    }
    sanitized_record = _sanitize_execution_record(execution_record)
    assert sanitized_record == {
        "request": {"arguments": {"secret": "[REDACTED]"}},
        "redacted_input": {"secret": "[REDACTED]"},
    }
    assert _sanitize_execution_record("bad") is None
    assert _sanitize_execution_record({"request": "bad", "redacted_input": {}}) == {
        "request": "bad",
        "redacted_input": {},
    }

    raw_result = ToolResult(
        ok=True,
        tool_name="read_file",
        tool_version="0.1.0",
        output={"content": "ok"},
        metadata={"execution_record": execution_record},
    )
    sanitized_result = _sanitize_tool_result_for_chat(raw_result)
    assert sanitized_result.metadata["execution_record"] == sanitized_record

    untouched_result = ToolResult(
        ok=True,
        tool_name="read_file",
        tool_version="0.1.0",
        output={"content": "ok"},
        metadata={"execution_record": "bad"},
    )
    assert (
        _sanitize_tool_result_for_chat(untouched_result).metadata["execution_record"]
        == "bad"
    )

    redaction_config = RedactionConfig(
        rules=[
            RedactionRule(
                field_names={"private_value"},
                targets={RedactionTarget.INPUT},
                replacement="[MASKED]",
            )
        ]
    )
    assert (
        _sanitize_parsed_response_for_inspector(
            "raw",
            redaction_config=redaction_config,
        )
        == "raw"
    )

    not_a_list = SimpleNamespace(
        model_dump=lambda mode="json": {"invocations": "bad", "final_response": None}
    )
    assert _sanitize_parsed_response_for_inspector(
        not_a_list,
        redaction_config=redaction_config,
    ) == {"invocations": "bad", "final_response": None}

    parsed = ParsedModelResponse(
        invocations=[
            {
                "tool_name": "read_file",
                "arguments": {"path": "a.txt", "private_value": "plain"},
            }
        ]
    )
    sanitized_parsed = _sanitize_parsed_response_for_inspector(
        parsed,
        redaction_config=redaction_config,
    )
    assert (
        sanitized_parsed["invocations"][0]["arguments"]["private_value"] == "[MASKED]"
    )

    assert (
        _sanitize_invocation_payload(
            "raw",
            redaction_config=redaction_config,
        )
        == "raw"
    )
    assert _sanitize_invocation_payload(
        {"tool_name": 1, "arguments": {}},
        redaction_config=redaction_config,
    ) == {"tool_name": 1, "arguments": {}}
    assert _sanitize_invocation_payload(
        {"tool_name": "read_file", "arguments": "bad"},
        redaction_config=redaction_config,
    ) == {"tool_name": "read_file", "arguments": "bad"}

    assert (
        _redact_tool_payload(
            "read_file",
            "plain-text payload",
            target=RedactionTarget.INPUT,
            redaction_config=redaction_config,
        )
        == {}
    )

    timestamp = _parse_timestamp("2026-01-01T00:00:00Z")
    assert timestamp.tzinfo is UTC
