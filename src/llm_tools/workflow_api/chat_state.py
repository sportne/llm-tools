"""State and transcript helpers for interactive chat sessions."""

from __future__ import annotations

import json
from datetime import UTC, datetime

from llm_tools.tool_api import ProtectionProvenanceSnapshot, ToolResult
from llm_tools.workflow_api.chat_models import (
    ChatFinalResponse,
    ChatMessage,
    ChatSessionConfig,
    ChatSessionState,
    ChatSessionTurnRecord,
    ChatTokenUsage,
    ChatWorkflowTurnResult,
)
from llm_tools.workflow_api.protection import collect_provenance_from_tool_results


def _build_turn_record(turn_result: ChatWorkflowTurnResult) -> ChatSessionTurnRecord:
    return ChatSessionTurnRecord(
        status=turn_result.status,
        new_messages=turn_result.new_messages,
        final_response=turn_result.final_response,
        token_usage=turn_result.token_usage,
        tool_results=turn_result.tool_results,
        continuation_reason=turn_result.continuation_reason,
        interruption_reason=turn_result.interruption_reason,
    )


def _build_continuation_result(
    *,
    new_messages: list[ChatMessage],
    tool_results: list[ToolResult],
    token_usage: ChatTokenUsage | None,
    reason: str,
) -> ChatWorkflowTurnResult:
    return ChatWorkflowTurnResult(
        status="needs_continuation",
        new_messages=new_messages,
        final_response=None,
        token_usage=token_usage,
        tool_results=tool_results,
        continuation_reason=reason,
    )


def _build_interrupted_result(
    *,
    new_messages: list[ChatMessage],
    tool_results: list[ToolResult],
    token_usage: ChatTokenUsage | None,
    reason: str,
) -> ChatWorkflowTurnResult:
    return ChatWorkflowTurnResult(
        status="interrupted",
        new_messages=new_messages,
        final_response=None,
        token_usage=token_usage,
        tool_results=tool_results,
        interruption_reason=reason,
    )


def _prepare_session_context(
    *,
    user_message: str,
    session_state: ChatSessionState,
    system_prompt: str,
    session_config: ChatSessionConfig,
) -> tuple[ChatMessage, ChatMessage, list[ChatMessage], int, str | None]:
    system_message = ChatMessage(role="system", content=system_prompt)
    user_chat_message = ChatMessage(role="user", content=user_message)
    active_context_start_turn = session_state.active_context_start_turn
    context_warning: str | None = None
    while active_context_start_turn < len(session_state.turns):
        prior_messages = _flatten_turn_messages(
            session_state.turns[active_context_start_turn:]
        )
        candidate_context_tokens = _estimate_messages_tokens(
            [system_message, *prior_messages, user_chat_message]
        )
        if candidate_context_tokens <= session_config.max_context_tokens:
            break
        active_context_start_turn += 1
        context_warning = (
            "Older turns were removed from active context to stay within the "
            "configured token limit."
        )
    prior_messages = _flatten_turn_messages(
        session_state.turns[active_context_start_turn:]
    )
    return (
        system_message,
        user_chat_message,
        prior_messages,
        active_context_start_turn,
        context_warning,
    )


def _flatten_turn_messages(turns: list[ChatSessionTurnRecord]) -> list[ChatMessage]:
    messages: list[ChatMessage] = []
    for turn in turns:
        messages.extend(turn.new_messages)
    return messages


def _estimate_messages_tokens(messages: list[ChatMessage]) -> int:
    return sum(_estimate_message_tokens(message) for message in messages)


def _estimate_message_tokens(message: ChatMessage) -> int:
    return len(_tokenize(message.content))


def _estimate_turn_total_tokens(turn: ChatSessionTurnRecord) -> int:
    if turn.token_usage is not None and turn.token_usage.total_tokens is not None:
        return turn.token_usage.total_tokens
    return _estimate_messages_tokens(turn.new_messages)


def _summarize_session_token_usage(
    *,
    base_usage: ChatTokenUsage | None,
    session_state: ChatSessionState,
    active_context_messages: list[ChatMessage],
) -> ChatTokenUsage:
    session_tokens = sum(
        _estimate_turn_total_tokens(turn) for turn in session_state.turns
    )
    active_context_tokens = _estimate_messages_tokens(active_context_messages)
    if base_usage is None:
        current_turn_total_tokens = (
            _estimate_turn_total_tokens(session_state.turns[-1])
            if session_state.turns
            else 0
        )
        return ChatTokenUsage(
            total_tokens=current_turn_total_tokens,
            session_tokens=session_tokens,
            active_context_tokens=active_context_tokens,
        )
    return base_usage.model_copy(
        update={
            "session_tokens": session_tokens,
            "active_context_tokens": active_context_tokens,
        }
    )


def _finalize_session_turn_result(
    *,
    turn_result: ChatWorkflowTurnResult,
    session_state: ChatSessionState,
    active_context_start_turn: int,
    context_warning: str | None,
    system_message: ChatMessage,
) -> ChatWorkflowTurnResult:
    updated_turns = [*session_state.turns, _build_turn_record(turn_result)]
    updated_session_state = ChatSessionState(
        turns=updated_turns,
        active_context_start_turn=active_context_start_turn,
        pending_protection_prompt=turn_result.pending_protection_prompt,
    )
    active_context_messages = [
        system_message,
        *_flatten_turn_messages(
            updated_session_state.turns[
                updated_session_state.active_context_start_turn :
            ]
        ),
    ]
    token_usage = _summarize_session_token_usage(
        base_usage=turn_result.token_usage,
        session_state=updated_session_state,
        active_context_messages=active_context_messages,
    )
    return turn_result.model_copy(
        update={
            "token_usage": token_usage,
            "session_state": updated_session_state,
            "context_warning": context_warning,
        }
    )


def _assistant_message_for_final_response(
    final_response: ChatFinalResponse,
) -> ChatMessage:
    return ChatMessage(
        role="assistant",
        content=json.dumps(
            {
                "actions": [],
                "final_response": final_response.model_dump(mode="json"),
            },
            sort_keys=True,
            default=str,
        ),
    )


def _collect_chat_provenance(
    *,
    session_state: ChatSessionState,
    current_tool_results: list[ToolResult],
) -> ProtectionProvenanceSnapshot:
    collected: list[ToolResult] = []
    for turn in session_state.turns:
        collected.extend(turn.tool_results)
    collected.extend(current_tool_results)
    return collect_provenance_from_tool_results(collected)


def _tokenize(text: str) -> list[str]:
    return [part.lower() for part in text.replace("\n", " ").split() if part.strip()]


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


__all__ = [
    "_assistant_message_for_final_response",
    "_build_continuation_result",
    "_build_interrupted_result",
    "_estimate_messages_tokens",
    "_finalize_session_turn_result",
    "_parse_timestamp",
    "_prepare_session_context",
    "_summarize_session_token_usage",
]
