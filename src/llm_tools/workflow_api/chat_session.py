"""Interactive multi-turn repository chat orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterator
from threading import Lock
from uuid import uuid4

from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import ToolContext, ToolResult
from llm_tools.tools.filesystem._content import dump_json
from llm_tools.tools.filesystem.models import ToolLimits
from llm_tools.workflow_api.chat_models import (
    ChatFinalResponse,
    ChatMessage,
    ChatSessionConfig,
    ChatSessionState,
    ChatSessionTurnRecord,
    ChatTokenUsage,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
)
from llm_tools.workflow_api.executor import WorkflowExecutor
from llm_tools.workflow_api.models import WorkflowInvocationStatus


class ChatSessionTurnRunner:
    """Session-aware cancellable chat workflow for the interactive UI."""

    def __init__(
        self,
        *,
        user_message: str,
        session_state: ChatSessionState,
        executor: WorkflowExecutor,
        provider: OpenAICompatibleProvider,
        system_prompt: str,
        base_context: ToolContext,
        session_config: ChatSessionConfig,
        tool_limits: ToolLimits,
        temperature: float,
    ) -> None:
        (
            self._system_message,
            self._user_chat_message,
            self._prior_messages,
            self._active_context_start_turn,
            self._context_warning,
        ) = _prepare_session_context(
            user_message=user_message,
            session_state=session_state,
            system_prompt=system_prompt,
            session_config=session_config,
        )
        self._session_state = session_state
        self._executor = executor
        self._provider = provider
        self._base_context = base_context
        self._session_config = session_config
        self._tool_limits = tool_limits
        self._temperature = temperature
        self._adapter = ActionEnvelopeAdapter()
        self._lock = Lock()
        self._cancel_requested = False

    def cancel(self) -> None:
        """Request cooperative cancellation at the next safe boundary."""
        with self._lock:
            self._cancel_requested = True

    def _finalized_event(
        self,
        *,
        result: ChatWorkflowTurnResult,
    ) -> ChatWorkflowResultEvent:
        return ChatWorkflowResultEvent(
            result=_finalize_session_turn_result(
                turn_result=result,
                session_state=self._session_state,
                active_context_start_turn=self._active_context_start_turn,
                context_warning=self._context_warning,
                system_message=self._system_message,
            )
        )

    def __iter__(
        self,
    ) -> Iterator[ChatWorkflowStatusEvent | ChatWorkflowResultEvent]:
        messages = [
            self._system_message,
            *self._prior_messages,
            self._user_chat_message,
        ]
        new_messages = [self._user_chat_message]
        tool_results: list[ToolResult] = []
        round_count = 0
        executed_tool_call_count = 0
        yield ChatWorkflowStatusEvent(status="thinking")

        while True:
            if self._cancel_requested:
                yield self._finalized_event(
                    result=_build_interrupted_result(
                        new_messages=new_messages,
                        tool_results=tool_results,
                        token_usage=None,
                        reason="Interrupted by user.",
                    )
                )
                return

            round_context = self._base_context.model_copy(
                update={
                    "invocation_id": f"{self._base_context.invocation_id}:{uuid4()}"
                }
            )
            prepared = self._executor.prepare_model_interaction(
                self._adapter,
                context=round_context,
                final_response_model=ChatFinalResponse,
            )
            parsed = self._provider.run(
                adapter=self._adapter,
                messages=[_serialize_chat_message(message) for message in messages],
                response_model=prepared.response_model,
                request_params={"temperature": self._temperature},
            )

            if self._cancel_requested:
                yield self._finalized_event(
                    result=_build_interrupted_result(
                        new_messages=new_messages,
                        tool_results=tool_results,
                        token_usage=None,
                        reason="Interrupted by user.",
                    )
                )
                return

            assistant_message = ChatMessage(
                role="assistant",
                content=json.dumps(
                    {
                        "actions": [
                            invocation.model_dump(mode="json")
                            for invocation in parsed.invocations
                        ],
                        "final_response": parsed.final_response,
                    },
                    sort_keys=True,
                    default=str,
                ),
            )
            messages.append(assistant_message)
            new_messages.append(assistant_message)

            if parsed.final_response is not None:
                yield ChatWorkflowStatusEvent(status="drafting answer")
                final_response = ChatFinalResponse.model_validate(parsed.final_response)
                yield self._finalized_event(
                    result=ChatWorkflowTurnResult(
                        status="completed",
                        new_messages=new_messages,
                        final_response=final_response,
                        token_usage=None,
                        tool_results=tool_results,
                    )
                )
                return

            if len(parsed.invocations) > self._session_config.max_tool_calls_per_round:
                yield self._finalized_event(
                    result=_build_continuation_result(
                        new_messages=new_messages,
                        tool_results=tool_results,
                        token_usage=None,
                        reason=(
                            "The model requested more tool calls in one round than "
                            "allowed. User confirmation is required before continuing."
                        ),
                    )
                )
                return

            if (
                executed_tool_call_count + len(parsed.invocations)
                > self._session_config.max_total_tool_calls_per_turn
            ):
                yield self._finalized_event(
                    result=_build_continuation_result(
                        new_messages=new_messages,
                        tool_results=tool_results,
                        token_usage=None,
                        reason=(
                            "The model needs more total tool-call budget before it can "
                            "continue this turn."
                        ),
                    )
                )
                return

            workflow_result = self._executor.execute_parsed_response(
                parsed, round_context
            )
            for outcome in workflow_result.outcomes:
                if self._cancel_requested:
                    yield self._finalized_event(
                        result=_build_interrupted_result(
                            new_messages=new_messages,
                            tool_results=tool_results,
                            token_usage=None,
                            reason="Interrupted by user.",
                        )
                    )
                    return
                if (
                    outcome.status is not WorkflowInvocationStatus.EXECUTED
                    or outcome.tool_result is None
                ):
                    continue
                yield ChatWorkflowStatusEvent(
                    status=_tool_status_label(outcome.request.tool_name)
                )
                tool_results.append(outcome.tool_result)
                executed_tool_call_count += 1
                tool_message = ChatMessage(
                    role="tool",
                    content=_format_tool_result_for_model(
                        _sanitize_tool_result_message(outcome.tool_result),
                        max_chars=self._tool_limits.max_tool_result_chars,
                    ),
                )
                messages.append(tool_message)
                new_messages.append(tool_message)

            round_count += 1
            if round_count >= self._session_config.max_tool_round_trips:
                yield self._finalized_event(
                    result=_build_continuation_result(
                        new_messages=new_messages,
                        tool_results=tool_results,
                        token_usage=None,
                        reason=(
                            "The model needs more tool rounds before it can provide "
                            "a final response."
                        ),
                    )
                )
                return

            yield ChatWorkflowStatusEvent(status="thinking")


def run_interactive_chat_session_turn(
    *,
    user_message: str,
    session_state: ChatSessionState,
    executor: WorkflowExecutor,
    provider: OpenAICompatibleProvider,
    system_prompt: str,
    base_context: ToolContext,
    session_config: ChatSessionConfig,
    tool_limits: ToolLimits,
    temperature: float,
) -> ChatSessionTurnRunner:
    """Return an interruptible multi-turn chat runner for one user message."""
    return ChatSessionTurnRunner(
        user_message=user_message,
        session_state=session_state,
        executor=executor,
        provider=provider,
        system_prompt=system_prompt,
        base_context=base_context,
        session_config=session_config,
        tool_limits=tool_limits,
        temperature=temperature,
    )


def _serialize_chat_message(message: ChatMessage) -> dict[str, str]:
    if message.role == "tool":
        return {"role": "user", "content": f"Tool result:\n{message.content}"}
    return {"role": message.role, "content": message.content}


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


def _tool_status_label(tool_name: str) -> str:
    if tool_name in {"list_directory", "find_files"}:
        return "listing files"
    if tool_name == "search_text":
        return "searching text"
    if tool_name in {"get_file_info", "read_file"}:
        return "reading file"
    return "thinking"


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


def _tokenize(text: str) -> list[str]:
    return [part.lower() for part in text.replace("\n", " ").split() if part.strip()]


def _sanitize_tool_result_message(tool_result: ToolResult) -> dict[str, object]:
    if tool_result.ok:
        return {
            "tool_name": tool_result.tool_name,
            "status": "ok",
            "output": tool_result.output,
        }
    return {
        "tool_name": tool_result.tool_name,
        "status": "error",
        "error": (
            tool_result.error.model_dump(mode="json")
            if tool_result.error is not None
            else {"message": "Unknown tool error"}
        ),
    }


def _format_tool_result_for_model(result: dict[str, object], *, max_chars: int) -> str:
    rendered = dump_json(result)
    if len(rendered) <= max_chars:
        return rendered
    return f"{rendered[:max_chars]}...(truncated)"
