"""Interactive multi-turn repository chat orchestration."""

from __future__ import annotations

import json
import os
from collections.abc import Generator, Iterator
from datetime import UTC, datetime
from threading import Condition
from typing import Any, Protocol, cast
from uuid import uuid4

from pydantic import BaseModel

from llm_tools.llm_adapters import (
    ActionEnvelopeAdapter,
    ParsedModelResponse,
)
from llm_tools.tool_api import (
    ProtectionProvenanceSnapshot,
    ToolContext,
    ToolResult,
)
from llm_tools.tool_api.redaction import RedactionConfig
from llm_tools.tools.filesystem.models import ToolLimits
from llm_tools.workflow_api.chat_inspector import (
    _build_approval_state,
    _format_tool_result_for_model,
    _sanitize_execution_record,
    _sanitize_parsed_response_for_inspector,
    _sanitize_tool_result_for_chat,
    _sanitize_tool_result_message,
    _serialize_chat_message,
    _tool_status_label,
)
from llm_tools.workflow_api.chat_models import (
    ChatApprovalResolution,
    ChatContextSummary,
    ChatFinalResponse,
    ChatMessage,
    ChatSessionConfig,
    ChatSessionState,
    ChatSessionTurnRecord,
    ChatWorkflowApprovalEvent,
    ChatWorkflowApprovalResolvedEvent,
    ChatWorkflowApprovalState,
    ChatWorkflowInspectorEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
)
from llm_tools.workflow_api.chat_state import (
    _assistant_message_for_final_response,
    _build_continuation_result,
    _build_interrupted_result,
    _collect_chat_provenance,
    _finalize_session_turn_result,
    _parse_timestamp,
    _prepare_session_context,
)
from llm_tools.workflow_api.executor import PreparedModelInteraction, WorkflowExecutor
from llm_tools.workflow_api.model_turn_protocol import (
    ModelTurnProtectionContext,
    ModelTurnProtocolEvent,
    ModelTurnProtocolRequest,
    ModelTurnProtocolRunner,
)
from llm_tools.workflow_api.models import (
    ApprovalRequest,
    WorkflowInvocationStatus,
    WorkflowTurnResult,
)
from llm_tools.workflow_api.protection import ProtectionAction, ProtectionController
from llm_tools.workflow_api.staged_structured import (
    format_invalid_payload,
    validation_error_summary,
)

_LOCAL_GROUNDING_TOOL_NAMES = frozenset(
    {
        "list_directory",
        "find_files",
        "get_file_info",
        "read_file",
        "search_text",
        "read_git_file",
        "run_git_status",
        "run_git_diff",
        "run_git_log",
        "show_git_status",
        "read_git_diff",
    }
)
_GIT_GROUNDING_TOOL_NAMES = frozenset(
    {
        "read_git_file",
        "run_git_status",
        "run_git_diff",
        "run_git_log",
        "show_git_status",
        "read_git_diff",
    }
)
_EXPLICIT_GROUNDING_HINTS = (
    "must use",
    "use one or more",
    "use the workspace tools",
    "use local workspace tools",
    "cite local files",
    "cite the most relevant local files",
    "ground your answer",
    "inspect the repo",
    "inspect the repository",
)
_REPOSITORY_GROUNDING_HINTS = (
    "repository",
    "repo",
    "codebase",
    "workspace",
    "project",
    "local files",
    "source tree",
    "interactive assistant",
    "app/runtime",
    "runtime flow",
    "wires together",
    "module",
    "function",
    "class",
    "file",
    "files",
)
_GIT_GROUNDING_HINTS = (
    "git status",
    "git diff",
    "git log",
    "recent change",
    "recent changes",
    "commit",
    "diff",
    "tracked file",
)

_JSON_AGENT_STRATEGY_ENV = "LLM_TOOLS_JSON_AGENT_STRATEGY"
_CONTEXT_LIMIT_ERROR_MARKERS = (
    "context_length_exceeded",
    "maximum context length",
    "max context length",
    "context length",
    "too many tokens",
    "prompt is too long",
    "prompt too long",
    "token limit",
    "tokens exceeds",
    "input is too long",
    "request too large",
)


class ModelTurnProvider(Protocol):
    """Minimal provider surface required by interactive chat orchestration."""

    def run(
        self,
        *,
        adapter: ActionEnvelopeAdapter,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        """Return one parsed model response for a workflow turn."""

    def run_structured(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> object:
        """Return one structured payload for a staged workflow step."""

    def uses_staged_schema_protocol(self) -> bool:
        """Return whether this provider expects staged strict schemas."""

    def run_text(
        self,
        *,
        messages: list[dict[str, Any]],
        request_params: dict[str, Any] | None = None,
    ) -> str:
        """Return one plain text model response."""


class ChatSessionTurnRunner:
    """Session-aware cancellable chat workflow for the interactive UI."""

    def __init__(
        self,
        *,
        user_message: str,
        session_state: ChatSessionState,
        executor: WorkflowExecutor,
        provider: ModelTurnProvider,
        system_prompt: str,
        base_context: ToolContext,
        session_config: ChatSessionConfig,
        tool_limits: ToolLimits,
        redaction_config: RedactionConfig,
        temperature: float,
        protection_controller: ProtectionController | None = None,
        enabled_tool_names: set[str] | None = None,
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
        self._context_summary = session_state.context_summary
        self._executor = executor
        self._provider = provider
        self._base_context = base_context
        self._session_config = session_config
        self._tool_limits = tool_limits
        self._redaction_config = redaction_config
        self._temperature = temperature
        self._protection_controller = protection_controller
        self._enabled_tool_names = frozenset(enabled_tool_names or ())
        self._adapter = ActionEnvelopeAdapter()
        self._approval_condition = Condition()
        self._cancel_requested: bool = False
        self._pending_approval: ChatWorkflowApprovalState | None = None
        self._pending_approval_decision: bool | None = None
        self._grounding_retry_used = False
        self._context_limit_retry_used = False
        self._protocol_metadata: dict[str, object] = {}

    def cancel(self) -> None:
        """Request cooperative cancellation at the next safe boundary."""
        with self._approval_condition:
            self._cancel_requested = True
            self._approval_condition.notify_all()

    def _is_cancel_requested(self) -> bool:
        return self._cancel_requested

    def resolve_pending_approval(self, approved: bool) -> bool:
        """Resolve the currently pending approval request, when present."""
        with self._approval_condition:
            if (
                self._pending_approval is None
                or self._pending_approval_decision is not None
            ):
                return False
            self._pending_approval_decision = approved
            self._approval_condition.notify_all()
            return True

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
                context_summary=self._context_summary,
                context_warning=self._context_warning,
                system_message=self._system_message,
            )
        )

    def __iter__(  # noqa: C901
        self,
    ) -> Iterator[
        ChatWorkflowStatusEvent
        | ChatWorkflowApprovalEvent
        | ChatWorkflowApprovalResolvedEvent
        | ChatWorkflowInspectorEvent
        | ChatWorkflowResultEvent
    ]:
        new_messages = [self._user_chat_message]
        tool_results: list[ToolResult] = []
        round_count = 0
        executed_tool_call_count = 0
        yield ChatWorkflowStatusEvent(status="thinking")
        if self._context_compaction_needed(force_all_prior=False):
            yield ChatWorkflowStatusEvent(status="compacting context")
            self._compact_context_if_needed(force_all_prior=False)
        messages = self._messages_with_current_turn(new_messages)

        pending_prompt_event = self._maybe_handle_pending_protection_prompt(
            new_messages=new_messages,
            tool_results=tool_results,
        )
        if pending_prompt_event is not None:
            yield pending_prompt_event
            return

        while True:
            if self._is_cancel_requested():
                yield self._interrupted_event(
                    new_messages=new_messages,
                    tool_results=tool_results,
                    reason="Interrupted by user.",
                )
                return

            (
                round_context,
                prepared,
                round_index,
                serialized_messages,
                provenance,
            ) = self._build_round_inputs(
                messages=messages,
                tool_results=tool_results,
                round_count=round_count,
            )
            decision_context = self._decision_tool_use_context(
                round_count=round_count,
                executed_tool_call_count=executed_tool_call_count,
                tool_results=tool_results,
            )
            try:
                parsed = yield from self._run_model_turn_protocol(
                    round_index=round_index,
                    messages=serialized_messages,
                    prepared=prepared,
                    decision_context=decision_context,
                    provenance=provenance,
                    mode="auto",
                )
            except Exception as exc:
                if not self._should_retry_after_context_limit(exc):
                    raise
                yield ChatWorkflowStatusEvent(status="compacting context")
                if not self._compact_context_if_needed(force_all_prior=True):
                    raise
                messages = self._messages_with_current_turn(new_messages)
                continue
            yield ChatWorkflowInspectorEvent(
                round_index=round_index,
                kind="parsed_response",
                payload=_sanitize_parsed_response_for_inspector(
                    parsed,
                    redaction_config=self._redaction_config,
                ),
            )

            if self._cancel_requested:
                yield self._interrupted_event(
                    new_messages=new_messages,
                    tool_results=tool_results,
                    reason="Interrupted by user.",
                )
                return

            if self._should_retry_ungrounded_final_response(
                parsed=parsed,
                tool_results=tool_results,
            ):
                self._grounding_retry_used = True
                messages.append(self._grounding_repair_message())
                round_count += 1
                round_limit_event = self._round_trip_limit_event(
                    round_count=round_count,
                    new_messages=new_messages,
                    tool_results=tool_results,
                )
                if round_limit_event is not None:
                    yield round_limit_event
                    return
                yield ChatWorkflowStatusEvent(status="gathering evidence")
                continue

            self._append_assistant_message(
                messages=messages,
                new_messages=new_messages,
                parsed=parsed,
            )

            final_response_event = self._maybe_finalize_from_response(
                parsed=parsed,
                new_messages=new_messages,
                tool_results=tool_results,
            )
            if final_response_event is not None:
                yield ChatWorkflowStatusEvent(status="drafting answer")
                yield final_response_event
                return

            continuation_event = self._continuation_limit_event(
                parsed=parsed,
                new_messages=new_messages,
                tool_results=tool_results,
                executed_tool_call_count=executed_tool_call_count,
            )
            if continuation_event is not None:
                yield continuation_event
                return

            workflow_result = self._executor.execute_parsed_response(
                parsed, round_context
            )
            interrupted = False
            for event in self._consume_workflow_result(
                workflow_result=workflow_result,
                round_index=round_index,
                messages=messages,
                new_messages=new_messages,
                tool_results=tool_results,
                executed_tool_call_count_ref=[executed_tool_call_count],
            ):
                if isinstance(event, ChatWorkflowResultEvent):
                    interrupted = True
                yield event
            executed_tool_call_count = self._executed_tool_call_count(
                tool_results=tool_results
            )
            if interrupted:
                return

            round_count += 1
            if round_count >= self._session_config.max_tool_round_trips:
                forced_final_event = yield from self._force_final_response_at_limit(
                    round_index=round_count + 1,
                    messages=messages,
                    new_messages=new_messages,
                    tool_results=tool_results,
                )
                if forced_final_event is not None:
                    yield forced_final_event
                    return
                round_limit_event = self._round_trip_limit_event(
                    round_count=round_count,
                    new_messages=new_messages,
                    tool_results=tool_results,
                )
                if round_limit_event is not None:
                    yield round_limit_event
                return

            yield ChatWorkflowStatusEvent(status="thinking")

    def _build_round_inputs(
        self,
        *,
        messages: list[ChatMessage],
        tool_results: list[ToolResult],
        round_count: int,
    ) -> tuple[
        ToolContext,
        PreparedModelInteraction,
        int,
        list[dict[str, Any]],
        ProtectionProvenanceSnapshot,
    ]:
        round_context = self._base_context.model_copy(
            update={"invocation_id": f"{self._base_context.invocation_id}:{uuid4()}"}
        )
        prepared = self._executor.prepare_model_interaction(
            self._adapter,
            context=round_context,
            include_requires_approval=True,
            final_response_model=ChatFinalResponse,
        )
        round_index = round_count + 1
        serialized_messages = self._serialize_messages_for_model(messages)
        provenance = _collect_chat_provenance(
            session_state=self._session_state,
            current_tool_results=tool_results,
        )
        return (
            round_context,
            prepared,
            round_index,
            serialized_messages,
            provenance,
        )

    def _messages_with_current_turn(
        self, new_messages: list[ChatMessage]
    ) -> list[ChatMessage]:
        return [self._system_message, *self._prior_messages, *new_messages]

    def _refresh_prepared_context(self) -> None:
        (
            self._system_message,
            self._user_chat_message,
            self._prior_messages,
            self._active_context_start_turn,
            context_warning,
        ) = _prepare_session_context(
            user_message=self._user_chat_message.content,
            session_state=self._session_state.model_copy(
                update={
                    "active_context_start_turn": self._active_context_start_turn,
                    "context_summary": self._context_summary,
                }
            ),
            system_prompt=self._system_message.content,
            session_config=self._session_config,
        )
        if context_warning is not None and self._context_warning is None:
            self._context_warning = context_warning

    def _context_compaction_needed(self, *, force_all_prior: bool) -> bool:
        _target_turn_count, turns_to_compact = self._context_turns_to_compact(
            force_all_prior=force_all_prior
        )
        return bool(turns_to_compact)

    def _context_turns_to_compact(
        self, *, force_all_prior: bool
    ) -> tuple[int, list[ChatSessionTurnRecord]]:
        target_turn_count = (
            len(self._session_state.turns)
            if force_all_prior
            else self._active_context_start_turn
        )
        covered_turn_count = (
            self._context_summary.covered_turn_count
            if self._context_summary is not None
            else 0
        )
        if target_turn_count <= covered_turn_count:
            return target_turn_count, []
        return target_turn_count, self._session_state.turns[
            covered_turn_count:target_turn_count
        ]

    def _compact_context_if_needed(self, *, force_all_prior: bool) -> bool:
        covered_turn_count = (
            self._context_summary.covered_turn_count
            if self._context_summary is not None
            else 0
        )
        target_turn_count, turns_to_compact = self._context_turns_to_compact(
            force_all_prior=force_all_prior
        )
        if not turns_to_compact:
            return False
        now = datetime.now(UTC).isoformat().replace("+00:00", "Z")
        summary = self._build_context_summary(
            previous_summary=self._context_summary,
            turns=turns_to_compact,
            starting_turn_number=covered_turn_count + 1,
        )
        self._context_summary = ChatContextSummary(
            content=summary,
            covered_turn_count=target_turn_count,
            created_at=(
                self._context_summary.created_at
                if self._context_summary is not None
                else now
            ),
            updated_at=now,
            compaction_count=(
                (self._context_summary.compaction_count if self._context_summary else 0)
                + 1
            ),
        )
        self._active_context_start_turn = max(
            self._active_context_start_turn,
            target_turn_count,
        )
        self._context_warning = (
            "Older turns were compacted into a summary to stay within the "
            "configured context limit."
        )
        self._refresh_prepared_context()
        return True

    def _build_context_summary(
        self,
        *,
        previous_summary: ChatContextSummary | None,
        turns: list[ChatSessionTurnRecord],
        starting_turn_number: int,
    ) -> str:
        fallback_summary = self._fallback_context_summary(
            previous_summary=previous_summary,
            turns=turns,
            starting_turn_number=starting_turn_number,
        )
        run_text = getattr(self._provider, "run_text", None)
        if callable(run_text):
            try:
                summary = str(
                    run_text(
                        messages=self._context_summary_prompt_messages(
                            previous_summary=previous_summary,
                            turns=turns,
                            starting_turn_number=starting_turn_number,
                        ),
                        request_params={"temperature": 0.0},
                    )
                ).strip()
                if summary:
                    return self._protect_context_summary(summary)
            except Exception:
                fallback_summary = fallback_summary.strip()
        return self._protect_context_summary(fallback_summary)

    @staticmethod
    def _context_summary_prompt_messages(
        *,
        previous_summary: ChatContextSummary | None,
        turns: list[ChatSessionTurnRecord],
        starting_turn_number: int,
    ) -> list[dict[str, str]]:
        previous = (
            previous_summary.content
            if previous_summary is not None
            else "No previous summary."
        )
        return [
            {
                "role": "system",
                "content": (
                    "Summarize older conversation turns for future chat context. "
                    "Keep durable user preferences, unresolved questions, important "
                    "facts, tool evidence, file paths, decisions, and caveats. "
                    "Do not add facts not present in the supplied transcript. "
                    "Return only the compact summary text."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Previous summary:\n{previous}\n\n"
                    "New turns to merge:\n"
                    f"{ChatSessionTurnRunner._render_turns_for_summary(turns, starting_turn_number)}"
                ),
            },
        ]

    @classmethod
    def _fallback_context_summary(
        cls,
        *,
        previous_summary: ChatContextSummary | None,
        turns: list[ChatSessionTurnRecord],
        starting_turn_number: int,
    ) -> str:
        sections: list[str] = []
        if previous_summary is not None:
            sections.append(f"Previous summary:\n{previous_summary.content}")
        sections.append(cls._render_turns_for_summary(turns, starting_turn_number))
        return cls._truncate_summary("\n\n".join(sections))

    @classmethod
    def _render_turns_for_summary(
        cls,
        turns: list[ChatSessionTurnRecord],
        starting_turn_number: int,
    ) -> str:
        lines: list[str] = []
        for offset, turn in enumerate(turns):
            turn_number = starting_turn_number + offset
            lines.append(f"Turn {turn_number} ({turn.status}):")
            for message in turn.new_messages:
                if message.role == "user":
                    lines.append(f"- User: {cls._compact_text(message.content)}")
                elif message.role == "assistant":
                    summary = cls._assistant_message_summary(message.content)
                    lines.append(
                        f"- Assistant: {cls._compact_text(summary or message.content)}"
                    )
                elif message.role == "tool":
                    lines.append(f"- Tool result: {cls._compact_text(message.content)}")
            if turn.final_response is not None:
                lines.append(
                    f"- Final answer: {cls._compact_text(turn.final_response.answer)}"
                )
            for tool_result in turn.tool_results:
                lines.append(f"- Tool call: {cls._tool_call_summary(tool_result)}")
        return "\n".join(lines)

    @staticmethod
    def _compact_text(text: str, *, max_chars: int = 700) -> str:
        compacted = " ".join(text.split())
        if len(compacted) <= max_chars:
            return compacted
        return f"{compacted[:max_chars]}...(truncated)"

    @staticmethod
    def _truncate_summary(summary: str, *, max_chars: int = 12000) -> str:
        stripped = summary.strip()
        if len(stripped) <= max_chars:
            return stripped
        return f"{stripped[:max_chars]}...(truncated)"

    def _protect_context_summary(self, summary: str) -> str:
        cleaned = self._truncate_summary(summary)
        if self._protection_controller is None:
            return cleaned
        decision = self._protection_controller.review_response(
            response_payload=ChatFinalResponse(answer=cleaned).model_dump(mode="json"),
            provenance=_collect_chat_provenance(
                session_state=self._session_state,
                current_tool_results=[],
            ),
        )
        if (
            decision.action is ProtectionAction.SANITIZE
            and decision.sanitized_payload is not None
        ):
            sanitized = decision.sanitized_payload.get("answer")
            if isinstance(sanitized, str) and sanitized.strip():
                return self._truncate_summary(sanitized)
        if decision.action is ProtectionAction.BLOCK:
            return (
                decision.safe_message
                or "Earlier context summary was withheld for this environment."
            )
        return cleaned

    def _serialize_messages_for_model(
        self,
        messages: list[ChatMessage],
        *,
        force_final: bool = False,
    ) -> list[dict[str, Any]]:
        protocol = self._model_visible_protocol(force_final=force_final)
        return [
            self._serialize_chat_message_for_protocol(message, protocol=protocol)
            for message in messages
        ]

    def _model_visible_protocol(self, *, force_final: bool = False) -> str:
        if force_final:
            return "final_only"
        if self._uses_prompt_tool_protocol():
            return "prompt_tools"
        if self._uses_staged_schema_protocol():
            return "staged_json"
        return "native_tools"

    def _serialize_chat_message_for_protocol(
        self,
        message: ChatMessage,
        *,
        protocol: str,
    ) -> dict[str, str]:
        if message.role != "assistant":
            return _serialize_chat_message(message)
        if protocol not in {
            "prompt_tools",
            "staged_json",
            "native_tools",
            "final_only",
        }:
            return _serialize_chat_message(message)
        summary = self._assistant_message_summary(message.content)
        if summary is None:
            return _serialize_chat_message(message)
        return {"role": "assistant", "content": summary}

    @classmethod
    def _assistant_message_summary(cls, content: str) -> str | None:
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            return None
        if not isinstance(payload, dict):
            return None
        invocations = payload.get("actions")
        final_response = payload.get("final_response")
        if isinstance(invocations, list) and invocations:
            lines = []
            for invocation in invocations:
                if not isinstance(invocation, dict):
                    continue
                tool_name = invocation.get("tool_name")
                arguments = invocation.get("arguments")
                if not isinstance(tool_name, str):
                    continue
                if not isinstance(arguments, dict):
                    arguments = {}
                lines.append(
                    "Tool call audit metadata, not evidence and not an answer: "
                    f"{tool_name}({cls._compact_json(arguments)})."
                )
            return "\n".join(lines) if lines else None
        if final_response is not None:
            if isinstance(final_response, dict):
                answer = final_response.get("answer")
                if isinstance(answer, str) and answer.strip():
                    return f"Assistant final answer: {answer}"
            if isinstance(final_response, str) and final_response.strip():
                return f"Assistant final answer: {final_response}"
        return None

    def _decision_tool_use_context(
        self,
        *,
        round_count: int,
        executed_tool_call_count: int,
        tool_results: list[ToolResult],
    ) -> str:
        max_rounds = self._session_config.max_tool_round_trips
        remaining_before_decision = max(max_rounds - round_count, 0)
        remaining_after_tool = max(max_rounds - round_count - 1, 0)
        max_total_calls = self._session_config.max_total_tool_calls_per_turn
        remaining_total_calls = max(max_total_calls - executed_tool_call_count, 0)
        lines = [
            "Tool budget for this turn:",
            f"- Tool rounds used: {round_count} of {max_rounds}.",
            (
                "- Tool rounds remaining before this decision: "
                f"{remaining_before_decision}."
            ),
            (
                "- Tool rounds remaining after choosing a tool now: "
                f"{remaining_after_tool}."
            ),
            (
                "- Total tool calls used: "
                f"{executed_tool_call_count} of {max_total_calls}."
            ),
            f"- Total tool calls remaining: {remaining_total_calls}.",
            "",
            "Tool calls already made this turn:",
        ]
        if tool_results:
            for index, tool_result in enumerate(tool_results, start=1):
                lines.append(f"{index}. {self._tool_call_summary(tool_result)}")
        else:
            lines.append("- None yet.")
        lines.extend(
            [
                "",
                "Decision guidance:",
                "- Choose the next necessary action.",
                (
                    "- Do not repeat an exact prior successful tool call unless "
                    "there is a concrete reason the same call will now return "
                    "different information."
                ),
                "- Compare tool arguments against the prior calls before choosing a tool.",
                (
                    "- It is fine to call the same tool again with different "
                    "arguments when that gathers new evidence."
                ),
                (
                    "- Finalize only when the available evidence is sufficient "
                    "for the requested answer."
                ),
            ]
        )
        return "\n".join(lines)

    @classmethod
    def _tool_call_summary(cls, tool_result: ToolResult) -> str:
        execution_record = tool_result.metadata.get("execution_record")
        request = (
            execution_record.get("request")
            if isinstance(execution_record, dict)
            else None
        )
        request_payload = request if isinstance(request, dict) else {}
        tool_name = request_payload.get("tool_name")
        if not isinstance(tool_name, str) or not tool_name.strip():
            tool_name = tool_result.tool_name
        arguments = request_payload.get("arguments")
        if not isinstance(arguments, dict):
            arguments = {}
        status = "success" if tool_result.ok else "error"
        if tool_result.error is not None:
            status = f"error:{tool_result.error.code.value}"
        return f"{tool_name}({cls._compact_json(arguments)}) -> {status}"

    @staticmethod
    def _native_tool_round_messages(
        *,
        messages: list[dict[str, Any]],
        decision_context: str,
    ) -> list[dict[str, Any]]:
        return [
            *messages,
            {
                "role": "system",
                "content": (
                    "Current native-tool step.\n"
                    f"{decision_context}\n\n"
                    "Use another native tool call only if it is necessary and within budget. "
                    "If available evidence is sufficient, return the final answer now."
                ),
            },
        ]

    @staticmethod
    def _prompt_tool_base_messages(
        messages: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            *messages,
            {
                "role": "system",
                "content": (
                    "Switching to prompt-tool protocol. Ignore earlier native tool-call "
                    "or structured JSON response instructions. Follow only the fenced "
                    "prompt-tool instructions in the next system message."
                ),
            },
        ]

    @staticmethod
    def _compact_json(payload: object, *, max_chars: int = 500) -> str:
        rendered = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        )
        if len(rendered) <= max_chars:
            return rendered
        return f"{rendered[:max_chars]}...(truncated)"

    def _run_model_turn_protocol(
        self,
        *,
        round_index: int,
        messages: list[dict[str, Any]],
        prepared: PreparedModelInteraction,
        decision_context: str | None,
        provenance: ProtectionProvenanceSnapshot,
        mode: str,
    ) -> Generator[
        ChatWorkflowStatusEvent | ChatWorkflowInspectorEvent | ChatWorkflowResultEvent,
        None,
        ParsedModelResponse,
    ]:
        events: list[ModelTurnProtocolEvent] = []
        self._protocol_metadata = {}
        parsed = ModelTurnProtocolRunner().run(
            ModelTurnProtocolRequest(
                provider=self._provider,
                messages=messages,
                prepared_interaction=prepared,
                adapter=self._adapter,
                final_response_model=ChatFinalResponse,
                temperature=self._temperature,
                decision_context=decision_context,
                protection=(
                    ModelTurnProtectionContext(
                        controller=self._protection_controller,
                        provenance=provenance,
                        metadata_sink=self._protocol_metadata,
                        original_user_message=self._user_chat_message.content,
                        session_id=self._base_context.invocation_id,
                    )
                    if self._protection_controller is not None
                    else None
                ),
                redaction_config=self._redaction_config,
                mode=cast(Any, mode),
            ),
            observer=events.append,
        )
        for event in events:
            chat_event = self._chat_event_from_protocol_event(event, round_index)
            if chat_event is not None:
                yield chat_event
        return parsed

    @staticmethod
    def _chat_event_from_protocol_event(
        event: ModelTurnProtocolEvent,
        round_index: int,
    ) -> ChatWorkflowStatusEvent | ChatWorkflowInspectorEvent | None:
        status = event.payload.get("status")
        if isinstance(status, str) and status.strip():
            return ChatWorkflowStatusEvent(status=status)
        if event.kind == "fallback":
            return ChatWorkflowStatusEvent(status="using prompt tools")
        if event.stage_name == "category_selected":
            category = event.payload.get("category")
            if isinstance(category, str) and category.strip():
                return ChatWorkflowStatusEvent(status=f"using {category} tools")
        if event.kind == "stage_start" and event.stage_name != "prepare_tool":
            return ChatWorkflowInspectorEvent(
                round_index=round_index,
                kind="provider_messages",
                payload={
                    "protocol": event.protocol,
                    "stage_name": event.stage_name,
                    "message_count": event.payload.get("message_count"),
                },
            )
        return None

    def _uses_staged_schema_protocol(self) -> bool:
        preference = getattr(
            self._provider,
            "uses_staged_schema_protocol",
            None,
        )
        if not callable(preference):
            return False
        return bool(preference())

    def _uses_prompt_tool_protocol(self) -> bool:
        preference = getattr(
            self._provider,
            "uses_prompt_tool_protocol",
            None,
        )
        if not callable(preference):
            return False
        return bool(preference())

    def _uses_json_single_action_strategy(self) -> bool:
        strategy_value = getattr(self._provider, "json_agent_strategy", None)
        if callable(strategy_value):
            strategy = str(strategy_value())
        elif isinstance(strategy_value, str):
            strategy = strategy_value
        else:
            strategy = os.environ.get(_JSON_AGENT_STRATEGY_ENV, "single_action")
        return strategy.strip().lower() not in {"staged", "split", "decision"}

    def _can_fallback_to_prompt_tools(self, exc: Exception) -> bool:
        if self._is_context_limit_error(exc):
            return False
        run_text = getattr(self._provider, "run_text", None)
        if not callable(run_text):
            return False
        can_fallback = getattr(self._provider, "can_fallback_to_prompt_tools", None)
        if not callable(can_fallback):
            return False
        return bool(can_fallback(exc))

    def _should_retry_after_context_limit(self, exc: Exception) -> bool:
        if self._context_limit_retry_used:
            return False
        if not self._is_context_limit_error(exc):
            return False
        self._context_limit_retry_used = True
        return True

    @staticmethod
    def _is_context_limit_error(exc: Exception) -> bool:
        for candidate in _iter_exception_chain(exc):
            message = str(candidate).lower()
            if any(marker in message for marker in _CONTEXT_LIMIT_ERROR_MARKERS):
                return True
            status_code = getattr(candidate, "status_code", None)
            if status_code == 400 and any(
                marker in message for marker in ("token", "context", "prompt")
            ):
                return True
        return False

    def _structured_provider(self) -> ModelTurnProvider:
        run_structured = getattr(self._provider, "run_structured", None)
        if not callable(run_structured):
            raise RuntimeError(
                "Staged schema protocol requires a provider that supports run_structured()."
            )
        return self._provider

    def _force_final_response_at_limit(
        self,
        *,
        round_index: int,
        messages: list[ChatMessage],
        new_messages: list[ChatMessage],
        tool_results: list[ToolResult],
    ) -> Generator[
        ChatWorkflowStatusEvent | ChatWorkflowInspectorEvent | ChatWorkflowResultEvent,
        None,
        ChatWorkflowResultEvent | None,
    ]:
        """Force a final answer from gathered evidence when no tool rounds remain."""
        if not tool_results:
            return None
        serialized_messages = self._serialize_messages_for_model(
            messages,
            force_final=True,
        )
        serialized_messages.append(
            {
                "role": "system",
                "content": (
                    "The tool round budget for this turn is exhausted. "
                    "Do not call any more tools. Produce the best final answer "
                    "using only the evidence already available in the transcript. "
                    "Mention uncertainty or missing information when the evidence "
                    "is incomplete."
                ),
            }
        )
        provenance = _collect_chat_provenance(
            session_state=self._session_state,
            current_tool_results=tool_results,
        )
        yield ChatWorkflowStatusEvent(status="drafting answer")
        parsed = yield from self._run_model_turn_protocol(
            round_index=round_index,
            messages=serialized_messages,
            prepared=self._executor.prepare_model_interaction(
                self._adapter,
                context=self._base_context,
                include_requires_approval=True,
                final_response_model=ChatFinalResponse,
            ),
            decision_context=None,
            provenance=provenance,
            mode="final_response",
        )
        yield ChatWorkflowInspectorEvent(
            round_index=round_index,
            kind="parsed_response",
            payload=_sanitize_parsed_response_for_inspector(
                parsed,
                redaction_config=self._redaction_config,
            ),
        )
        self._append_assistant_message(
            messages=messages,
            new_messages=new_messages,
            parsed=parsed,
        )
        return self._maybe_finalize_from_response(
            parsed=parsed,
            new_messages=new_messages,
            tool_results=tool_results,
        )

    @staticmethod
    def _uses_prompt_tool_single_action_strategy() -> bool:
        return ModelTurnProtocolRunner._uses_prompt_tool_single_action_strategy()

    @staticmethod
    def _prompt_tool_category_threshold() -> int:
        return ModelTurnProtocolRunner._prompt_tool_category_threshold()

    @staticmethod
    def _validation_error_summary(error: Exception) -> str:
        return validation_error_summary(error)

    @staticmethod
    def _format_invalid_payload(invalid_payload: object | None) -> str:
        return format_invalid_payload(invalid_payload)

    def _append_assistant_message(
        self,
        *,
        messages: list[ChatMessage],
        new_messages: list[ChatMessage],
        parsed: ParsedModelResponse,
    ) -> None:
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

    def _maybe_finalize_from_response(
        self,
        *,
        parsed: ParsedModelResponse,
        new_messages: list[ChatMessage],
        tool_results: list[ToolResult],
    ) -> ChatWorkflowResultEvent | None:
        if parsed.final_response is None:
            return None
        final_response = ChatFinalResponse.model_validate(parsed.final_response)
        pending_prompt = self._protocol_metadata.get("pending_protection_prompt")
        return self._finalized_event(
            result=ChatWorkflowTurnResult(
                status="completed",
                new_messages=new_messages,
                final_response=final_response,
                token_usage=None,
                tool_results=tool_results,
                pending_protection_prompt=cast(Any, pending_prompt),
            )
        )

    def _should_retry_ungrounded_final_response(
        self,
        *,
        parsed: ParsedModelResponse,
        tool_results: list[ToolResult],
    ) -> bool:
        if self._grounding_retry_used:
            return False
        if parsed.final_response is None or parsed.invocations:
            return False
        if tool_results:
            return False
        if self._base_context.workspace is None:
            return False
        if not (self._enabled_tool_names & _LOCAL_GROUNDING_TOOL_NAMES):
            return False
        request = self._user_chat_message.content.lower()
        if any(token in request for token in _EXPLICIT_GROUNDING_HINTS):
            return True
        if any(token in request for token in _REPOSITORY_GROUNDING_HINTS):
            return True
        return bool(self._enabled_tool_names & _GIT_GROUNDING_TOOL_NAMES) and any(
            token in request for token in _GIT_GROUNDING_HINTS
        )

    def _grounding_repair_message(self) -> ChatMessage:
        preferred_tools = sorted(self._enabled_tool_names & _LOCAL_GROUNDING_TOOL_NAMES)
        tool_list = ", ".join(preferred_tools[:6]) or "local workspace or git tools"
        return ChatMessage(
            role="system",
            content=(
                "This request requires local workspace or git evidence before a "
                "final answer. Your previous response skipped that evidence. "
                "On the next response, call at least one relevant local tool first "
                f"and do not return final_response until tool results are available. Prefer: {tool_list}."
            ),
        )

    def _continuation_limit_event(
        self,
        *,
        parsed: ParsedModelResponse,
        new_messages: list[ChatMessage],
        tool_results: list[ToolResult],
        executed_tool_call_count: int,
    ) -> ChatWorkflowResultEvent | None:
        if len(parsed.invocations) > self._session_config.max_tool_calls_per_round:
            return self._finalized_event(
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
        if (
            executed_tool_call_count + len(parsed.invocations)
            <= self._session_config.max_total_tool_calls_per_turn
        ):
            return None
        return self._finalized_event(
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

    def _round_trip_limit_event(
        self,
        *,
        round_count: int,
        new_messages: list[ChatMessage],
        tool_results: list[ToolResult],
    ) -> ChatWorkflowResultEvent | None:
        if round_count < self._session_config.max_tool_round_trips:
            return None
        return self._finalized_event(
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

    def _consume_workflow_result(
        self,
        *,
        workflow_result: WorkflowTurnResult,
        round_index: int,
        messages: list[ChatMessage],
        new_messages: list[ChatMessage],
        tool_results: list[ToolResult],
        executed_tool_call_count_ref: list[int],
    ) -> Iterator[
        ChatWorkflowStatusEvent
        | ChatWorkflowApprovalEvent
        | ChatWorkflowApprovalResolvedEvent
        | ChatWorkflowInspectorEvent
        | ChatWorkflowResultEvent
    ]:
        for outcome in workflow_result.outcomes:
            if self._cancel_requested:
                yield self._interrupted_event(
                    new_messages=new_messages,
                    tool_results=tool_results,
                    reason="Interrupted by user.",
                )
                return

            if (
                outcome.status is WorkflowInvocationStatus.EXECUTED
                and outcome.tool_result is not None
            ):
                sanitized_tool_result = _sanitize_tool_result_for_chat(
                    outcome.tool_result
                )
                yield ChatWorkflowStatusEvent(
                    status=_tool_status_label(outcome.request.tool_name)
                )
                tool_results.append(sanitized_tool_result)
                executed_tool_call_count_ref[0] += 1
                execution_record = _sanitize_execution_record(
                    sanitized_tool_result.metadata.get("execution_record")
                )
                if execution_record is not None:
                    yield ChatWorkflowInspectorEvent(
                        round_index=round_index,
                        kind="tool_execution",
                        payload=execution_record,
                    )
                tool_message = ChatMessage(
                    role="tool",
                    content=_format_tool_result_for_model(
                        _sanitize_tool_result_message(sanitized_tool_result),
                        max_chars=self._tool_limits.max_tool_result_chars,
                    ),
                )
                messages.append(tool_message)
                new_messages.append(tool_message)
                continue

            if (
                outcome.status is WorkflowInvocationStatus.APPROVAL_REQUESTED
                and outcome.approval_request is not None
            ):
                approval_state = _build_approval_state(
                    outcome.approval_request,
                    redaction_config=self._redaction_config,
                )
                self._set_pending_approval(approval_state)
                yield ChatWorkflowApprovalEvent(approval=approval_state)
                resolution = self._wait_for_approval_resolution(
                    outcome.approval_request
                )
                yield ChatWorkflowApprovalResolvedEvent(
                    approval=approval_state,
                    resolution=resolution,
                )
                self._clear_pending_approval()

                if resolution == "cancelled":
                    if outcome.approval_request.approval_id:
                        self._executor.cancel_pending_approval(
                            outcome.approval_request.approval_id
                        )
                    yield self._interrupted_event(
                        new_messages=new_messages,
                        tool_results=tool_results,
                        reason="Interrupted while waiting for approval.",
                    )
                    return

                if resolution == "timed_out":
                    finalized = self._executor.finalize_expired_approvals(
                        now=datetime.now(UTC)
                    )
                    if finalized:
                        yield from self._consume_workflow_result(
                            workflow_result=finalized[0],
                            round_index=round_index,
                            messages=messages,
                            new_messages=new_messages,
                            tool_results=tool_results,
                            executed_tool_call_count_ref=executed_tool_call_count_ref,
                        )
                    return

                resumed = self._executor.resolve_pending_approval(
                    outcome.approval_request.approval_id,
                    approved=resolution == "approved",
                )
                yield from self._consume_workflow_result(
                    workflow_result=resumed,
                    round_index=round_index,
                    messages=messages,
                    new_messages=new_messages,
                    tool_results=tool_results,
                    executed_tool_call_count_ref=executed_tool_call_count_ref,
                )
                return

    def _set_pending_approval(self, approval: ChatWorkflowApprovalState) -> None:
        with self._approval_condition:
            self._pending_approval = approval
            self._pending_approval_decision = None

    def _clear_pending_approval(self) -> None:
        with self._approval_condition:
            self._pending_approval = None
            self._pending_approval_decision = None

    def _wait_for_approval_resolution(
        self,
        approval_request: ApprovalRequest,
    ) -> ChatApprovalResolution:
        expires_at = _parse_timestamp(approval_request.expires_at)
        with self._approval_condition:
            while True:
                if self._cancel_requested:
                    return "cancelled"
                if self._pending_approval_decision is not None:
                    return "approved" if self._pending_approval_decision else "denied"
                remaining_seconds = (expires_at - datetime.now(UTC)).total_seconds()
                if remaining_seconds <= 0:
                    return "timed_out"
                self._approval_condition.wait(timeout=remaining_seconds)

    def _interrupted_event(
        self,
        *,
        new_messages: list[ChatMessage],
        tool_results: list[ToolResult],
        reason: str,
    ) -> ChatWorkflowResultEvent:
        return self._finalized_event(
            result=_build_interrupted_result(
                new_messages=new_messages,
                tool_results=tool_results,
                token_usage=None,
                reason=reason,
            )
        )

    def _maybe_handle_pending_protection_prompt(
        self,
        *,
        new_messages: list[ChatMessage],
        tool_results: list[ToolResult],
    ) -> ChatWorkflowResultEvent | None:
        if self._protection_controller is None:
            return None
        pending_prompt = self._session_state.pending_protection_prompt
        if pending_prompt is None:
            return None

        parsed_feedback = self._protection_controller.parse_feedback_message(
            self._user_chat_message.content
        )
        if parsed_feedback is None:
            final_response = ChatFinalResponse(
                answer=self._protection_controller.feedback_required_message()
            )
            assistant_message = _assistant_message_for_final_response(final_response)
            return self._finalized_event(
                result=ChatWorkflowTurnResult(
                    status="completed",
                    new_messages=[*new_messages, assistant_message],
                    final_response=final_response,
                    token_usage=None,
                    tool_results=tool_results,
                    pending_protection_prompt=pending_prompt,
                )
            )

        if parsed_feedback.analysis_is_correct:
            final_response = ChatFinalResponse(
                answer=self._protection_controller.confirmation_block_message()
            )
            assistant_message = _assistant_message_for_final_response(final_response)
            return self._finalized_event(
                result=ChatWorkflowTurnResult(
                    status="completed",
                    new_messages=[*new_messages, assistant_message],
                    final_response=final_response,
                    token_usage=None,
                    tool_results=tool_results,
                )
            )

        feedback_entry = self._protection_controller.record_feedback(
            pending_prompt=pending_prompt,
            feedback_prompt=parsed_feedback,
        )
        final_response = ChatFinalResponse(
            answer=self._protection_controller.correction_recorded_message(
                expected_sensitivity_label=feedback_entry.expected_sensitivity_label,
            )
        )
        assistant_message = _assistant_message_for_final_response(final_response)
        return self._finalized_event(
            result=ChatWorkflowTurnResult(
                status="completed",
                new_messages=[*new_messages, assistant_message],
                final_response=final_response,
                token_usage=None,
                tool_results=tool_results,
            )
        )

    @staticmethod
    def _executed_tool_call_count(*, tool_results: list[ToolResult]) -> int:
        return len(tool_results)


def _iter_exception_chain(exc: Exception) -> list[BaseException]:
    seen: set[int] = set()
    pending: list[BaseException] = [exc]
    chain: list[BaseException] = []
    while pending:
        current = pending.pop()
        current_id = id(current)
        if current_id in seen:
            continue
        seen.add(current_id)
        chain.append(current)
        cause = getattr(current, "__cause__", None)
        if isinstance(cause, BaseException):
            pending.append(cause)
        context = getattr(current, "__context__", None)
        if isinstance(context, BaseException):
            pending.append(context)
    return chain


def run_interactive_chat_session_turn(
    *,
    user_message: str,
    session_state: ChatSessionState,
    executor: WorkflowExecutor,
    provider: ModelTurnProvider,
    system_prompt: str,
    base_context: ToolContext,
    session_config: ChatSessionConfig,
    tool_limits: ToolLimits,
    redaction_config: RedactionConfig,
    temperature: float,
    protection_controller: ProtectionController | None = None,
    enabled_tool_names: set[str] | None = None,
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
        redaction_config=redaction_config,
        temperature=temperature,
        protection_controller=protection_controller,
        enabled_tool_names=enabled_tool_names,
    )
