"""Interactive multi-turn repository chat orchestration."""

from __future__ import annotations

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from threading import Condition
from typing import Any, Protocol
from uuid import uuid4

from pydantic import BaseModel

from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import (
    ProtectionProvenanceSnapshot,
    ToolContext,
    ToolResult,
)
from llm_tools.tool_api.redaction import RedactionConfig, RedactionTarget, Redactor
from llm_tools.tools.filesystem._content import dump_json
from llm_tools.tools.filesystem.models import ToolLimits
from llm_tools.workflow_api.chat_models import (
    ChatApprovalResolution,
    ChatFinalResponse,
    ChatMessage,
    ChatSessionConfig,
    ChatSessionState,
    ChatSessionTurnRecord,
    ChatTokenUsage,
    ChatWorkflowApprovalEvent,
    ChatWorkflowApprovalResolvedEvent,
    ChatWorkflowApprovalState,
    ChatWorkflowInspectorEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
)
from llm_tools.workflow_api.executor import WorkflowExecutor
from llm_tools.workflow_api.models import (
    ApprovalRequest,
    WorkflowInvocationStatus,
    WorkflowTurnResult,
)
from llm_tools.workflow_api.protection import (
    ProtectionAction,
    ProtectionController,
    collect_provenance_from_tool_results,
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
        self._redaction_config = redaction_config
        self._temperature = temperature
        self._protection_controller = protection_controller
        self._adapter = ActionEnvelopeAdapter()
        self._approval_condition = Condition()
        self._cancel_requested = False
        self._pending_approval: ChatWorkflowApprovalState | None = None
        self._pending_approval_decision: bool | None = None

    def cancel(self) -> None:
        """Request cooperative cancellation at the next safe boundary."""
        with self._approval_condition:
            self._cancel_requested = True
            self._approval_condition.notify_all()

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
                context_warning=self._context_warning,
                system_message=self._system_message,
            )
        )

    def __iter__(
        self,
    ) -> Iterator[
        ChatWorkflowStatusEvent
        | ChatWorkflowApprovalEvent
        | ChatWorkflowApprovalResolvedEvent
        | ChatWorkflowInspectorEvent
        | ChatWorkflowResultEvent
    ]:
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

        pending_prompt_event = self._maybe_handle_pending_protection_prompt(
            new_messages=new_messages,
            tool_results=tool_results,
        )
        if pending_prompt_event is not None:
            yield pending_prompt_event
            return

        while True:
            if self._cancel_requested:
                yield self._interrupted_event(
                    new_messages=new_messages,
                    tool_results=tool_results,
                    reason="Interrupted by user.",
                )
                return

            (
                round_context,
                response_model,
                round_index,
                serialized_messages,
                provenance,
            ) = self._build_round_inputs(
                messages=messages,
                tool_results=tool_results,
                round_count=round_count,
            )
            serialized_messages, prompt_event = self._apply_prompt_protection(
                serialized_messages=serialized_messages,
                provenance=provenance,
                new_messages=new_messages,
                tool_results=tool_results,
            )
            if prompt_event is not None:
                yield prompt_event
                return

            yield ChatWorkflowInspectorEvent(
                round_index=round_index,
                kind="provider_messages",
                payload=serialized_messages,
            )
            parsed = self._provider.run(
                adapter=self._adapter,
                messages=serialized_messages,
                response_model=response_model,
                request_params={"temperature": self._temperature},
            )
            parsed = self._apply_response_protection(
                parsed=parsed,
                provenance=provenance,
            )
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
        type[BaseModel],
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
        serialized_messages = [_serialize_chat_message(message) for message in messages]
        provenance = _collect_chat_provenance(
            session_state=self._session_state,
            current_tool_results=tool_results,
        )
        return (
            round_context,
            prepared.response_model,
            round_index,
            serialized_messages,
            provenance,
        )

    def _apply_prompt_protection(
        self,
        *,
        serialized_messages: list[dict[str, Any]],
        provenance: ProtectionProvenanceSnapshot,
        new_messages: list[ChatMessage],
        tool_results: list[ToolResult],
    ) -> tuple[list[dict[str, Any]], ChatWorkflowResultEvent | None]:
        if self._protection_controller is None:
            return serialized_messages, None

        prompt_decision = self._protection_controller.assess_prompt(
            messages=serialized_messages,
            provenance=provenance,
        )
        if (
            prompt_decision.action is ProtectionAction.CONSTRAIN
            and prompt_decision.guard_text is not None
        ):
            return [
                {"role": "system", "content": prompt_decision.guard_text},
                *serialized_messages,
            ], None
        if prompt_decision.action not in {
            ProtectionAction.CHALLENGE,
            ProtectionAction.BLOCK,
        }:
            return serialized_messages, None

        challenge_response = ChatFinalResponse(
            answer=(
                prompt_decision.challenge_message
                or "The request was withheld for this environment."
            )
        )
        assistant_message = _assistant_message_for_final_response(challenge_response)
        pending_prompt = (
            self._protection_controller.build_pending_prompt(
                original_user_message=self._user_chat_message.content,
                serialized_messages=serialized_messages,
                decision=prompt_decision,
                session_id=self._base_context.invocation_id,
            )
            if prompt_decision.action is ProtectionAction.CHALLENGE
            else None
        )
        return serialized_messages, self._finalized_event(
            result=ChatWorkflowTurnResult(
                status="completed",
                new_messages=[*new_messages, assistant_message],
                final_response=challenge_response,
                token_usage=None,
                tool_results=tool_results,
                pending_protection_prompt=pending_prompt,
            )
        )

    def _apply_response_protection(
        self,
        *,
        parsed: ParsedModelResponse,
        provenance: ProtectionProvenanceSnapshot,
    ) -> ParsedModelResponse:
        if self._protection_controller is None or parsed.final_response is None:
            return parsed

        response_decision = self._protection_controller.review_response(
            response_payload=parsed.final_response,
            provenance=provenance,
        )
        if (
            response_decision.action is ProtectionAction.SANITIZE
            and response_decision.sanitized_payload is not None
        ):
            return parsed.model_copy(
                update={"final_response": response_decision.sanitized_payload}
            )
        if response_decision.action is not ProtectionAction.BLOCK:
            return parsed
        return parsed.model_copy(
            update={
                "final_response": ChatFinalResponse(
                    answer=(
                        response_decision.safe_message
                        or "The response was withheld for this environment."
                    )
                ).model_dump(mode="json")
            }
        )

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
        return self._finalized_event(
            result=ChatWorkflowTurnResult(
                status="completed",
                new_messages=new_messages,
                final_response=final_response,
                token_usage=None,
                tool_results=tool_results,
            )
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


def _sanitize_tool_result_for_chat(tool_result: ToolResult) -> ToolResult:
    metadata = dict(tool_result.metadata)
    execution_record = _sanitize_execution_record(metadata.get("execution_record"))
    if execution_record is not None:
        metadata["execution_record"] = execution_record
    return tool_result.model_copy(update={"metadata": metadata})


def _sanitize_execution_record(record: object) -> dict[str, object] | None:
    if not isinstance(record, dict):
        return None
    sanitized = dict(record)
    sanitized.pop("validated_input", None)
    sanitized.pop("validated_output", None)
    request = sanitized.get("request")
    redacted_input = sanitized.get("redacted_input")
    if isinstance(request, dict) and isinstance(redacted_input, dict):
        updated_request = dict(request)
        updated_request["arguments"] = redacted_input
        sanitized["request"] = updated_request
    return sanitized


def _sanitize_parsed_response_for_inspector(
    parsed_response: object,
    *,
    redaction_config: RedactionConfig,
) -> object:
    if not hasattr(parsed_response, "model_dump"):
        return parsed_response
    payload = parsed_response.model_dump(mode="json")
    actions = payload.get("invocations", [])
    if isinstance(actions, list):
        payload["invocations"] = [
            _sanitize_invocation_payload(action, redaction_config=redaction_config)
            for action in actions
        ]
    return payload


def _sanitize_invocation_payload(
    payload: object,
    *,
    redaction_config: RedactionConfig,
) -> object:
    if not isinstance(payload, dict):
        return payload
    tool_name = payload.get("tool_name")
    arguments = payload.get("arguments")
    if not isinstance(tool_name, str) or not isinstance(arguments, dict):
        return payload
    sanitized = dict(payload)
    sanitized["arguments"] = _redact_tool_payload(
        tool_name,
        arguments,
        target=RedactionTarget.INPUT,
        redaction_config=redaction_config,
    )
    return sanitized


def _build_approval_state(
    approval_request: ApprovalRequest,
    *,
    redaction_config: RedactionConfig,
) -> ChatWorkflowApprovalState:
    return ChatWorkflowApprovalState(
        approval_request=approval_request,
        tool_name=approval_request.tool_name,
        redacted_arguments=_redact_tool_payload(
            approval_request.tool_name,
            approval_request.request.arguments,
            target=RedactionTarget.INPUT,
            redaction_config=redaction_config,
        ),
        policy_reason=approval_request.policy_reason,
        policy_metadata=dict(approval_request.policy_metadata),
    )


def _redact_tool_payload(
    tool_name: str,
    payload: object,
    *,
    target: RedactionTarget,
    redaction_config: RedactionConfig,
) -> dict[str, object]:
    redactor = Redactor(redaction_config, tool_name=tool_name)
    redacted = redactor.redact_structured(payload, target=target)
    if not isinstance(redacted, dict):
        return {}
    return redacted


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
