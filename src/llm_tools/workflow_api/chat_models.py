"""Typed models for interactive repository chat workflows."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator

from llm_tools.tool_api import ToolResult
from llm_tools.workflow_api.models import ApprovalRequest
from llm_tools.workflow_api.protection import ProtectionPendingPrompt

ChatWorkflowTurnStatus = Literal["completed", "needs_continuation", "interrupted"]
ChatMessageRole = Literal["system", "user", "assistant", "tool"]
ChatAssistantCompletionState = Literal["complete", "interrupted"]
ChatApprovalResolution = Literal[
    "approved",
    "denied",
    "timed_out",
    "cancelled",
]
ChatWorkflowInspectorKind = Literal[
    "provider_messages",
    "parsed_response",
    "tool_execution",
]


class ChatSessionConfig(BaseModel):
    """Per-session context and tool-call safety limits."""

    max_context_tokens: int = 24000
    max_tool_round_trips: int = 8
    max_tool_calls_per_round: int = 4
    max_total_tool_calls_per_turn: int = 12

    @field_validator(
        "max_context_tokens",
        "max_tool_round_trips",
        "max_tool_calls_per_round",
        "max_total_tool_calls_per_turn",
    )
    @classmethod
    def validate_positive_ints(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("chat session limits must be positive integers")
        return value


class ChatCitation(BaseModel):
    """Grounding citation displayed with a final chat answer."""

    source_path: str
    line_start: int | None = Field(default=None, ge=1)
    line_end: int | None = Field(default=None, ge=1)
    excerpt: str | None = None

    @field_validator("source_path")
    @classmethod
    def validate_source_path(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("source_path must not be empty")
        return cleaned

    @model_validator(mode="after")
    def validate_line_range(self) -> ChatCitation:
        if (
            self.line_start is not None
            and self.line_end is not None
            and self.line_end < self.line_start
        ):
            raise ValueError("line_end must be greater than or equal to line_start")
        return self


class ChatFinalResponse(BaseModel):
    """Validated final answer emitted by the interactive chat loop."""

    answer: str
    citations: list[ChatCitation] = Field(default_factory=list)
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    uncertainty: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    follow_up_suggestions: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def validate_answer(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("answer must not be empty")
        return value

    @field_validator("uncertainty", "missing_information", "follow_up_suggestions")
    @classmethod
    def validate_string_lists(cls, value: list[str]) -> list[str]:
        cleaned = [entry.strip() for entry in value]
        if any(not entry for entry in cleaned):
            raise ValueError("final response list entries must not be empty")
        return cleaned


class ChatTokenUsage(BaseModel):
    """Token-usage accounting for one turn or session summary."""

    input_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    total_tokens: int | None = Field(default=None, ge=0)
    session_tokens: int | None = Field(default=None, ge=0)
    active_context_tokens: int | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def populate_total_tokens(self) -> ChatTokenUsage:
        derived_total = self.input_tokens + self.output_tokens
        if self.total_tokens is None:
            self.total_tokens = derived_total
        elif self.total_tokens < derived_total:
            raise ValueError(
                "total_tokens cannot be less than input_tokens + output_tokens"
            )
        return self


class ChatMessage(BaseModel):
    """Provider-neutral chat message envelope."""

    role: ChatMessageRole
    content: str
    completion_state: ChatAssistantCompletionState = "complete"

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("chat message content must not be empty")
        return value

    @model_validator(mode="after")
    def validate_completion_state(self) -> ChatMessage:
        if self.role != "assistant" and self.completion_state != "complete":
            raise ValueError(
                "Only assistant messages may use non-default completion_state"
            )
        return self


class ChatSessionTurnRecord(BaseModel):
    """One stored visible chat turn in the in-memory session."""

    status: ChatWorkflowTurnStatus
    new_messages: list[ChatMessage] = Field(default_factory=list)
    final_response: ChatFinalResponse | None = None
    token_usage: ChatTokenUsage | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    continuation_reason: str | None = None
    interruption_reason: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> ChatSessionTurnRecord:
        if self.status == "completed":
            if self.final_response is None:
                raise ValueError("completed chat turns require final_response")
            if (
                self.continuation_reason is not None
                or self.interruption_reason is not None
            ):
                raise ValueError(
                    "completed chat turns do not allow continuation/interruption reasons"
                )
            return self

        if self.status == "needs_continuation":
            if self.final_response is not None:
                raise ValueError(
                    "needs_continuation chat turns do not allow final_response"
                )
            if not (self.continuation_reason or "").strip():
                raise ValueError(
                    "needs_continuation chat turns require continuation_reason"
                )
            if self.interruption_reason is not None:
                raise ValueError(
                    "needs_continuation chat turns do not allow interruption_reason"
                )
            return self

        if self.final_response is not None:
            raise ValueError("interrupted chat turns do not allow final_response")
        if self.continuation_reason is not None:
            raise ValueError("interrupted chat turns do not allow continuation_reason")
        if not (self.interruption_reason or "").strip():
            raise ValueError("interrupted chat turns require interruption_reason")
        return self


class ChatSessionState(BaseModel):
    """In-memory visible chat history plus active-context window metadata."""

    turns: list[ChatSessionTurnRecord] = Field(default_factory=list)
    active_context_start_turn: int = Field(default=0, ge=0)
    pending_protection_prompt: ProtectionPendingPrompt | None = None

    @model_validator(mode="after")
    def validate_active_context_start_turn(self) -> ChatSessionState:
        if self.active_context_start_turn > len(self.turns):
            raise ValueError(
                "active_context_start_turn cannot be greater than the number of turns"
            )
        return self


class ChatWorkflowTurnResult(BaseModel):
    """Provider-neutral result for one orchestrated chat turn attempt."""

    status: ChatWorkflowTurnStatus
    new_messages: list[ChatMessage] = Field(default_factory=list)
    final_response: ChatFinalResponse | None = None
    token_usage: ChatTokenUsage | None = None
    tool_results: list[ToolResult] = Field(default_factory=list)
    continuation_reason: str | None = None
    interruption_reason: str | None = None
    pending_protection_prompt: ProtectionPendingPrompt | None = None
    session_state: ChatSessionState | None = None
    context_warning: str | None = None

    @model_validator(mode="after")
    def validate_status_fields(self) -> ChatWorkflowTurnResult:
        if self.status == "completed":
            if self.final_response is None:
                raise ValueError("completed chat turns require final_response")
            if (
                self.continuation_reason is not None
                or self.interruption_reason is not None
            ):
                raise ValueError(
                    "completed chat turns do not allow continuation/interruption reasons"
                )
            return self

        if self.status == "needs_continuation":
            if self.final_response is not None:
                raise ValueError(
                    "needs_continuation chat turns do not allow final_response"
                )
            if not (self.continuation_reason or "").strip():
                raise ValueError(
                    "needs_continuation chat turns require continuation_reason"
                )
            if self.interruption_reason is not None:
                raise ValueError(
                    "needs_continuation chat turns do not allow interruption_reason"
                )
            return self

        if self.final_response is not None:
            raise ValueError("interrupted chat turns do not allow final_response")
        if self.continuation_reason is not None:
            raise ValueError("interrupted chat turns do not allow continuation_reason")
        if not (self.interruption_reason or "").strip():
            raise ValueError("interrupted chat turns require interruption_reason")
        return self


class ChatWorkflowStatusEvent(BaseModel):
    """One transient workflow status update for the UI."""

    event_type: Literal["status"] = "status"
    status: str


class ChatWorkflowApprovalState(BaseModel):
    """Redacted approval state surfaced to the interactive chat UI."""

    approval_request: ApprovalRequest
    tool_name: str
    redacted_arguments: dict[str, object] = Field(default_factory=dict)
    policy_reason: str
    policy_metadata: dict[str, object] = Field(default_factory=dict)


class ChatWorkflowApprovalEvent(BaseModel):
    """Non-terminal event announcing that a turn is waiting for approval."""

    event_type: Literal["approval_requested"] = "approval_requested"
    approval: ChatWorkflowApprovalState


class ChatWorkflowApprovalResolvedEvent(BaseModel):
    """Non-terminal event announcing approval resolution for the active turn."""

    event_type: Literal["approval_resolved"] = "approval_resolved"
    approval: ChatWorkflowApprovalState
    resolution: ChatApprovalResolution


class ChatWorkflowInspectorEvent(BaseModel):
    """Append-only inspector event emitted while a turn is running."""

    event_type: Literal["inspector"] = "inspector"
    round_index: int = Field(ge=1)
    kind: ChatWorkflowInspectorKind
    payload: object


class ChatWorkflowResultEvent(BaseModel):
    """Terminal workflow event carrying the completed turn result."""

    event_type: Literal["result"] = "result"
    result: ChatWorkflowTurnResult
