"""Workflow-layer result models built on top of the core tool runtime."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field, model_validator

from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolInvocationRequest, ToolResult


class WorkflowInvocationStatus(str, Enum):  # noqa: UP042
    """Per-invocation outcome status for workflow execution."""

    EXECUTED = "executed"
    APPROVAL_REQUESTED = "approval_requested"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_TIMED_OUT = "approval_timed_out"


class ApprovalRequest(BaseModel):
    """Pending approval request emitted by workflow execution."""

    approval_id: str = Field(min_length=1)
    invocation_index: int = Field(ge=1)
    request: ToolInvocationRequest
    tool_name: str = Field(min_length=1)
    tool_version: str = Field(min_length=1)
    policy_reason: str
    policy_metadata: dict[str, object] = Field(default_factory=dict)
    requested_at: str
    expires_at: str


class WorkflowInvocationOutcome(BaseModel):
    """Outcome for one invocation in a workflow turn."""

    invocation_index: int = Field(ge=1)
    request: ToolInvocationRequest
    status: WorkflowInvocationStatus
    tool_result: ToolResult | None = None
    approval_request: ApprovalRequest | None = None

    @model_validator(mode="after")
    def validate_payload_shape(self) -> WorkflowInvocationOutcome:
        """Require status-specific payload fields."""
        if self.status is WorkflowInvocationStatus.EXECUTED:
            if self.tool_result is None:
                raise ValueError("Executed outcomes must include tool_result.")
            if self.approval_request is not None:
                raise ValueError("Executed outcomes must not include approval_request.")
            return self

        if self.approval_request is None:
            raise ValueError("Approval outcomes must include approval_request.")
        if self.tool_result is not None:
            raise ValueError("Approval outcomes must not include tool_result.")
        return self


class WorkflowTurnResult(BaseModel):
    """Normalized result of one parsed model turn."""

    parsed_response: ParsedModelResponse
    outcomes: list[WorkflowInvocationOutcome] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_consistency(self) -> WorkflowTurnResult:
        """Require result shape to match the parsed response mode."""
        if self.parsed_response.final_response is not None:
            if self.outcomes:
                raise ValueError(
                    "WorkflowTurnResult with final_response must not include outcomes."
                )
            return self

        invocation_count = len(self.parsed_response.invocations)
        last_index = 0
        for outcome in self.outcomes:
            if outcome.invocation_index > invocation_count:
                raise ValueError(
                    "Outcome invocation_index must reference an invocation in "
                    "parsed_response."
                )
            if outcome.invocation_index <= last_index:
                raise ValueError(
                    "Workflow outcomes must be ordered by ascending invocation_index."
                )
            last_index = outcome.invocation_index

        return self
