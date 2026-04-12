"""Thin workflow bridge for one model turn of parsing and execution."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from pydantic import BaseModel

from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import (
    PolicyVerdict,
    Tool,
    ToolContext,
    ToolInvocationRequest,
    ToolPolicy,
    ToolRegistry,
    ToolRuntime,
    ToolSpec,
)
from llm_tools.workflow_api.models import (
    ApprovalRequest,
    WorkflowInvocationOutcome,
    WorkflowInvocationStatus,
    WorkflowTurnResult,
)


@dataclass(slots=True)
class _PendingApprovalState:
    """In-memory state for one pending approval request."""

    approval_request: ApprovalRequest
    parsed_response: ParsedModelResponse
    base_context: ToolContext
    pending_index: int
    expires_at: datetime


@dataclass(slots=True, frozen=True)
class PreparedModelInteraction:
    """Prepared model-facing contract for one workflow turn."""

    response_model: type[BaseModel]
    schema: dict[str, Any]
    tool_names: list[str]


class WorkflowExecutor:
    """Bridge adapters and runtime for one parsed model turn."""

    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicy | None = None,
    ) -> None:
        self._registry = registry
        self._policy = policy or ToolPolicy()
        self._runtime = ToolRuntime(registry, policy=self._policy)
        self._pending_approvals: dict[str, _PendingApprovalState] = {}

    def export_tools(
        self,
        adapter: ActionEnvelopeAdapter,
        *,
        context: ToolContext | None = None,
        include_requires_approval: bool = False,
    ) -> object:
        """Export model-facing schema through the supplied action adapter."""
        prepared = self.prepare_model_interaction(
            adapter,
            context=context,
            include_requires_approval=include_requires_approval,
        )
        return prepared.schema

    def prepare_model_interaction(
        self,
        adapter: ActionEnvelopeAdapter,
        *,
        context: ToolContext | None = None,
        include_requires_approval: bool = False,
        final_response_model: object = str,
    ) -> PreparedModelInteraction:
        """Prepare a typed response model and schema for one model turn."""
        tools = self._registry.list_registered_tools()
        if context is not None:
            tools = self._filter_tools_for_exposure(
                tools,
                context=context,
                include_requires_approval=include_requires_approval,
            )
        specs: list[ToolSpec] = [tool.spec for tool in tools]
        input_models: dict[str, type[BaseModel]] = {
            tool.spec.name: tool.input_model for tool in tools
        }
        response_model = adapter.build_response_model(
            specs,
            input_models,
            final_response_model=final_response_model,
        )
        return PreparedModelInteraction(
            response_model=response_model,
            schema=adapter.export_schema(response_model),
            tool_names=[spec.name for spec in specs],
        )

    def execute_parsed_response(
        self,
        parsed_response: ParsedModelResponse,
        context: ToolContext,
    ) -> WorkflowTurnResult:
        """Execute parsed invocations or return a final-response-only result."""
        if parsed_response.final_response is not None:
            return WorkflowTurnResult(parsed_response=parsed_response)

        outcomes = self._execute_sequence(
            parsed_response=parsed_response,
            base_context=context,
            start_index=1,
            approved_indices=set(),
        )
        return WorkflowTurnResult(parsed_response=parsed_response, outcomes=outcomes)

    async def execute_parsed_response_async(
        self,
        parsed_response: ParsedModelResponse,
        context: ToolContext,
    ) -> WorkflowTurnResult:
        """Asynchronously execute parsed invocations or return a final response."""
        if parsed_response.final_response is not None:
            return WorkflowTurnResult(parsed_response=parsed_response)

        outcomes = await self._execute_sequence_async(
            parsed_response=parsed_response,
            base_context=context,
            start_index=1,
            approved_indices=set(),
        )
        return WorkflowTurnResult(parsed_response=parsed_response, outcomes=outcomes)

    def execute_model_output(
        self,
        adapter: ActionEnvelopeAdapter,
        payload: object,
        context: ToolContext,
    ) -> WorkflowTurnResult:
        """Parse one model output and execute any resulting tool invocations."""
        parsed_response = adapter.parse_model_output(payload)
        return self.execute_parsed_response(parsed_response, context)

    async def execute_model_output_async(
        self,
        adapter: ActionEnvelopeAdapter,
        payload: object,
        context: ToolContext,
    ) -> WorkflowTurnResult:
        """Asynchronously parse one model output and execute invocations."""
        parsed_response = adapter.parse_model_output(payload)
        return await self.execute_parsed_response_async(parsed_response, context)

    def list_pending_approvals(self) -> list[ApprovalRequest]:
        """Return pending approvals in insertion order."""
        return [state.approval_request for state in self._pending_approvals.values()]

    def cancel_pending_approval(self, approval_id: str) -> ApprovalRequest:
        """Discard one pending approval without resuming execution."""
        try:
            state = self._pending_approvals.pop(approval_id)
        except KeyError as exc:
            raise ValueError(f"Unknown approval id: {approval_id}") from exc
        return state.approval_request

    def resolve_pending_approval(
        self,
        approval_id: str,
        approved: bool,
        *,
        now: datetime | None = None,
    ) -> WorkflowTurnResult:
        """Resolve one pending approval and continue execution from that point."""
        del now
        try:
            state = self._pending_approvals.pop(approval_id)
        except KeyError as exc:
            raise ValueError(f"Unknown approval id: {approval_id}") from exc

        if approved:
            outcomes = self._execute_sequence(
                parsed_response=state.parsed_response,
                base_context=state.base_context,
                start_index=state.pending_index,
                approved_indices={state.pending_index},
            )
            return WorkflowTurnResult(
                parsed_response=state.parsed_response,
                outcomes=outcomes,
            )

        denied_outcome = self._build_approval_outcome(
            status=WorkflowInvocationStatus.APPROVAL_DENIED,
            approval_request=state.approval_request,
        )
        remaining = self._execute_sequence(
            parsed_response=state.parsed_response,
            base_context=state.base_context,
            start_index=state.pending_index + 1,
            approved_indices=set(),
        )
        return WorkflowTurnResult(
            parsed_response=state.parsed_response,
            outcomes=[denied_outcome, *remaining],
        )

    async def resolve_pending_approval_async(
        self,
        approval_id: str,
        approved: bool,
        *,
        now: datetime | None = None,
    ) -> WorkflowTurnResult:
        """Asynchronously resolve one pending approval and continue execution."""
        del now
        try:
            state = self._pending_approvals.pop(approval_id)
        except KeyError as exc:
            raise ValueError(f"Unknown approval id: {approval_id}") from exc

        if approved:
            outcomes = await self._execute_sequence_async(
                parsed_response=state.parsed_response,
                base_context=state.base_context,
                start_index=state.pending_index,
                approved_indices={state.pending_index},
            )
            return WorkflowTurnResult(
                parsed_response=state.parsed_response,
                outcomes=outcomes,
            )

        denied_outcome = self._build_approval_outcome(
            status=WorkflowInvocationStatus.APPROVAL_DENIED,
            approval_request=state.approval_request,
        )
        remaining = await self._execute_sequence_async(
            parsed_response=state.parsed_response,
            base_context=state.base_context,
            start_index=state.pending_index + 1,
            approved_indices=set(),
        )
        return WorkflowTurnResult(
            parsed_response=state.parsed_response,
            outcomes=[denied_outcome, *remaining],
        )

    def finalize_expired_approvals(
        self,
        *,
        now: datetime | None = None,
    ) -> list[WorkflowTurnResult]:
        """Finalize approvals whose deadline has passed as timed-out denials."""
        current_time = now or self._utc_now()
        expired_ids = [
            approval_id
            for approval_id, state in self._pending_approvals.items()
            if state.expires_at <= current_time
        ]

        results: list[WorkflowTurnResult] = []
        for approval_id in expired_ids:
            state = self._pending_approvals.pop(approval_id)
            timeout_outcome = self._build_approval_outcome(
                status=WorkflowInvocationStatus.APPROVAL_TIMED_OUT,
                approval_request=state.approval_request,
            )
            remaining = self._execute_sequence(
                parsed_response=state.parsed_response,
                base_context=state.base_context,
                start_index=state.pending_index + 1,
                approved_indices=set(),
            )
            results.append(
                WorkflowTurnResult(
                    parsed_response=state.parsed_response,
                    outcomes=[timeout_outcome, *remaining],
                )
            )

        return results

    async def finalize_expired_approvals_async(
        self,
        *,
        now: datetime | None = None,
    ) -> list[WorkflowTurnResult]:
        """Asynchronously finalize approvals past their deadline."""
        current_time = now or self._utc_now()
        expired_ids = [
            approval_id
            for approval_id, state in self._pending_approvals.items()
            if state.expires_at <= current_time
        ]

        results: list[WorkflowTurnResult] = []
        for approval_id in expired_ids:
            state = self._pending_approvals.pop(approval_id)
            timeout_outcome = self._build_approval_outcome(
                status=WorkflowInvocationStatus.APPROVAL_TIMED_OUT,
                approval_request=state.approval_request,
            )
            remaining = await self._execute_sequence_async(
                parsed_response=state.parsed_response,
                base_context=state.base_context,
                start_index=state.pending_index + 1,
                approved_indices=set(),
            )
            results.append(
                WorkflowTurnResult(
                    parsed_response=state.parsed_response,
                    outcomes=[timeout_outcome, *remaining],
                )
            )

        return results

    def _execute_sequence(
        self,
        *,
        parsed_response: ParsedModelResponse,
        base_context: ToolContext,
        start_index: int,
        approved_indices: set[int],
    ) -> list[WorkflowInvocationOutcome]:
        outcomes: list[WorkflowInvocationOutcome] = []
        invocation_count = len(parsed_response.invocations)

        for index in range(start_index, invocation_count + 1):
            request = parsed_response.invocations[index - 1]
            tool_context = self._make_tool_context(
                base_context=base_context,
                index=index,
                invocation_count=invocation_count,
            )
            outcome, pending_state = self._execute_one(
                parsed_response=parsed_response,
                base_context=base_context,
                request=request,
                index=index,
                context=tool_context,
                approval_override=index in approved_indices,
            )
            outcomes.append(outcome)
            if pending_state is not None:
                self._pending_approvals[pending_state.approval_request.approval_id] = (
                    pending_state
                )
                break

        return outcomes

    async def _execute_sequence_async(
        self,
        *,
        parsed_response: ParsedModelResponse,
        base_context: ToolContext,
        start_index: int,
        approved_indices: set[int],
    ) -> list[WorkflowInvocationOutcome]:
        outcomes: list[WorkflowInvocationOutcome] = []
        invocation_count = len(parsed_response.invocations)

        for index in range(start_index, invocation_count + 1):
            request = parsed_response.invocations[index - 1]
            tool_context = self._make_tool_context(
                base_context=base_context,
                index=index,
                invocation_count=invocation_count,
            )
            outcome, pending_state = await self._execute_one_async(
                parsed_response=parsed_response,
                base_context=base_context,
                request=request,
                index=index,
                context=tool_context,
                approval_override=index in approved_indices,
            )
            outcomes.append(outcome)
            if pending_state is not None:
                self._pending_approvals[pending_state.approval_request.approval_id] = (
                    pending_state
                )
                break

        return outcomes

    def _execute_one(
        self,
        *,
        parsed_response: ParsedModelResponse,
        base_context: ToolContext,
        request: ToolInvocationRequest,
        index: int,
        context: ToolContext,
        approval_override: bool,
    ) -> tuple[WorkflowInvocationOutcome, _PendingApprovalState | None]:
        try:
            tool = self._registry.get(request.tool_name)
            policy_decision = self._policy.evaluate(tool, context)
        except Exception:
            result = self._runtime.execute(request, context)
            return self._build_executed_outcome(
                index=index, request=request, result=result
            ), None

        if policy_decision.requires_approval and not approval_override:
            approval_request, expires_at = self._make_approval_request(
                request=request,
                index=index,
                tool=tool,
                policy_reason=policy_decision.reason,
                policy_metadata=policy_decision.metadata,
            )
            pending_state = _PendingApprovalState(
                approval_request=approval_request,
                parsed_response=parsed_response.model_copy(deep=True),
                base_context=base_context.model_copy(deep=True),
                pending_index=index,
                expires_at=expires_at,
            )
            return (
                self._build_approval_outcome(
                    status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                    approval_request=approval_request,
                ),
                pending_state,
            )

        runtime = self._runtime
        if approval_override and policy_decision.requires_approval:
            runtime = self._runtime_with_approval_override(tool)
        result = runtime.execute(request, context)
        return self._build_executed_outcome(
            index=index, request=request, result=result
        ), None

    async def _execute_one_async(
        self,
        *,
        parsed_response: ParsedModelResponse,
        base_context: ToolContext,
        request: ToolInvocationRequest,
        index: int,
        context: ToolContext,
        approval_override: bool,
    ) -> tuple[WorkflowInvocationOutcome, _PendingApprovalState | None]:
        try:
            tool = self._registry.get(request.tool_name)
            policy_decision = self._policy.evaluate(tool, context)
        except Exception:
            result = await self._runtime.execute_async(request, context)
            return self._build_executed_outcome(
                index=index, request=request, result=result
            ), None

        if policy_decision.requires_approval and not approval_override:
            approval_request, expires_at = self._make_approval_request(
                request=request,
                index=index,
                tool=tool,
                policy_reason=policy_decision.reason,
                policy_metadata=policy_decision.metadata,
            )
            pending_state = _PendingApprovalState(
                approval_request=approval_request,
                parsed_response=parsed_response.model_copy(deep=True),
                base_context=base_context.model_copy(deep=True),
                pending_index=index,
                expires_at=expires_at,
            )
            return (
                self._build_approval_outcome(
                    status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                    approval_request=approval_request,
                ),
                pending_state,
            )

        runtime = self._runtime
        if approval_override and policy_decision.requires_approval:
            runtime = self._runtime_with_approval_override(tool)
        result = await runtime.execute_async(request, context)
        return self._build_executed_outcome(
            index=index, request=request, result=result
        ), None

    def _runtime_with_approval_override(self, tool: Tool[Any, Any]) -> ToolRuntime:
        policy = self._policy.model_copy(deep=True)
        policy.require_approval_for = set(policy.require_approval_for)
        policy.require_approval_for.discard(tool.spec.side_effects)
        return ToolRuntime(self._registry, policy=policy)

    def _make_approval_request(
        self,
        *,
        request: ToolInvocationRequest,
        index: int,
        tool: Tool[Any, Any],
        policy_reason: str,
        policy_metadata: dict[str, Any],
    ) -> tuple[ApprovalRequest, datetime]:
        requested_at = self._utc_now()
        expires_at = requested_at + timedelta(
            seconds=self._policy.approval_timeout_seconds
        )
        approval_request = ApprovalRequest(
            approval_id=f"approval-{uuid4().hex}",
            invocation_index=index,
            request=request,
            tool_name=tool.spec.name,
            tool_version=tool.spec.version,
            policy_reason=policy_reason,
            policy_metadata=dict(policy_metadata),
            requested_at=self._to_timestamp(requested_at),
            expires_at=self._to_timestamp(expires_at),
        )
        return approval_request, expires_at

    @staticmethod
    def _build_executed_outcome(
        *,
        index: int,
        request: ToolInvocationRequest,
        result: Any,
    ) -> WorkflowInvocationOutcome:
        return WorkflowInvocationOutcome(
            invocation_index=index,
            request=request,
            status=WorkflowInvocationStatus.EXECUTED,
            tool_result=result,
        )

    @staticmethod
    def _build_approval_outcome(
        *,
        status: WorkflowInvocationStatus,
        approval_request: ApprovalRequest,
    ) -> WorkflowInvocationOutcome:
        return WorkflowInvocationOutcome(
            invocation_index=approval_request.invocation_index,
            request=approval_request.request,
            status=status,
            approval_request=approval_request,
        )

    def _make_tool_context(
        self,
        *,
        base_context: ToolContext,
        index: int,
        invocation_count: int,
    ) -> ToolContext:
        invocation_id = base_context.invocation_id
        if invocation_count > 1:
            invocation_id = f"{invocation_id}:{index}"

        return ToolContext(
            invocation_id=invocation_id,
            workspace=base_context.workspace,
            env=dict(base_context.env),
            logs=[],
            artifacts=[],
            metadata=dict(base_context.metadata),
        )

    def _filter_tools_for_exposure(
        self,
        tools: list[Tool[Any, Any]],
        *,
        context: ToolContext,
        include_requires_approval: bool,
    ) -> list[Tool[Any, Any]]:
        filtered: list[Tool[Any, Any]] = []
        for tool in tools:
            verdict = self._policy.verdict(tool, context)
            if verdict is PolicyVerdict.ALLOW or (
                verdict is PolicyVerdict.REQUIRE_APPROVAL and include_requires_approval
            ):
                filtered.append(tool)
        return filtered

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _to_timestamp(value: datetime) -> str:
        return value.isoformat().replace("+00:00", "Z")
