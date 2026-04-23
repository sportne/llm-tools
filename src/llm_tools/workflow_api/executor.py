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
    tool_specs: list[ToolSpec]
    input_models: dict[str, type[BaseModel]]


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
        simplify_json_schema: bool = False,
    ) -> PreparedModelInteraction:
        """Prepare a typed response model and schema for one model turn."""
        tools = self._registry.list_bindings()
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
            simplify_json_schema=simplify_json_schema,
        )
        return PreparedModelInteraction(
            response_model=response_model,
            schema=adapter.export_schema(response_model),
            tool_names=[spec.name for spec in specs],
            tool_specs=specs,
            input_models=input_models,
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

        resolution = "approve" if approved else "deny"
        return self._resume_persisted_approval(
            parsed_response=state.parsed_response,
            base_context=state.base_context,
            approval_request=state.approval_request,
            pending_index=state.pending_index,
            resolution=resolution,
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

        resolution = "approve" if approved else "deny"
        return await self._resume_persisted_approval_async(
            parsed_response=state.parsed_response,
            base_context=state.base_context,
            approval_request=state.approval_request,
            pending_index=state.pending_index,
            resolution=resolution,
        )

    def resume_persisted_approval(
        self,
        record: Any,
        resolution: str,
        *,
        now: datetime | None = None,
    ) -> WorkflowTurnResult:
        """Resume execution from a durable approval record."""
        del now
        return self._resume_persisted_approval(
            parsed_response=record.parsed_response,
            base_context=record.base_context,
            approval_request=record.approval_request,
            pending_index=record.pending_index,
            resolution=resolution,
        )

    async def resume_persisted_approval_async(
        self,
        record: Any,
        resolution: str,
        *,
        now: datetime | None = None,
    ) -> WorkflowTurnResult:
        """Asynchronously resume execution from a durable approval record."""
        del now
        return await self._resume_persisted_approval_async(
            parsed_response=record.parsed_response,
            base_context=record.base_context,
            approval_request=record.approval_request,
            pending_index=record.pending_index,
            resolution=resolution,
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
            results.append(
                self._resume_persisted_approval(
                    parsed_response=state.parsed_response,
                    base_context=state.base_context,
                    approval_request=state.approval_request,
                    pending_index=state.pending_index,
                    resolution="expire",
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
            results.append(
                await self._resume_persisted_approval_async(
                    parsed_response=state.parsed_response,
                    base_context=state.base_context,
                    approval_request=state.approval_request,
                    pending_index=state.pending_index,
                    resolution="expire",
                )
            )

        return results

    def _resume_persisted_approval(
        self,
        *,
        parsed_response: ParsedModelResponse,
        base_context: ToolContext,
        approval_request: ApprovalRequest,
        pending_index: int,
        resolution: str,
    ) -> WorkflowTurnResult:
        normalized = self._normalize_approval_resolution(resolution)
        if normalized == "approve":
            outcomes = self._execute_sequence(
                parsed_response=parsed_response,
                base_context=base_context,
                start_index=pending_index,
                approved_indices={pending_index},
            )
            return WorkflowTurnResult(
                parsed_response=parsed_response, outcomes=outcomes
            )

        approval_status = (
            WorkflowInvocationStatus.APPROVAL_TIMED_OUT
            if normalized == "expire"
            else WorkflowInvocationStatus.APPROVAL_DENIED
        )
        approval_outcome = self._build_approval_outcome(
            status=approval_status,
            approval_request=approval_request,
        )
        return WorkflowTurnResult(
            parsed_response=parsed_response,
            outcomes=[approval_outcome],
        )

    async def _resume_persisted_approval_async(
        self,
        *,
        parsed_response: ParsedModelResponse,
        base_context: ToolContext,
        approval_request: ApprovalRequest,
        pending_index: int,
        resolution: str,
    ) -> WorkflowTurnResult:
        normalized = self._normalize_approval_resolution(resolution)
        if normalized == "approve":
            outcomes = await self._execute_sequence_async(
                parsed_response=parsed_response,
                base_context=base_context,
                start_index=pending_index,
                approved_indices={pending_index},
            )
            return WorkflowTurnResult(
                parsed_response=parsed_response, outcomes=outcomes
            )

        approval_status = (
            WorkflowInvocationStatus.APPROVAL_TIMED_OUT
            if normalized == "expire"
            else WorkflowInvocationStatus.APPROVAL_DENIED
        )
        approval_outcome = self._build_approval_outcome(
            status=approval_status,
            approval_request=approval_request,
        )
        return WorkflowTurnResult(
            parsed_response=parsed_response,
            outcomes=[approval_outcome],
        )

    @staticmethod
    def _normalize_approval_resolution(resolution: str) -> str:
        if resolution not in {"approve", "deny", "cancel", "expire"}:
            raise ValueError(f"Unsupported approval resolution: {resolution}")
        return resolution

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
            inspection = self._runtime.inspect_invocation(
                request,
                context,
                approval_override=approval_override,
            )
        except Exception:
            result = self._runtime.execute(
                request,
                context,
                approval_override=approval_override,
            )
            return self._build_executed_outcome(
                index=index, request=request, result=result
            ), None

        policy_decision = inspection.policy_decision
        if policy_decision.requires_approval and not approval_override:
            approval_request, expires_at = self._make_approval_request(
                request=request,
                index=index,
                spec=self._registry.get_spec(request.tool_name),
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

        result = self._runtime.execute(
            request,
            context,
            approval_override=approval_override,
        )
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
            inspection = self._runtime.inspect_invocation(
                request,
                context,
                approval_override=approval_override,
            )
        except Exception:
            result = await self._runtime.execute_async(
                request,
                context,
                approval_override=approval_override,
            )
            return self._build_executed_outcome(
                index=index, request=request, result=result
            ), None

        policy_decision = inspection.policy_decision
        if policy_decision.requires_approval and not approval_override:
            approval_request, expires_at = self._make_approval_request(
                request=request,
                index=index,
                spec=self._registry.get_spec(request.tool_name),
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

        result = await self._runtime.execute_async(
            request,
            context,
            approval_override=approval_override,
        )
        return self._build_executed_outcome(
            index=index, request=request, result=result
        ), None

    def _make_approval_request(
        self,
        *,
        request: ToolInvocationRequest,
        index: int,
        spec: ToolSpec,
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
            tool_name=spec.name,
            tool_version=spec.version,
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
        tools: list[Any],
        *,
        context: ToolContext,
        include_requires_approval: bool,
    ) -> list[Any]:
        filtered: list[Any] = []
        for tool in tools:
            verdict = self._policy.verdict(tool.spec, context)
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
