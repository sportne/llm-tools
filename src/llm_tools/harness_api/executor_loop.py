"""Durable multi-turn executor above the one-turn workflow layer."""

from __future__ import annotations

import os
from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from llm_tools.harness_api.executor_approvals import (
    ApprovalResolution,
    HarnessRetryPolicy,
)
from llm_tools.harness_api.executor_persistence import (
    load_required_snapshot,
    save_session_with_conflict_retry,
)
from llm_tools.harness_api.models import (
    HarnessState,
    HarnessStopReason,
    HarnessTurn,
    PendingApprovalRecord,
    TurnDecision,
    TurnDecisionAction,
    rehydrate_pending_approval_context,
    sanitize_pending_approval_context,
)
from llm_tools.harness_api.protection import scrub_state_for_protection
from llm_tools.harness_api.replay import (
    StoredHarnessArtifacts,
    build_stored_artifacts,
    build_turn_trace,
)
from llm_tools.harness_api.resume import (
    ResumedHarnessSession,
    ResumeDisposition,
    resume_session,
)
from llm_tools.harness_api.store import (
    HarnessStateStore,
    StoredHarnessState,
)
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolContext
from llm_tools.workflow_api import (
    ApprovalRequest,
    WorkflowExecutor,
    WorkflowInvocationStatus,
    WorkflowTurnResult,
)


class HarnessExecutionResult(BaseModel):
    """Final stored snapshot and resume view after executor work stops."""

    snapshot: StoredHarnessState
    resumed: ResumedHarnessSession


@runtime_checkable
class HarnessTurnDriver(Protocol):
    """Driver contract for selecting work, building context, and producing a turn."""

    def select_task_ids(self, *, state: HarnessState) -> list[str]:
        """Return the next actionable task ids in deterministic order."""

    def build_context(
        self,
        *,
        state: HarnessState,
        selected_task_ids: Sequence[str],
        turn_index: int,
    ) -> ToolContext:
        """Build the workflow base context for one harness turn."""

    def run_turn(
        self,
        *,
        state: HarnessState,
        selected_task_ids: Sequence[str],
        context: ToolContext,
    ) -> ParsedModelResponse:
        """Produce the parsed model response for one turn."""

    async def run_turn_async(
        self,
        *,
        state: HarnessState,
        selected_task_ids: Sequence[str],
        context: ToolContext,
    ) -> ParsedModelResponse:
        """Asynchronously produce the parsed model response for one turn."""


@runtime_checkable
class HarnessTurnApplier(Protocol):
    """Apply a completed workflow turn back into canonical harness state."""

    def apply_turn(
        self,
        *,
        state: HarnessState,
        turn: HarnessTurn,
    ) -> tuple[HarnessState, TurnDecision]:
        """Apply one completed turn and return the next harness decision."""


class HarnessExecutor:
    """Own the durable harness loop while delegating one-turn execution."""

    def __init__(
        self,
        *,
        store: HarnessStateStore,
        workflow_executor: WorkflowExecutor,
        driver: HarnessTurnDriver,
        applier: HarnessTurnApplier,
        retry_policy: HarnessRetryPolicy | None = None,
    ) -> None:
        self._store = store
        self._workflow_executor = workflow_executor
        self._driver = driver
        self._applier = applier
        self._retry_policy = retry_policy or HarnessRetryPolicy()

    def run(
        self,
        state: HarnessState,
        *,
        expected_revision: str | None = None,
        now: datetime | None = None,
    ) -> HarnessExecutionResult:
        """Persist a starting state and run until the harness stops."""
        snapshot = self._store.save_session(
            state,
            expected_revision=expected_revision,
            artifacts=build_stored_artifacts(state=state),
        )
        return self._drive(snapshot=snapshot, approval_resolution=None, now=now)

    async def run_async(
        self,
        state: HarnessState,
        *,
        expected_revision: str | None = None,
        now: datetime | None = None,
    ) -> HarnessExecutionResult:
        """Asynchronously persist a starting state and run the harness."""
        snapshot = self._store.save_session(
            state,
            expected_revision=expected_revision,
            artifacts=build_stored_artifacts(state=state),
        )
        return await self._drive_async(
            snapshot=snapshot,
            approval_resolution=None,
            now=now,
        )

    def resume(
        self,
        session_id: str,
        *,
        approval_resolution: ApprovalResolution | None = None,
        now: datetime | None = None,
    ) -> HarnessExecutionResult:
        """Resume a stored session, optionally resolving a pending approval."""
        snapshot = self._load_required_snapshot(session_id)
        return self._drive(
            snapshot=snapshot,
            approval_resolution=approval_resolution,
            now=now,
        )

    async def resume_async(
        self,
        session_id: str,
        *,
        approval_resolution: ApprovalResolution | None = None,
        now: datetime | None = None,
    ) -> HarnessExecutionResult:
        """Asynchronously resume a stored session."""
        snapshot = self._load_required_snapshot(session_id)
        return await self._drive_async(
            snapshot=snapshot,
            approval_resolution=approval_resolution,
            now=now,
        )

    def _drive(
        self,
        *,
        snapshot: StoredHarnessState,
        approval_resolution: ApprovalResolution | None,
        now: datetime | None,
    ) -> HarnessExecutionResult:
        pending_resolution = approval_resolution
        while True:
            current_time = now or self._utc_now()
            resumed = resume_session(snapshot, now=current_time)
            if resumed.disposition in {
                ResumeDisposition.TERMINAL,
                ResumeDisposition.INCOMPATIBLE_SCHEMA,
                ResumeDisposition.CORRUPT,
            }:
                return HarnessExecutionResult(snapshot=snapshot, resumed=resumed)

            if resumed.disposition in {
                ResumeDisposition.WAITING_FOR_APPROVAL,
                ResumeDisposition.APPROVAL_EXPIRED,
            }:
                if pending_resolution is None:
                    raise ValueError(
                        "approval_resolution is required to resume a pending approval."
                    )
                snapshot, should_continue = self._resume_waiting_approval(
                    snapshot=snapshot,
                    resumed=resumed,
                    resolution=pending_resolution,
                    now=current_time,
                )
                pending_resolution = None
                if not should_continue:
                    return self._execution_result(snapshot, now=current_time)
                continue

            snapshot, should_continue = self._run_one_turn(
                snapshot=snapshot, now=current_time
            )
            if not should_continue:
                return self._execution_result(snapshot, now=current_time)

    async def _drive_async(
        self,
        *,
        snapshot: StoredHarnessState,
        approval_resolution: ApprovalResolution | None,
        now: datetime | None,
    ) -> HarnessExecutionResult:
        pending_resolution = approval_resolution
        while True:
            current_time = now or self._utc_now()
            resumed = resume_session(snapshot, now=current_time)
            if resumed.disposition in {
                ResumeDisposition.TERMINAL,
                ResumeDisposition.INCOMPATIBLE_SCHEMA,
                ResumeDisposition.CORRUPT,
            }:
                return HarnessExecutionResult(snapshot=snapshot, resumed=resumed)

            if resumed.disposition in {
                ResumeDisposition.WAITING_FOR_APPROVAL,
                ResumeDisposition.APPROVAL_EXPIRED,
            }:
                if pending_resolution is None:
                    raise ValueError(
                        "approval_resolution is required to resume a pending approval."
                    )
                snapshot, should_continue = await self._resume_waiting_approval_async(
                    snapshot=snapshot,
                    resumed=resumed,
                    resolution=pending_resolution,
                    now=current_time,
                )
                pending_resolution = None
                if not should_continue:
                    return self._execution_result(snapshot, now=current_time)
                continue

            snapshot, should_continue = await self._run_one_turn_async(
                snapshot=snapshot,
                now=current_time,
            )
            if not should_continue:
                return self._execution_result(snapshot, now=current_time)

    def _run_one_turn(
        self,
        *,
        snapshot: StoredHarnessState,
        now: datetime,
    ) -> tuple[StoredHarnessState, bool]:
        state = snapshot.state
        budget_stop = self._pre_turn_budget_stop(state, now=now)
        if budget_stop is not None:
            saved = self._save_with_conflict_retry(
                base_snapshot=snapshot,
                new_state=self._terminal_state(state, stop_reason=budget_stop, now=now),
                artifacts=build_stored_artifacts(
                    state=self._terminal_state(state, stop_reason=budget_stop, now=now),
                    prior_artifacts=snapshot.artifacts,
                ),
            )
            return saved, False

        selected_task_ids = self._driver.select_task_ids(state=state)
        if not selected_task_ids:
            saved = self._save_with_conflict_retry(
                base_snapshot=snapshot,
                new_state=self._terminal_state(
                    state,
                    stop_reason=HarnessStopReason.COMPLETED,
                    now=now,
                ),
                artifacts=build_stored_artifacts(
                    state=self._terminal_state(
                        state,
                        stop_reason=HarnessStopReason.COMPLETED,
                        now=now,
                    ),
                    prior_artifacts=snapshot.artifacts,
                ),
            )
            return saved, False

        turn_index = state.session.current_turn_index + 1
        context = self._driver.build_context(
            state=state,
            selected_task_ids=selected_task_ids,
            turn_index=turn_index,
        )
        started_at = self._timestamp(now)
        retry_state = state
        provider_retries = 0
        tool_retries = 0

        while True:
            try:
                parsed_response = self._driver.run_turn(
                    state=retry_state,
                    selected_task_ids=selected_task_ids,
                    context=context,
                )
            except Exception as exc:
                if provider_retries < self._retry_policy.max_provider_retries:
                    provider_retries += 1
                    retry_state = self._increment_retry_counts(
                        retry_state,
                        task_ids=selected_task_ids,
                    )
                    continue

                exhausted_state = self._mark_retry_exhaustion(
                    retry_state,
                    task_ids=selected_task_ids,
                    summary=(
                        f"Provider retry budget exhausted: {type(exc).__name__}: {exc}"
                    ),
                )
                terminal_state = self._terminal_state(
                    exhausted_state,
                    stop_reason=HarnessStopReason.ERROR,
                    now=now,
                )
                saved = self._save_with_conflict_retry(
                    base_snapshot=snapshot,
                    new_state=terminal_state,
                    artifacts=build_stored_artifacts(
                        state=terminal_state,
                        prior_artifacts=snapshot.artifacts,
                    ),
                )
                return saved, False

            workflow_result = self._workflow_executor.execute_parsed_response(
                parsed_response,
                context,
            )
            if self._is_waiting_for_approval(workflow_result):
                approval_request = self._approval_request_from_result(workflow_result)
                pending_record = PendingApprovalRecord(
                    approval_request=approval_request,
                    parsed_response=workflow_result.parsed_response,
                    base_context=sanitize_pending_approval_context(context),
                    pending_index=approval_request.invocation_index,
                )
                waiting_turn = HarnessTurn(
                    turn_index=turn_index,
                    started_at=started_at,
                    selected_task_ids=list(selected_task_ids),
                    workflow_result=workflow_result,
                )
                waiting_state = self._append_incomplete_turn(
                    retry_state,
                    turn=waiting_turn,
                    pending_approval=pending_record,
                )
                saved = self._save_with_conflict_retry(
                    base_snapshot=snapshot,
                    new_state=waiting_state,
                    artifacts=build_stored_artifacts(
                        state=waiting_state,
                        prior_artifacts=snapshot.artifacts,
                        turn_trace=build_turn_trace(
                            turn=waiting_turn,
                            context=context,
                            tasks_state=waiting_state,
                        ),
                    ),
                )
                return saved, False

            retryable_error = self._has_retryable_tool_error(workflow_result)
            if (
                retryable_error
                and tool_retries < self._retry_policy.max_retryable_tool_retries
            ):
                tool_retries += 1
                retry_state = self._increment_retry_counts(
                    retry_state,
                    task_ids=selected_task_ids,
                )
                continue

            completed_turn = HarnessTurn(
                turn_index=turn_index,
                started_at=started_at,
                selected_task_ids=list(selected_task_ids),
                workflow_result=workflow_result,
            )
            updated_state, decision = self._applier.apply_turn(
                state=retry_state,
                turn=completed_turn,
            )
            if retryable_error:
                updated_state = self._mark_retry_exhaustion(
                    updated_state,
                    task_ids=selected_task_ids,
                    summary="Retry budget exhausted for retryable tool errors.",
                )
                decision = self._forced_stop_decision(
                    decision=decision,
                    selected_task_ids=selected_task_ids,
                    stop_reason=HarnessStopReason.ERROR,
                    summary="Retry budget exhausted for retryable tool errors.",
                )
            elif self._exceeds_tool_invocation_budget(updated_state, workflow_result):
                decision = self._forced_stop_decision(
                    decision=decision,
                    selected_task_ids=selected_task_ids,
                    stop_reason=HarnessStopReason.BUDGET_EXHAUSTED,
                    summary="Maximum tool invocation budget exhausted.",
                )

            saved = self._save_completed_turn(
                base_snapshot=snapshot,
                state=updated_state,
                turn=completed_turn,
                decision=decision,
                now=now,
                context=context,
            )
            return saved, decision.action is not TurnDecisionAction.STOP

    async def _run_one_turn_async(
        self,
        *,
        snapshot: StoredHarnessState,
        now: datetime,
    ) -> tuple[StoredHarnessState, bool]:
        state = snapshot.state
        budget_stop = self._pre_turn_budget_stop(state, now=now)
        if budget_stop is not None:
            saved = self._save_with_conflict_retry(
                base_snapshot=snapshot,
                new_state=self._terminal_state(state, stop_reason=budget_stop, now=now),
                artifacts=build_stored_artifacts(
                    state=self._terminal_state(state, stop_reason=budget_stop, now=now),
                    prior_artifacts=snapshot.artifacts,
                ),
            )
            return saved, False

        selected_task_ids = self._driver.select_task_ids(state=state)
        if not selected_task_ids:
            saved = self._save_with_conflict_retry(
                base_snapshot=snapshot,
                new_state=self._terminal_state(
                    state,
                    stop_reason=HarnessStopReason.COMPLETED,
                    now=now,
                ),
                artifacts=build_stored_artifacts(
                    state=self._terminal_state(
                        state,
                        stop_reason=HarnessStopReason.COMPLETED,
                        now=now,
                    ),
                    prior_artifacts=snapshot.artifacts,
                ),
            )
            return saved, False

        turn_index = state.session.current_turn_index + 1
        context = self._driver.build_context(
            state=state,
            selected_task_ids=selected_task_ids,
            turn_index=turn_index,
        )
        started_at = self._timestamp(now)
        retry_state = state
        provider_retries = 0
        tool_retries = 0

        while True:
            try:
                parsed_response = await self._driver.run_turn_async(
                    state=retry_state,
                    selected_task_ids=selected_task_ids,
                    context=context,
                )
            except Exception as exc:
                if provider_retries < self._retry_policy.max_provider_retries:
                    provider_retries += 1
                    retry_state = self._increment_retry_counts(
                        retry_state,
                        task_ids=selected_task_ids,
                    )
                    continue

                exhausted_state = self._mark_retry_exhaustion(
                    retry_state,
                    task_ids=selected_task_ids,
                    summary=(
                        f"Provider retry budget exhausted: {type(exc).__name__}: {exc}"
                    ),
                )
                terminal_state = self._terminal_state(
                    exhausted_state,
                    stop_reason=HarnessStopReason.ERROR,
                    now=now,
                )
                saved = self._save_with_conflict_retry(
                    base_snapshot=snapshot,
                    new_state=terminal_state,
                    artifacts=build_stored_artifacts(
                        state=terminal_state,
                        prior_artifacts=snapshot.artifacts,
                    ),
                )
                return saved, False

            workflow_result = (
                await self._workflow_executor.execute_parsed_response_async(
                    parsed_response,
                    context,
                )
            )
            if self._is_waiting_for_approval(workflow_result):
                approval_request = self._approval_request_from_result(workflow_result)
                pending_record = PendingApprovalRecord(
                    approval_request=approval_request,
                    parsed_response=workflow_result.parsed_response,
                    base_context=sanitize_pending_approval_context(context),
                    pending_index=approval_request.invocation_index,
                )
                waiting_turn = HarnessTurn(
                    turn_index=turn_index,
                    started_at=started_at,
                    selected_task_ids=list(selected_task_ids),
                    workflow_result=workflow_result,
                )
                waiting_state = self._append_incomplete_turn(
                    retry_state,
                    turn=waiting_turn,
                    pending_approval=pending_record,
                )
                saved = self._save_with_conflict_retry(
                    base_snapshot=snapshot,
                    new_state=waiting_state,
                    artifacts=build_stored_artifacts(
                        state=waiting_state,
                        prior_artifacts=snapshot.artifacts,
                        turn_trace=build_turn_trace(
                            turn=waiting_turn,
                            context=context,
                            tasks_state=waiting_state,
                        ),
                    ),
                )
                return saved, False

            retryable_error = self._has_retryable_tool_error(workflow_result)
            if (
                retryable_error
                and tool_retries < self._retry_policy.max_retryable_tool_retries
            ):
                tool_retries += 1
                retry_state = self._increment_retry_counts(
                    retry_state,
                    task_ids=selected_task_ids,
                )
                continue

            completed_turn = HarnessTurn(
                turn_index=turn_index,
                started_at=started_at,
                selected_task_ids=list(selected_task_ids),
                workflow_result=workflow_result,
            )
            updated_state, decision = self._applier.apply_turn(
                state=retry_state,
                turn=completed_turn,
            )
            if retryable_error:
                updated_state = self._mark_retry_exhaustion(
                    updated_state,
                    task_ids=selected_task_ids,
                    summary="Retry budget exhausted for retryable tool errors.",
                )
                decision = self._forced_stop_decision(
                    decision=decision,
                    selected_task_ids=selected_task_ids,
                    stop_reason=HarnessStopReason.ERROR,
                    summary="Retry budget exhausted for retryable tool errors.",
                )
            elif self._exceeds_tool_invocation_budget(updated_state, workflow_result):
                decision = self._forced_stop_decision(
                    decision=decision,
                    selected_task_ids=selected_task_ids,
                    stop_reason=HarnessStopReason.BUDGET_EXHAUSTED,
                    summary="Maximum tool invocation budget exhausted.",
                )

            saved = self._save_completed_turn(
                base_snapshot=snapshot,
                state=updated_state,
                turn=completed_turn,
                decision=decision,
                now=now,
                context=context,
            )
            return saved, decision.action is not TurnDecisionAction.STOP

    def _resume_waiting_approval(
        self,
        *,
        snapshot: StoredHarnessState,
        resumed: ResumedHarnessSession,
        resolution: ApprovalResolution,
        now: datetime,
    ) -> tuple[StoredHarnessState, bool]:
        self._validate_approval_resolution(resumed, resolution)
        pending_approval = resumed.pending_approval
        incomplete_turn = resumed.incomplete_turn
        if pending_approval is None or incomplete_turn is None:
            raise ValueError(
                "Approval resume requires a pending approval and tail turn."
            )

        execution_context = self._rehydrate_pending_approval_context(
            pending_approval.base_context
        )
        workflow_result = self._workflow_executor.resume_persisted_approval(
            pending_approval.model_copy(
                update={"base_context": execution_context},
                deep=True,
            ),
            resolution.value,
            now=now,
        )
        completed_turn = incomplete_turn.model_copy(
            update={"workflow_result": workflow_result}
        )
        resumed_state = snapshot.state.model_copy(
            update={"pending_approvals": []},
            deep=True,
        )
        updated_state, decision = self._applier.apply_turn(
            state=resumed_state,
            turn=completed_turn,
        )

        forced_stop = self._approval_stop_reason(resolution)
        if forced_stop is not None:
            decision = self._forced_stop_decision(
                decision=decision,
                selected_task_ids=completed_turn.selected_task_ids,
                stop_reason=forced_stop,
                summary=f"Approval resolved with {resolution.value}.",
            )
        elif self._exceeds_tool_invocation_budget(updated_state, workflow_result):
            decision = self._forced_stop_decision(
                decision=decision,
                selected_task_ids=completed_turn.selected_task_ids,
                stop_reason=HarnessStopReason.BUDGET_EXHAUSTED,
                summary="Maximum tool invocation budget exhausted.",
            )

        saved = self._save_resumed_turn(
            base_snapshot=snapshot,
            state=updated_state,
            turn=completed_turn,
            decision=decision,
            now=now,
            context=sanitize_pending_approval_context(pending_approval.base_context),
        )
        return saved, decision.action is not TurnDecisionAction.STOP

    async def _resume_waiting_approval_async(
        self,
        *,
        snapshot: StoredHarnessState,
        resumed: ResumedHarnessSession,
        resolution: ApprovalResolution,
        now: datetime,
    ) -> tuple[StoredHarnessState, bool]:
        self._validate_approval_resolution(resumed, resolution)
        pending_approval = resumed.pending_approval
        incomplete_turn = resumed.incomplete_turn
        if pending_approval is None or incomplete_turn is None:
            raise ValueError(
                "Approval resume requires a pending approval and tail turn."
            )

        execution_context = self._rehydrate_pending_approval_context(
            pending_approval.base_context
        )
        workflow_result = await self._workflow_executor.resume_persisted_approval_async(
            pending_approval.model_copy(
                update={"base_context": execution_context},
                deep=True,
            ),
            resolution.value,
            now=now,
        )
        completed_turn = incomplete_turn.model_copy(
            update={"workflow_result": workflow_result}
        )
        resumed_state = snapshot.state.model_copy(
            update={"pending_approvals": []},
            deep=True,
        )
        updated_state, decision = self._applier.apply_turn(
            state=resumed_state,
            turn=completed_turn,
        )

        forced_stop = self._approval_stop_reason(resolution)
        if forced_stop is not None:
            decision = self._forced_stop_decision(
                decision=decision,
                selected_task_ids=completed_turn.selected_task_ids,
                stop_reason=forced_stop,
                summary=f"Approval resolved with {resolution.value}.",
            )
        elif self._exceeds_tool_invocation_budget(updated_state, workflow_result):
            decision = self._forced_stop_decision(
                decision=decision,
                selected_task_ids=completed_turn.selected_task_ids,
                stop_reason=HarnessStopReason.BUDGET_EXHAUSTED,
                summary="Maximum tool invocation budget exhausted.",
            )

        saved = self._save_resumed_turn(
            base_snapshot=snapshot,
            state=updated_state,
            turn=completed_turn,
            decision=decision,
            now=now,
            context=sanitize_pending_approval_context(pending_approval.base_context),
        )
        return saved, decision.action is not TurnDecisionAction.STOP

    def _save_completed_turn(
        self,
        *,
        base_snapshot: StoredHarnessState,
        state: HarnessState,
        turn: HarnessTurn,
        decision: TurnDecision,
        now: datetime,
        context: ToolContext,
    ) -> StoredHarnessState:
        finalized_turn = turn.model_copy(
            update={
                "decision": self._normalize_decision(decision, turn.selected_task_ids),
                "ended_at": self._timestamp(now),
            }
        )
        new_turns = [*state.turns, finalized_turn]
        session = state.session.model_copy(
            update={
                "current_turn_index": finalized_turn.turn_index,
                "ended_at": (
                    self._timestamp(now)
                    if finalized_turn.decision is not None
                    and finalized_turn.decision.action is TurnDecisionAction.STOP
                    else state.session.ended_at
                ),
                "stop_reason": (
                    finalized_turn.decision.stop_reason
                    if finalized_turn.decision is not None
                    and finalized_turn.decision.action is TurnDecisionAction.STOP
                    else state.session.stop_reason
                ),
            }
        )
        new_state = state.model_copy(
            update={
                "session": session,
                "turns": new_turns,
                "pending_approvals": [],
            },
            deep=True,
        )
        protection_review = context.metadata.get("protection_review", {})
        if isinstance(protection_review, dict) and protection_review.get(
            "purge_requested"
        ):
            new_state = scrub_state_for_protection(
                new_state,
                safe_message=protection_review.get("safe_message"),
            )
            finalized_turn = new_state.turns[-1]
        return self._save_with_conflict_retry(
            base_snapshot=base_snapshot,
            new_state=new_state,
            artifacts=build_stored_artifacts(
                state=new_state,
                prior_artifacts=base_snapshot.artifacts,
                turn_trace=build_turn_trace(
                    turn=finalized_turn,
                    context=context,
                    tasks_state=new_state,
                ),
            ),
        )

    def _save_resumed_turn(
        self,
        *,
        base_snapshot: StoredHarnessState,
        state: HarnessState,
        turn: HarnessTurn,
        decision: TurnDecision,
        now: datetime,
        context: ToolContext,
    ) -> StoredHarnessState:
        finalized_turn = turn.model_copy(
            update={
                "decision": self._normalize_decision(decision, turn.selected_task_ids),
                "ended_at": self._timestamp(now),
            }
        )
        turns = [*state.turns[:-1], finalized_turn]
        session = state.session.model_copy(
            update={
                "current_turn_index": len(turns),
                "ended_at": (
                    self._timestamp(now)
                    if finalized_turn.decision is not None
                    and finalized_turn.decision.action is TurnDecisionAction.STOP
                    else state.session.ended_at
                ),
                "stop_reason": (
                    finalized_turn.decision.stop_reason
                    if finalized_turn.decision is not None
                    and finalized_turn.decision.action is TurnDecisionAction.STOP
                    else state.session.stop_reason
                ),
            }
        )
        new_state = state.model_copy(
            update={
                "session": session,
                "turns": turns,
                "pending_approvals": [],
            },
            deep=True,
        )
        protection_review = context.metadata.get("protection_review", {})
        if isinstance(protection_review, dict) and protection_review.get(
            "purge_requested"
        ):
            new_state = scrub_state_for_protection(
                new_state,
                safe_message=protection_review.get("safe_message"),
            )
            finalized_turn = new_state.turns[-1]
        return self._save_with_conflict_retry(
            base_snapshot=base_snapshot,
            new_state=new_state,
            artifacts=build_stored_artifacts(
                state=new_state,
                prior_artifacts=base_snapshot.artifacts,
                turn_trace=build_turn_trace(
                    turn=finalized_turn,
                    context=context,
                    tasks_state=new_state,
                ),
            ),
        )

    def _rehydrate_pending_approval_context(self, context: ToolContext) -> ToolContext:
        return rehydrate_pending_approval_context(context, env=os.environ)

    def _append_incomplete_turn(
        self,
        state: HarnessState,
        *,
        turn: HarnessTurn,
        pending_approval: PendingApprovalRecord,
    ) -> HarnessState:
        session = state.session.model_copy(
            update={
                "current_turn_index": turn.turn_index,
                "ended_at": None,
                "stop_reason": None,
            }
        )
        return state.model_copy(
            update={
                "session": session,
                "turns": [*state.turns, turn],
                "pending_approvals": [pending_approval],
            },
            deep=True,
        )

    def _terminal_state(
        self,
        state: HarnessState,
        *,
        stop_reason: HarnessStopReason,
        now: datetime,
    ) -> HarnessState:
        session = state.session.model_copy(
            update={
                "ended_at": self._timestamp(now),
                "stop_reason": stop_reason,
            }
        )
        return state.model_copy(
            update={"session": session, "pending_approvals": []},
            deep=True,
        )

    def _save_with_conflict_retry(
        self,
        *,
        base_snapshot: StoredHarnessState,
        new_state: HarnessState,
        artifacts: StoredHarnessArtifacts | None = None,
    ) -> StoredHarnessState:
        return save_session_with_conflict_retry(
            store=self._store,
            base_snapshot=base_snapshot,
            new_state=new_state,
            artifacts=artifacts,
            max_persistence_retries=self._retry_policy.max_persistence_retries,
        )

    def _load_required_snapshot(self, session_id: str) -> StoredHarnessState:
        return load_required_snapshot(self._store, session_id)

    def _normalize_decision(
        self,
        decision: TurnDecision,
        selected_task_ids: Sequence[str],
    ) -> TurnDecision:
        if decision.selected_task_ids:
            return decision
        return decision.model_copy(
            update={"selected_task_ids": list(selected_task_ids)}
        )

    def _forced_stop_decision(
        self,
        *,
        decision: TurnDecision,
        selected_task_ids: Sequence[str],
        stop_reason: HarnessStopReason,
        summary: str,
    ) -> TurnDecision:
        return TurnDecision(
            action=TurnDecisionAction.STOP,
            selected_task_ids=(
                list(decision.selected_task_ids)
                if decision.selected_task_ids
                else list(selected_task_ids)
            ),
            stop_reason=stop_reason,
            summary=summary,
        )

    def _increment_retry_counts(
        self,
        state: HarnessState,
        *,
        task_ids: Sequence[str],
    ) -> HarnessState:
        task_id_set = set(task_ids)
        tasks = [
            task.model_copy(update={"retry_count": task.retry_count + 1})
            if task.task_id in task_id_set
            else task.model_copy(deep=True)
            for task in state.tasks
        ]
        session = state.session.model_copy(
            update={"retry_count": state.session.retry_count + 1}
        )
        return state.model_copy(update={"session": session, "tasks": tasks}, deep=True)

    def _mark_retry_exhaustion(
        self,
        state: HarnessState,
        *,
        task_ids: Sequence[str],
        summary: str,
    ) -> HarnessState:
        task_id_set = set(task_ids)
        tasks = [
            task.model_copy(update={"status_summary": summary})
            if task.task_id in task_id_set
            else task.model_copy(deep=True)
            for task in state.tasks
        ]
        return state.model_copy(update={"tasks": tasks}, deep=True)

    def _pre_turn_budget_stop(
        self,
        state: HarnessState,
        *,
        now: datetime,
    ) -> HarnessStopReason | None:
        policy = state.session.budget_policy
        if (
            policy.max_turns is not None
            and state.session.current_turn_index >= policy.max_turns
        ):
            return HarnessStopReason.BUDGET_EXHAUSTED
        if policy.max_elapsed_seconds is not None:
            started_at = self._parse_timestamp(state.session.started_at)
            elapsed_seconds = int((now - started_at).total_seconds())
            if elapsed_seconds >= policy.max_elapsed_seconds:
                return HarnessStopReason.BUDGET_EXHAUSTED
        return None

    def _exceeds_tool_invocation_budget(
        self,
        state: HarnessState,
        workflow_result: WorkflowTurnResult,
    ) -> bool:
        max_tool_invocations = state.session.budget_policy.max_tool_invocations
        if max_tool_invocations is None:
            return False
        executed = self._count_executed_outcomes(workflow_result)
        return (
            self._count_persisted_tool_invocations(state) + executed
            > max_tool_invocations
        )

    @staticmethod
    def _count_persisted_tool_invocations(state: HarnessState) -> int:
        count = 0
        for turn in state.turns:
            if turn.workflow_result is None:
                continue
            count += HarnessExecutor._count_executed_outcomes(turn.workflow_result)
        return count

    @staticmethod
    def _count_executed_outcomes(workflow_result: WorkflowTurnResult) -> int:
        return sum(
            1
            for outcome in workflow_result.outcomes
            if outcome.status is WorkflowInvocationStatus.EXECUTED
        )

    @staticmethod
    def _has_retryable_tool_error(workflow_result: WorkflowTurnResult) -> bool:
        for outcome in workflow_result.outcomes:
            if outcome.status is not WorkflowInvocationStatus.EXECUTED:
                continue
            if outcome.tool_result is None or outcome.tool_result.error is None:
                continue
            if outcome.tool_result.error.retryable:
                return True
        return False

    @staticmethod
    def _is_waiting_for_approval(workflow_result: WorkflowTurnResult) -> bool:
        return bool(workflow_result.outcomes) and (
            workflow_result.outcomes[-1].status
            is WorkflowInvocationStatus.APPROVAL_REQUESTED
        )

    @staticmethod
    def _approval_request_from_result(
        workflow_result: WorkflowTurnResult,
    ) -> ApprovalRequest:
        approval_request = workflow_result.outcomes[-1].approval_request
        if approval_request is None:
            raise ValueError(
                "approval_requested workflow outcomes must include approval_request."
            )
        return approval_request

    @staticmethod
    def _approval_stop_reason(
        resolution: ApprovalResolution,
    ) -> HarnessStopReason | None:
        if resolution is ApprovalResolution.DENY:
            return HarnessStopReason.APPROVAL_DENIED
        if resolution is ApprovalResolution.EXPIRE:
            return HarnessStopReason.APPROVAL_EXPIRED
        if resolution is ApprovalResolution.CANCEL:
            return HarnessStopReason.APPROVAL_CANCELED
        return None

    @staticmethod
    def _validate_approval_resolution(
        resumed: ResumedHarnessSession,
        resolution: ApprovalResolution,
    ) -> None:
        if (
            resumed.disposition is ResumeDisposition.APPROVAL_EXPIRED
            and resolution is not ApprovalResolution.EXPIRE
        ):
            raise ValueError(
                "Expired approvals may only resume with EXPIRE resolution."
            )

    @staticmethod
    def _execution_result(
        snapshot: StoredHarnessState,
        *,
        now: datetime,
    ) -> HarnessExecutionResult:
        return HarnessExecutionResult(
            snapshot=snapshot,
            resumed=resume_session(snapshot, now=now),
        )

    @staticmethod
    def _utc_now() -> datetime:
        return datetime.now(UTC)

    @staticmethod
    def _timestamp(value: datetime) -> str:
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _parse_timestamp(value: str) -> datetime:
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
