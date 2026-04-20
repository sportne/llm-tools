"""Public Python session interfaces for harness-backed execution."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Protocol, runtime_checkable
from uuid import uuid4

from pydantic import BaseModel, Field

from llm_tools.harness_api.context import (
    DefaultHarnessContextBuilder,
    HarnessContextBuilder,
)
from llm_tools.harness_api.executor import (
    ApprovalResolution,
    HarnessExecutionResult,
    HarnessExecutor,
    HarnessRetryPolicy,
    HarnessTurnApplier,
    HarnessTurnDriver,
)
from llm_tools.harness_api.models import (
    BudgetPolicy,
    HarnessState,
    HarnessStopReason,
    HarnessTurn,
    TaskLifecycleStatus,
    TurnDecision,
    TurnDecisionAction,
    sanitize_pending_approval_context,
)
from llm_tools.harness_api.planning import DeterministicHarnessPlanner, HarnessPlanner
from llm_tools.harness_api.replay import (
    HarnessReplayResult,
    HarnessSessionSummary,
    build_canonical_artifacts,
    build_stored_artifacts,
    build_turn_trace,
    replay_session,
)
from llm_tools.harness_api.resume import ResumedHarnessSession, resume_session
from llm_tools.harness_api.store import (
    CURRENT_HARNESS_STATE_SCHEMA_VERSION,
    HarnessStateStore,
    StoredHarnessState,
)
from llm_tools.harness_api.tasks import (
    block_task,
    complete_task,
    create_root_task,
    fail_task,
    start_task,
)
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import ToolContext
from llm_tools.workflow_api import PreparedModelInteraction, WorkflowExecutor


class HarnessSessionCreateRequest(BaseModel):
    """Create a new persisted root-task harness session."""

    title: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    budget_policy: BudgetPolicy
    session_id: str | None = None
    root_task_id: str = "task-1"
    started_at: str | None = None


class HarnessSessionRunRequest(BaseModel):
    """Run a persisted harness session to its next durable stop."""

    session_id: str = Field(min_length=1)
    expected_revision: str | None = None
    allow_interrupted_turn_replay: bool = False


class HarnessSessionResumeRequest(BaseModel):
    """Resume a persisted harness session, optionally resolving approval."""

    session_id: str = Field(min_length=1)
    approval_resolution: ApprovalResolution | None = None
    allow_interrupted_turn_replay: bool = False


class HarnessSessionStopRequest(BaseModel):
    """Stop a persisted harness session without further execution."""

    session_id: str = Field(min_length=1)
    stop_reason: HarnessStopReason = HarnessStopReason.CANCELED


class HarnessSessionInspectRequest(BaseModel):
    """Inspect one persisted harness session."""

    session_id: str = Field(min_length=1)
    include_replay: bool = False


class HarnessSessionListRequest(BaseModel):
    """List recent persisted harness sessions."""

    limit: int | None = Field(default=None, ge=1)
    include_replay: bool = False


class HarnessSessionInspection(BaseModel):
    """Typed inspection payload for one stored harness session."""

    snapshot: StoredHarnessState
    resumed: ResumedHarnessSession
    summary: HarnessSessionSummary
    replay: HarnessReplayResult | None = None


class HarnessSessionListItem(BaseModel):
    """One recent persisted harness session."""

    snapshot: StoredHarnessState
    summary: HarnessSessionSummary
    replay: HarnessReplayResult | None = None


class HarnessSessionListResult(BaseModel):
    """Recent stored harness sessions in newest-first order."""

    sessions: list[HarnessSessionListItem] = Field(default_factory=list)


@runtime_checkable
class HarnessModelTurnProvider(Protocol):
    """Model-turn callback used by the default harness driver."""

    def run(
        self,
        *,
        state: HarnessState,
        selected_task_ids: Sequence[str],
        context: ToolContext,
        adapter: ActionEnvelopeAdapter,
        prepared_interaction: PreparedModelInteraction,
    ) -> ParsedModelResponse:
        """Return the parsed model response for one harness turn."""

    async def run_async(
        self,
        *,
        state: HarnessState,
        selected_task_ids: Sequence[str],
        context: ToolContext,
        adapter: ActionEnvelopeAdapter,
        prepared_interaction: PreparedModelInteraction,
    ) -> ParsedModelResponse:
        """Asynchronously return the parsed model response for one harness turn."""


class ScriptedParsedResponseProvider:
    """Deterministic provider that replays parsed responses by turn index."""

    def __init__(self, responses: Sequence[ParsedModelResponse]) -> None:
        self._responses = [response.model_copy(deep=True) for response in responses]

    def run(
        self,
        *,
        state: HarnessState,
        selected_task_ids: Sequence[str],
        context: ToolContext,
        adapter: ActionEnvelopeAdapter,
        prepared_interaction: PreparedModelInteraction,
    ) -> ParsedModelResponse:
        del selected_task_ids, context, adapter, prepared_interaction
        return self._response_for_turn(state.session.current_turn_index)

    async def run_async(
        self,
        *,
        state: HarnessState,
        selected_task_ids: Sequence[str],
        context: ToolContext,
        adapter: ActionEnvelopeAdapter,
        prepared_interaction: PreparedModelInteraction,
    ) -> ParsedModelResponse:
        del selected_task_ids, context, adapter, prepared_interaction
        return self._response_for_turn(state.session.current_turn_index)

    def _response_for_turn(self, current_turn_index: int) -> ParsedModelResponse:
        if current_turn_index >= len(self._responses):
            raise RuntimeError(
                "No scripted parsed response remains for the requested turn index."
            )
        return self._responses[current_turn_index].model_copy(deep=True)


class DefaultHarnessTurnDriver:
    """Minimal built-in driver based on planner, context builder, and provider."""

    def __init__(
        self,
        *,
        workflow_executor: WorkflowExecutor,
        provider: HarnessModelTurnProvider,
        planner: HarnessPlanner | None = None,
        context_builder: HarnessContextBuilder | None = None,
        workspace: str | None = None,
        retry_policy: HarnessRetryPolicy | None = None,
    ) -> None:
        self._workflow_executor = workflow_executor
        self._provider = provider
        self._planner = planner or DeterministicHarnessPlanner()
        self._context_builder = context_builder or DefaultHarnessContextBuilder()
        self._workspace = workspace
        self._adapter = ActionEnvelopeAdapter()
        self._last_selection: list[str] = []
        self._last_state: HarnessState | None = None

    def select_task_ids(self, *, state: HarnessState) -> list[str]:
        selection = self._planner.select_tasks(state=state)
        self._last_selection = list(selection.selected_task_ids)
        return list(selection.selected_task_ids)

    def build_context(
        self,
        *,
        state: HarnessState,
        selected_task_ids: Sequence[str],
        turn_index: int,
    ) -> ToolContext:
        bundle = self._context_builder.build(
            state=state,
            selected_task_ids=selected_task_ids,
            turn_index=turn_index,
            workspace=self._workspace,
        )
        previous_selected = [] if not state.turns else state.turns[-1].selected_task_ids
        decision = None if not state.turns else state.turns[-1].decision
        replanning_triggers = []
        if self._last_state is not None:
            replanning_triggers = [
                trigger.value
                for trigger in self._planner.detect_replanning_triggers(
                    previous_state=self._last_state,
                    current_state=state,
                    previous_selected_task_ids=previous_selected,
                    decision=decision,
                )
            ]
        self._last_state = state.model_copy(deep=True)
        metadata = dict(bundle.tool_context.metadata)
        metadata["harness_planning"] = {
            "selected_task_ids": list(selected_task_ids),
            "replanning_triggers": replanning_triggers,
        }
        return bundle.tool_context.model_copy(update={"metadata": metadata})

    def run_turn(
        self,
        *,
        state: HarnessState,
        selected_task_ids: Sequence[str],
        context: ToolContext,
    ) -> ParsedModelResponse:
        prepared = self._workflow_executor.prepare_model_interaction(
            self._adapter,
            context=context,
            include_requires_approval=True,
            final_response_model=str,
        )
        return self._provider.run(
            state=state,
            selected_task_ids=selected_task_ids,
            context=context,
            adapter=self._adapter,
            prepared_interaction=prepared,
        )

    async def run_turn_async(
        self,
        *,
        state: HarnessState,
        selected_task_ids: Sequence[str],
        context: ToolContext,
    ) -> ParsedModelResponse:
        prepared = self._workflow_executor.prepare_model_interaction(
            self._adapter,
            context=context,
            include_requires_approval=True,
            final_response_model=str,
        )
        return await self._provider.run_async(
            state=state,
            selected_task_ids=selected_task_ids,
            context=context,
            adapter=self._adapter,
            prepared_interaction=prepared,
        )


class MinimalHarnessTurnApplier:
    """Narrow built-in turn applier for simple root-session execution."""

    def apply_turn(
        self,
        *,
        state: HarnessState,
        turn: HarnessTurn,
    ) -> tuple[HarnessState, TurnDecision]:
        workflow_result = turn.workflow_result
        if workflow_result is None:
            return state, TurnDecision(
                action=TurnDecisionAction.STOP,
                selected_task_ids=list(turn.selected_task_ids),
                stop_reason=HarnessStopReason.ERROR,
                summary="Harness turn completed without a workflow_result.",
            )

        updated_state = state
        selected_task_ids = list(turn.selected_task_ids)
        selected_tasks = {
            task.task_id: task
            for task in state.tasks
            if task.task_id in set(selected_task_ids)
        }
        for task_id, task in selected_tasks.items():
            if task.status is TaskLifecycleStatus.PENDING:
                updated_state = start_task(
                    updated_state,
                    task_id=task_id,
                    started_at=turn.started_at,
                )

        approval_denied = any(
            outcome.status is not None
            and outcome.status.value in {"approval_denied", "approval_timed_out"}
            for outcome in workflow_result.outcomes
        )
        if approval_denied:
            for task_id in selected_task_ids:
                updated_state = block_task(
                    updated_state,
                    task_id=task_id,
                    status_summary="Approval prevented task completion.",
                )
            return updated_state, TurnDecision(
                action=TurnDecisionAction.SELECT_TASKS,
                selected_task_ids=selected_task_ids,
                summary="Approval prevented task completion.",
            )

        failure_messages = [
            outcome.tool_result.error.message
            for outcome in workflow_result.outcomes
            if outcome.tool_result is not None
            and outcome.tool_result.error is not None
            and not outcome.tool_result.ok
        ]
        if failure_messages:
            for task_id in selected_task_ids:
                updated_state = fail_task(
                    updated_state,
                    task_id=task_id,
                    finished_at=turn.started_at,
                    status_summary=failure_messages[0],
                )
            return updated_state, TurnDecision(
                action=TurnDecisionAction.STOP,
                selected_task_ids=selected_task_ids,
                stop_reason=HarnessStopReason.ERROR,
                summary=failure_messages[0],
            )

        for task_id in selected_task_ids:
            task = next(task for task in updated_state.tasks if task.task_id == task_id)
            if task.status is TaskLifecycleStatus.IN_PROGRESS:
                updated_state = complete_task(
                    updated_state,
                    task_id=task_id,
                    finished_at=turn.started_at,
                )

        has_active_tasks = any(
            task.status
            in {
                TaskLifecycleStatus.PENDING,
                TaskLifecycleStatus.IN_PROGRESS,
                TaskLifecycleStatus.BLOCKED,
            }
            for task in updated_state.tasks
        )
        summary = "Completed selected tasks."
        if workflow_result.parsed_response.final_response is not None:
            summary = "Model returned a final response."
        action = (
            TurnDecisionAction.CONTINUE if has_active_tasks else TurnDecisionAction.STOP
        )
        stop_reason = None if has_active_tasks else HarnessStopReason.COMPLETED
        return updated_state, TurnDecision(
            action=action,
            selected_task_ids=selected_task_ids,
            stop_reason=stop_reason,
            summary=summary,
        )


class HarnessSessionService:
    """Public session-level API for creating, running, and inspecting harness work."""

    def __init__(
        self,
        *,
        store: HarnessStateStore,
        workflow_executor: WorkflowExecutor,
        driver: HarnessTurnDriver | None = None,
        applier: HarnessTurnApplier | None = None,
        provider: HarnessModelTurnProvider | None = None,
        planner: HarnessPlanner | None = None,
        context_builder: HarnessContextBuilder | None = None,
        workspace: str | None = None,
        retry_policy: HarnessRetryPolicy | None = None,
    ) -> None:
        if driver is None:
            if provider is None:
                raise ValueError("provider is required when driver is not supplied.")
            driver = DefaultHarnessTurnDriver(
                workflow_executor=workflow_executor,
                provider=provider,
                planner=planner,
                context_builder=context_builder,
                workspace=workspace,
            )
        self._store = store
        self._driver = driver
        self._applier = applier or MinimalHarnessTurnApplier()
        self._executor = HarnessExecutor(
            store=store,
            workflow_executor=workflow_executor,
            driver=self._driver,
            applier=self._applier,
            retry_policy=retry_policy,
        )

    def create_session(
        self,
        request: HarnessSessionCreateRequest,
    ) -> StoredHarnessState:
        """Create and persist a new root-task harness session."""
        state = create_root_task(
            schema_version=CURRENT_HARNESS_STATE_SCHEMA_VERSION,
            session_id=request.session_id or f"session-{uuid4().hex}",
            root_task_id=request.root_task_id,
            title=request.title,
            intent=request.intent,
            budget_policy=request.budget_policy,
            started_at=request.started_at or _timestamp(datetime.now(UTC)),
        )
        return self._store.save_session(
            state,
            artifacts=build_stored_artifacts(state=state),
        )

    def run_session(
        self,
        request: HarnessSessionRunRequest,
    ) -> HarnessExecutionResult:
        """Run a stored harness session until its next durable stop."""
        snapshot = self._store.load_session(request.session_id)
        if snapshot is None:
            raise ValueError(f"Unknown session id: {request.session_id}")
        return self._executor.run(
            snapshot.state,
            expected_revision=request.expected_revision or snapshot.revision,
            allow_interrupted_turn_replay=request.allow_interrupted_turn_replay,
        )

    async def run_session_async(
        self,
        request: HarnessSessionRunRequest,
    ) -> HarnessExecutionResult:
        """Asynchronously run a stored harness session."""
        snapshot = self._store.load_session(request.session_id)
        if snapshot is None:
            raise ValueError(f"Unknown session id: {request.session_id}")
        return await self._executor.run_async(
            snapshot.state,
            expected_revision=request.expected_revision or snapshot.revision,
            allow_interrupted_turn_replay=request.allow_interrupted_turn_replay,
        )

    def resume_session(
        self,
        request: HarnessSessionResumeRequest,
    ) -> HarnessExecutionResult:
        """Resume a stored harness session."""
        return self._executor.resume(
            request.session_id,
            approval_resolution=request.approval_resolution,
            allow_interrupted_turn_replay=request.allow_interrupted_turn_replay,
        )

    async def resume_session_async(
        self,
        request: HarnessSessionResumeRequest,
    ) -> HarnessExecutionResult:
        """Asynchronously resume a stored harness session."""
        return await self._executor.resume_async(
            request.session_id,
            approval_resolution=request.approval_resolution,
            allow_interrupted_turn_replay=request.allow_interrupted_turn_replay,
        )

    def stop_session(
        self,
        request: HarnessSessionStopRequest,
    ) -> HarnessSessionInspection:
        """Stop a stored harness session without further execution."""
        snapshot = self._required_snapshot(request.session_id)
        state = snapshot.state.model_copy(deep=True)
        now = _timestamp(datetime.now(UTC))
        turn_trace = None
        if state.turns and state.turns[-1].decision is None:
            tail_turn = state.turns[-1].model_copy(
                update={
                    "decision": TurnDecision(
                        action=TurnDecisionAction.STOP,
                        selected_task_ids=list(state.turns[-1].selected_task_ids),
                        stop_reason=request.stop_reason,
                        summary="Session stopped by operator.",
                    ),
                    "ended_at": now,
                }
            )
            turns = [*state.turns[:-1], tail_turn]
            state = state.model_copy(
                update={"turns": turns, "pending_approvals": []}, deep=True
            )
            context = None
            if snapshot.state.pending_approvals:
                context = sanitize_pending_approval_context(
                    snapshot.state.pending_approvals[0].base_context
                )
            turn_trace = build_turn_trace(
                turn=tail_turn, context=context, tasks_state=state
            )
        session = state.session.model_copy(
            update={"ended_at": now, "stop_reason": request.stop_reason}
        )
        state = state.model_copy(
            update={"session": session, "pending_approvals": []}, deep=True
        )
        saved = self._store.save_session(
            state,
            expected_revision=snapshot.revision,
            artifacts=build_stored_artifacts(
                state=state,
                prior_artifacts=snapshot.artifacts,
                turn_trace=turn_trace,
            ),
        )
        return self._inspect_snapshot(saved, include_replay=False)

    def inspect_session(
        self,
        request: HarnessSessionInspectRequest,
    ) -> HarnessSessionInspection:
        """Inspect a stored harness session."""
        return self._inspect_snapshot(
            self._required_snapshot(request.session_id),
            include_replay=request.include_replay,
        )

    def list_sessions(
        self,
        request: HarnessSessionListRequest,
    ) -> HarnessSessionListResult:
        """List recent stored harness sessions."""
        items = []
        for snapshot in self._store.list_sessions(limit=request.limit):
            normalized = _normalized_snapshot(snapshot)
            summary = _required_artifact_summary(normalized)
            items.append(
                HarnessSessionListItem(
                    snapshot=normalized,
                    summary=summary,
                    replay=(
                        replay_session(normalized) if request.include_replay else None
                    ),
                )
            )
        return HarnessSessionListResult(sessions=items)

    def _inspect_snapshot(
        self,
        snapshot: StoredHarnessState,
        *,
        include_replay: bool,
    ) -> HarnessSessionInspection:
        normalized = _normalized_snapshot(snapshot)
        return HarnessSessionInspection(
            snapshot=normalized,
            resumed=resume_session(normalized),
            summary=_required_artifact_summary(normalized),
            replay=replay_session(normalized) if include_replay else None,
        )

    def _required_snapshot(self, session_id: str) -> StoredHarnessState:
        snapshot = self._store.load_session(session_id)
        if snapshot is None:
            raise ValueError(f"Unknown session id: {session_id}")
        return snapshot


def _required_artifact_summary(snapshot: StoredHarnessState) -> HarnessSessionSummary:
    summary = snapshot.artifacts.summary
    if summary is None:
        raise ValueError("Normalized harness snapshots must include a summary.")
    return summary


def _normalized_snapshot(snapshot: StoredHarnessState) -> StoredHarnessState:
    normalized_state = HarnessState.model_validate(
        snapshot.state.model_dump(mode="python")
    )
    trusted_artifacts = build_canonical_artifacts(normalized_state)
    return snapshot.model_copy(
        update={"state": normalized_state, "artifacts": trusted_artifacts},
        deep=True,
    )


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
