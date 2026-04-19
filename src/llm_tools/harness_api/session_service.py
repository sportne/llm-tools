"""Public session service models and orchestration helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, Field

from llm_tools.harness_api.context import HarnessContextBuilder
from llm_tools.harness_api.defaults import (
    DefaultHarnessTurnDriver,
    HarnessModelTurnProvider,
    MinimalHarnessTurnApplier,
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
    HarnessStopReason,
    TurnDecision,
    TurnDecisionAction,
    sanitize_pending_approval_context,
)
from llm_tools.harness_api.planning import HarnessPlanner
from llm_tools.harness_api.replay import (
    HarnessReplayResult,
    HarnessSessionSummary,
    build_session_summary,
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
from llm_tools.harness_api.tasks import create_root_task
from llm_tools.workflow_api import WorkflowExecutor


class HarnessSessionCreateRequest(BaseModel):
    title: str = Field(min_length=1)
    intent: str = Field(min_length=1)
    budget_policy: BudgetPolicy
    session_id: str | None = None
    root_task_id: str = "task-1"
    started_at: str | None = None


class HarnessSessionRunRequest(BaseModel):
    session_id: str = Field(min_length=1)
    expected_revision: str | None = None


class HarnessSessionResumeRequest(BaseModel):
    session_id: str = Field(min_length=1)
    approval_resolution: ApprovalResolution | None = None


class HarnessSessionStopRequest(BaseModel):
    session_id: str = Field(min_length=1)
    stop_reason: HarnessStopReason = HarnessStopReason.CANCELED


class HarnessSessionInspectRequest(BaseModel):
    session_id: str = Field(min_length=1)
    include_replay: bool = False


class HarnessSessionListRequest(BaseModel):
    limit: int | None = Field(default=None, ge=1)
    include_replay: bool = False


class HarnessSessionInspection(BaseModel):
    snapshot: StoredHarnessState
    resumed: ResumedHarnessSession
    summary: HarnessSessionSummary
    replay: HarnessReplayResult | None = None


class HarnessSessionListItem(BaseModel):
    snapshot: StoredHarnessState
    summary: HarnessSessionSummary
    replay: HarnessReplayResult | None = None


class HarnessSessionListResult(BaseModel):
    sessions: list[HarnessSessionListItem] = Field(default_factory=list)


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
        self, request: HarnessSessionCreateRequest
    ) -> StoredHarnessState:
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

    def run_session(self, request: HarnessSessionRunRequest) -> HarnessExecutionResult:
        snapshot = self._store.load_session(request.session_id)
        if snapshot is None:
            raise ValueError(f"Unknown session id: {request.session_id}")
        return self._executor.run(
            snapshot.state,
            expected_revision=request.expected_revision or snapshot.revision,
        )

    async def run_session_async(
        self, request: HarnessSessionRunRequest
    ) -> HarnessExecutionResult:
        snapshot = self._store.load_session(request.session_id)
        if snapshot is None:
            raise ValueError(f"Unknown session id: {request.session_id}")
        return await self._executor.run_async(
            snapshot.state,
            expected_revision=request.expected_revision or snapshot.revision,
        )

    def resume_session(
        self, request: HarnessSessionResumeRequest
    ) -> HarnessExecutionResult:
        return self._executor.resume(
            request.session_id,
            approval_resolution=request.approval_resolution,
        )

    async def resume_session_async(
        self, request: HarnessSessionResumeRequest
    ) -> HarnessExecutionResult:
        return await self._executor.resume_async(
            request.session_id,
            approval_resolution=request.approval_resolution,
        )

    def stop_session(
        self, request: HarnessSessionStopRequest
    ) -> HarnessSessionInspection:
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
        self, request: HarnessSessionInspectRequest
    ) -> HarnessSessionInspection:
        return self._inspect_snapshot(
            self._required_snapshot(request.session_id),
            include_replay=request.include_replay,
        )

    def list_sessions(
        self, request: HarnessSessionListRequest
    ) -> HarnessSessionListResult:
        items = []
        for snapshot in self._store.list_sessions(limit=request.limit):
            summary = snapshot.artifacts.summary or build_session_summary(
                snapshot.state
            )
            replay = replay_session(snapshot) if request.include_replay else None
            items.append(
                HarnessSessionListItem(
                    snapshot=snapshot,
                    summary=summary,
                    replay=replay,
                )
            )
        return HarnessSessionListResult(sessions=items)

    def _inspect_snapshot(
        self,
        snapshot: StoredHarnessState,
        *,
        include_replay: bool,
    ) -> HarnessSessionInspection:
        summary = snapshot.artifacts.summary or build_session_summary(snapshot.state)
        return HarnessSessionInspection(
            snapshot=snapshot,
            resumed=resume_session(snapshot),
            summary=summary,
            replay=replay_session(snapshot) if include_replay else None,
        )

    def _required_snapshot(self, session_id: str) -> StoredHarnessState:
        snapshot = self._store.load_session(session_id)
        if snapshot is None:
            raise ValueError(f"Unknown session id: {session_id}")
        return snapshot


def _timestamp(value: datetime) -> str:
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")


__all__ = [
    "HarnessSessionCreateRequest",
    "HarnessSessionInspectRequest",
    "HarnessSessionInspection",
    "HarnessSessionListItem",
    "HarnessSessionListRequest",
    "HarnessSessionListResult",
    "HarnessSessionResumeRequest",
    "HarnessSessionRunRequest",
    "HarnessSessionService",
    "HarnessSessionStopRequest",
]
