"""Harness session service models and orchestration."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import UTC, datetime
from uuid import uuid4

from llm_tools.harness_api.approval_context import sanitize_pending_approval_context
from llm_tools.harness_api.context import HarnessContextBuilder
from llm_tools.harness_api.defaults import (
    DefaultHarnessTurnDriver,
    HarnessModelTurnProvider,
    MinimalHarnessTurnApplier,
)
from llm_tools.harness_api.executor import (
    HarnessExecutionResult,
    HarnessExecutor,
    HarnessRetryPolicy,
    HarnessTurnApplier,
    HarnessTurnDriver,
)
from llm_tools.harness_api.models import (
    HarnessState,
    TurnDecision,
    TurnDecisionAction,
)
from llm_tools.harness_api.planning import HarnessPlanner
from llm_tools.harness_api.replay import (
    HarnessSessionSummary,
    build_canonical_artifacts,
    build_stored_artifacts,
    build_turn_trace,
    replay_session,
)
from llm_tools.harness_api.resume import resume_session
from llm_tools.harness_api.session_service_models import (
    HarnessSessionCreateRequest as HarnessSessionCreateRequest,
)
from llm_tools.harness_api.session_service_models import (
    HarnessSessionInspection as HarnessSessionInspection,
)
from llm_tools.harness_api.session_service_models import (
    HarnessSessionInspectRequest as HarnessSessionInspectRequest,
)
from llm_tools.harness_api.session_service_models import (
    HarnessSessionListItem as HarnessSessionListItem,
)
from llm_tools.harness_api.session_service_models import (
    HarnessSessionListRequest as HarnessSessionListRequest,
)
from llm_tools.harness_api.session_service_models import (
    HarnessSessionListResult as HarnessSessionListResult,
)
from llm_tools.harness_api.session_service_models import (
    HarnessSessionResumeRequest as HarnessSessionResumeRequest,
)
from llm_tools.harness_api.session_service_models import (
    HarnessSessionRunRequest as HarnessSessionRunRequest,
)
from llm_tools.harness_api.session_service_models import (
    HarnessSessionStopRequest as HarnessSessionStopRequest,
)
from llm_tools.harness_api.store import (
    CURRENT_HARNESS_STATE_SCHEMA_VERSION,
    HarnessStateStore,
    StoredHarnessState,
)
from llm_tools.harness_api.tasks import create_root_task
from llm_tools.workflow_api import WorkflowExecutor


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
        approval_context_env: Mapping[str, str] | None = None,
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
            approval_context_env=approval_context_env,
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
            turn_trace = build_turn_trace(turn=tail_turn, context=context)
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
