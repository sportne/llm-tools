"""Harness executor integration tests."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel

from llm_tools.harness_api import (
    CURRENT_HARNESS_STATE_SCHEMA_VERSION,
    ApprovalResolution,
    BudgetPolicy,
    HarnessExecutionResult,
    HarnessExecutor,
    HarnessRetryPolicy,
    HarnessState,
    HarnessStateConflictError,
    HarnessStopReason,
    HarnessTurn,
    InMemoryHarnessStateStore,
    ResumeDisposition,
    StoredHarnessArtifacts,
    TaskLifecycleStatus,
    TurnDecision,
    TurnDecisionAction,
    create_root_task,
)
from llm_tools.harness_api.models import PendingApprovalRecord
from llm_tools.harness_api.resume import ResumedHarnessSession
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import (
    ErrorCode,
    SideEffectClass,
    SourceProvenanceRef,
    Tool,
    ToolContext,
    ToolError,
    ToolExecutionContext,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from llm_tools.tools.filesystem import register_filesystem_tools
from llm_tools.workflow_api import (
    ApprovalRequest,
    WorkflowExecutor,
    WorkflowInvocationOutcome,
    WorkflowInvocationStatus,
    WorkflowTurnResult,
)


class _WorkInput(BaseModel):
    value: str


class _WorkOutput(BaseModel):
    value: str


class _WorkTool(Tool[_WorkInput, _WorkOutput]):
    spec = ToolSpec(
        name="harness_work",
        description="Harness test tool.",
        side_effects=SideEffectClass.NONE,
    )
    input_model = _WorkInput
    output_model = _WorkOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: _WorkInput
    ) -> _WorkOutput:
        context.log(f"work:{args.value}")
        return _WorkOutput(value=args.value)


class _EnvEchoInput(BaseModel):
    key: str


class _EnvEchoOutput(BaseModel):
    value: str


class _EnvEchoTool(Tool[_EnvEchoInput, _EnvEchoOutput]):
    spec = ToolSpec(
        name="env_echo",
        description="Return one value from the scoped tool secrets.",
        side_effects=SideEffectClass.NONE,
        required_secrets=["HARNESS_RESUME_SECRET"],
    )
    input_model = _EnvEchoInput
    output_model = _EnvEchoOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: _EnvEchoInput
    ) -> _EnvEchoOutput:
        return _EnvEchoOutput(value=context.secrets.get(args.key, ""))


class _StaticDriver:
    def __init__(
        self,
        *,
        workspace: Path,
        selected_task_ids: list[str] | None = None,
        payloads: list[ParsedModelResponse | Exception] | None = None,
        context_env: dict[str, str] | None = None,
        context_logs: list[str] | None = None,
        context_artifacts: list[str] | None = None,
        context_source_provenance: list[SourceProvenanceRef] | None = None,
        context_metadata: dict[str, object] | None = None,
    ) -> None:
        self.workspace = workspace
        self._selected_task_ids = selected_task_ids or ["task-1"]
        self._payloads = list(
            payloads
            or [
                ParsedModelResponse(
                    invocations=[
                        {"tool_name": "harness_work", "arguments": {"value": "ok"}}
                    ]
                )
            ]
        )
        self.run_calls = 0
        self._context_env = dict(context_env or {})
        self._context_logs = list(context_logs or [])
        self._context_artifacts = list(context_artifacts or [])
        self._context_source_provenance = list(context_source_provenance or [])
        self._context_metadata = dict(context_metadata or {})

    def select_task_ids(self, *, state: HarnessState) -> list[str]:
        if state.session.stop_reason is not None:
            return []
        task = next(
            task for task in state.tasks if task.task_id == state.session.root_task_id
        )
        if task.status is TaskLifecycleStatus.COMPLETED:
            return []
        return list(self._selected_task_ids)

    def build_context(
        self,
        *,
        state: HarnessState,
        selected_task_ids: list[str],
        turn_index: int,
    ) -> ToolContext:
        del state, selected_task_ids
        return ToolContext(
            invocation_id=f"turn-{turn_index}",
            workspace=str(self.workspace),
            env=dict(self._context_env),
            logs=list(self._context_logs),
            artifacts=list(self._context_artifacts),
            source_provenance=list(self._context_source_provenance),
            metadata=dict(self._context_metadata),
        )

    def run_turn(
        self,
        *,
        state: HarnessState,
        selected_task_ids: list[str],
        context: ToolContext,
    ) -> ParsedModelResponse:
        del state, selected_task_ids, context
        self.run_calls += 1
        payload = self._payloads.pop(0)
        if isinstance(payload, Exception):
            raise payload
        return payload

    async def run_turn_async(
        self,
        *,
        state: HarnessState,
        selected_task_ids: list[str],
        context: ToolContext,
    ) -> ParsedModelResponse:
        return self.run_turn(
            state=state,
            selected_task_ids=selected_task_ids,
            context=context,
        )


class _RootTaskApplier:
    def __init__(self, *, stop_after_turn: bool = False) -> None:
        self.stop_after_turn = stop_after_turn

    def apply_turn(
        self,
        *,
        state: HarnessState,
        turn,
    ) -> tuple[HarnessState, TurnDecision]:
        root_task = next(
            task for task in state.tasks if task.task_id == state.session.root_task_id
        )
        workflow_result = turn.workflow_result
        assert workflow_result is not None

        if workflow_result.outcomes and workflow_result.outcomes[0].status in {
            WorkflowInvocationStatus.APPROVAL_DENIED,
            WorkflowInvocationStatus.APPROVAL_TIMED_OUT,
        }:
            updated_root = root_task.model_copy(
                update={
                    "status": TaskLifecycleStatus.BLOCKED,
                    "status_summary": workflow_result.outcomes[0].status.value,
                }
            )
        else:
            updated_root = root_task.model_copy(
                update={
                    "status": TaskLifecycleStatus.COMPLETED,
                    "finished_at": "2026-01-01T00:00:10Z",
                    "status_summary": "done",
                }
            )

        tasks = [
            updated_root if task.task_id == updated_root.task_id else task
            for task in state.tasks
        ]
        updated_state = state.model_copy(update={"tasks": tasks}, deep=True)
        if self.stop_after_turn:
            return updated_state, TurnDecision(
                action=TurnDecisionAction.STOP,
                selected_task_ids=list(turn.selected_task_ids),
                stop_reason=HarnessStopReason.ERROR,
                summary="forced stop",
            )
        return updated_state, TurnDecision(
            action=TurnDecisionAction.CONTINUE,
            selected_task_ids=list(turn.selected_task_ids),
            summary="continue",
        )


class _ConflictOnceStore(InMemoryHarnessStateStore):
    def __init__(self) -> None:
        super().__init__()
        self._raised_conflict = False

    def save_session(
        self,
        state: HarnessState,
        *,
        expected_revision: str | None = None,
        artifacts: StoredHarnessArtifacts | None = None,
    ):
        if expected_revision is not None and not self._raised_conflict:
            current = self.load_session(state.session.session_id)
            if current is not None and current.revision == expected_revision:
                self._raised_conflict = True
                super().save_session(
                    current.state,
                    expected_revision=expected_revision,
                    artifacts=current.artifacts if artifacts is None else artifacts,
                )
                raise HarnessStateConflictError("synthetic conflict")
        return super().save_session(
            state,
            expected_revision=expected_revision,
            artifacts=artifacts,
        )


@pytest.fixture
def _workflow_executor(tmp_path: Path) -> WorkflowExecutor:
    registry = ToolRegistry()
    registry.register(_WorkTool())
    registry.register(_EnvEchoTool())
    register_filesystem_tools(registry)
    return WorkflowExecutor(
        registry,
        policy=ToolPolicy(
            allowed_side_effects={
                SideEffectClass.NONE,
                SideEffectClass.LOCAL_READ,
                SideEffectClass.LOCAL_WRITE,
            },
            require_approval_for={SideEffectClass.LOCAL_READ},
        ),
    )


def _state(
    *, max_turns: int = 4, max_tool_invocations: int | None = None
) -> HarnessState:
    return create_root_task(
        schema_version=CURRENT_HARNESS_STATE_SCHEMA_VERSION,
        session_id="session-1",
        root_task_id="task-1",
        title="Root task",
        intent="Complete the request.",
        budget_policy=BudgetPolicy(
            max_turns=max_turns,
            max_tool_invocations=max_tool_invocations,
        ),
        started_at="2026-01-01T00:00:00Z",
    )


def _retryable_workflow_result() -> WorkflowTurnResult:
    request = {"tool_name": "harness_work", "arguments": {"value": "retry"}}
    return WorkflowTurnResult(
        parsed_response=ParsedModelResponse(invocations=[request]),
        outcomes=[
            WorkflowInvocationOutcome(
                invocation_index=1,
                request=request,
                status=WorkflowInvocationStatus.EXECUTED,
                tool_result=ToolResult(
                    ok=False,
                    tool_name="harness_work",
                    tool_version="0.1.0",
                    error=ToolError(
                        code=ErrorCode.EXECUTION_FAILED,
                        message="temporary failure",
                        retryable=True,
                    ),
                ),
            )
        ],
    )


def _success_workflow_result() -> WorkflowTurnResult:
    request = {"tool_name": "harness_work", "arguments": {"value": "ok"}}
    return WorkflowTurnResult(
        parsed_response=ParsedModelResponse(invocations=[request]),
        outcomes=[
            WorkflowInvocationOutcome(
                invocation_index=1,
                request=request,
                status=WorkflowInvocationStatus.EXECUTED,
                tool_result=ToolResult(
                    ok=True,
                    tool_name="harness_work",
                    tool_version="0.1.0",
                    output={"value": "ok"},
                ),
            )
        ],
    )


def test_harness_executor_runs_to_completion(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    store = InMemoryHarnessStateStore()
    driver = _StaticDriver(workspace=tmp_path)
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=driver,
        applier=_RootTaskApplier(),
    )

    result = executor.run(_state())

    assert isinstance(result, HarnessExecutionResult)
    assert result.resumed.disposition is ResumeDisposition.TERMINAL
    assert result.snapshot.state.session.stop_reason is HarnessStopReason.COMPLETED
    assert len(result.snapshot.state.turns) == 1
    assert result.snapshot.state.tasks[0].status is TaskLifecycleStatus.COMPLETED


def test_harness_executor_respects_pre_turn_budget_exhaustion(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    store = InMemoryHarnessStateStore()
    driver = _StaticDriver(workspace=tmp_path)
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=driver,
        applier=_RootTaskApplier(),
    )
    prior = _state(max_turns=1).model_copy(
        update={
            "session": _state(max_turns=1).session.model_copy(
                update={"current_turn_index": 1}
            ),
            "turns": [
                HarnessTurn(
                    turn_index=1,
                    started_at="2026-01-01T00:00:00Z",
                    selected_task_ids=["task-1"],
                    workflow_result=_success_workflow_result(),
                    decision=TurnDecision(
                        action=TurnDecisionAction.CONTINUE,
                        selected_task_ids=["task-1"],
                    ),
                    ended_at="2026-01-01T00:00:05Z",
                )
            ],
        },
        deep=True,
    )

    result = executor.run(prior)

    assert (
        result.snapshot.state.session.stop_reason is HarnessStopReason.BUDGET_EXHAUSTED
    )
    assert driver.run_calls == 0


def test_harness_executor_persists_pending_approval_and_can_resume(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    store = InMemoryHarnessStateStore()
    driver = _StaticDriver(
        workspace=tmp_path,
        payloads=[
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "list_directory", "arguments": {"path": "."}},
                    {
                        "tool_name": "write_file",
                        "arguments": {"path": "after.txt", "content": "approved"},
                    },
                ]
            )
        ],
        context_env={"PERSISTED_SECRET": "should-not-save"},
        context_logs=["driver-log"],
        context_artifacts=["artifact-ref"],
        context_source_provenance=[
            SourceProvenanceRef(
                source_kind="file",
                source_id="README.md",
                content_hash="abc123",
            )
        ],
        context_metadata={"keep": "yes"},
    )
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=driver,
        applier=_RootTaskApplier(),
    )

    waiting = executor.run(_state())

    assert waiting.resumed.disposition is ResumeDisposition.WAITING_FOR_APPROVAL
    assert len(waiting.snapshot.state.pending_approvals) == 1
    assert waiting.snapshot.state.turns[-1].decision is None
    pending = waiting.snapshot.state.pending_approvals[0]
    assert pending.base_context.metadata == {"keep": "yes"}
    assert pending.base_context.env == {}
    assert pending.base_context.logs == []
    assert pending.base_context.artifacts == []
    assert pending.base_context.source_provenance == []

    approved = executor.resume(
        waiting.snapshot.session_id,
        approval_resolution=ApprovalResolution.APPROVE,
    )

    assert approved.resumed.disposition is ResumeDisposition.TERMINAL
    assert approved.snapshot.state.session.stop_reason is HarnessStopReason.COMPLETED
    assert approved.snapshot.state.pending_approvals == []
    assert approved.snapshot.state.turns[-1].decision is not None
    assert (
        approved.snapshot.state.turns[-1].workflow_result.outcomes[0].status
        is WorkflowInvocationStatus.EXECUTED
    )


def test_harness_executor_resume_rehydrates_current_environment(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("HARNESS_RESUME_SECRET", raising=False)
    store = InMemoryHarnessStateStore()
    driver = _StaticDriver(
        workspace=tmp_path,
        payloads=[
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "list_directory", "arguments": {"path": "."}},
                    {
                        "tool_name": "env_echo",
                        "arguments": {"key": "HARNESS_RESUME_SECRET"},
                    },
                ]
            )
        ],
    )
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=driver,
        applier=_RootTaskApplier(),
    )

    waiting = executor.run(_state())
    monkeypatch.setenv("HARNESS_RESUME_SECRET", "available-on-resume")

    approved = executor.resume(
        waiting.snapshot.session_id,
        approval_resolution=ApprovalResolution.APPROVE,
    )

    outcome = approved.snapshot.state.turns[-1].workflow_result.outcomes[1]
    assert outcome.tool_result is not None
    assert outcome.tool_result.output == {"value": "available-on-resume"}


@pytest.mark.parametrize(
    ("resolution", "stop_reason"),
    [
        (ApprovalResolution.DENY, HarnessStopReason.APPROVAL_DENIED),
        (ApprovalResolution.EXPIRE, HarnessStopReason.APPROVAL_EXPIRED),
        (ApprovalResolution.CANCEL, HarnessStopReason.APPROVAL_CANCELED),
    ],
)
def test_harness_executor_forces_stop_after_non_approved_resolution(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    resolution: ApprovalResolution,
    stop_reason: HarnessStopReason,
) -> None:
    store = InMemoryHarnessStateStore()
    driver = _StaticDriver(
        workspace=tmp_path,
        payloads=[
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "list_directory", "arguments": {"path": "."}},
                    {
                        "tool_name": "write_file",
                        "arguments": {"path": "after.txt", "content": "done"},
                    },
                ]
            )
        ],
    )
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=driver,
        applier=_RootTaskApplier(),
    )

    waiting = executor.run(_state())
    resumed = executor.resume(
        waiting.snapshot.session_id,
        approval_resolution=resolution,
    )

    assert resumed.snapshot.state.session.stop_reason is stop_reason
    assert resumed.snapshot.state.pending_approvals == []
    assert resumed.snapshot.state.turns[-1].decision is not None


def test_harness_executor_retries_driver_failures_and_persists_retry_counts(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    store = InMemoryHarnessStateStore()
    driver = _StaticDriver(
        workspace=tmp_path,
        payloads=[
            RuntimeError("boom"),
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "harness_work", "arguments": {"value": "ok"}}
                ]
            ),
        ],
    )
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=driver,
        applier=_RootTaskApplier(),
        retry_policy=HarnessRetryPolicy(max_provider_retries=1),
    )

    result = executor.run(_state())
    task = result.snapshot.state.tasks[0]

    assert result.snapshot.state.session.retry_count == 1
    assert task.retry_count == 1
    assert driver.run_calls == 2


def test_harness_executor_retries_retryable_tool_errors(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryHarnessStateStore()
    driver = _StaticDriver(
        workspace=tmp_path,
        payloads=[
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "harness_work", "arguments": {"value": "ok"}}
                ]
            ),
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "harness_work", "arguments": {"value": "ok"}}
                ]
            ),
        ],
    )
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=driver,
        applier=_RootTaskApplier(),
        retry_policy=HarnessRetryPolicy(max_retryable_tool_retries=1),
    )
    results = [_retryable_workflow_result(), _success_workflow_result()]

    def _execute(
        parsed_response: ParsedModelResponse, context: ToolContext
    ) -> WorkflowTurnResult:
        del parsed_response, context
        return results.pop(0)

    monkeypatch.setattr(_workflow_executor, "execute_parsed_response", _execute)

    result = executor.run(_state())

    assert result.snapshot.state.session.retry_count == 1
    assert result.snapshot.state.tasks[0].retry_count == 1
    assert result.snapshot.state.session.stop_reason is HarnessStopReason.COMPLETED


def test_harness_executor_stops_when_retryable_errors_exhaust_budget(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryHarnessStateStore()
    driver = _StaticDriver(workspace=tmp_path)
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=driver,
        applier=_RootTaskApplier(),
        retry_policy=HarnessRetryPolicy(max_retryable_tool_retries=0),
    )

    monkeypatch.setattr(
        _workflow_executor,
        "execute_parsed_response",
        lambda parsed_response, context: _retryable_workflow_result(),
    )

    result = executor.run(_state())

    assert result.snapshot.state.session.stop_reason is HarnessStopReason.ERROR
    assert "Retry budget exhausted" in (
        result.snapshot.state.tasks[0].status_summary or ""
    )


def test_harness_executor_retries_persistence_conflicts_without_rerunning_turn(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    store = _ConflictOnceStore()
    driver = _StaticDriver(workspace=tmp_path)
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=driver,
        applier=_RootTaskApplier(),
        retry_policy=HarnessRetryPolicy(max_persistence_retries=1),
    )

    result = executor.run(_state())

    assert result.snapshot.revision == "5"


def test_harness_executor_sync_provider_retry_exhaustion_after_retry(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    driver = _StaticDriver(
        workspace=tmp_path,
        payloads=[RuntimeError("boom"), RuntimeError("boom")],
    )
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=driver,
        applier=_RootTaskApplier(),
        retry_policy=HarnessRetryPolicy(max_provider_retries=1),
    )

    result = executor.run(_state())

    assert result.snapshot.state.session.stop_reason is HarnessStopReason.NO_PROGRESS
    assert result.snapshot.state.tasks[0].retry_count == 1
    assert result.snapshot.state.turns[-1].no_progress_signals
    assert driver.run_calls == 2


def test_harness_executor_run_async_smoke(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryHarnessStateStore()
    driver = _StaticDriver(workspace=tmp_path)
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=driver,
        applier=_RootTaskApplier(),
    )

    async def _execute_async(
        parsed_response: ParsedModelResponse,
        context: ToolContext,
    ) -> WorkflowTurnResult:
        del parsed_response, context
        return _success_workflow_result()

    monkeypatch.setattr(
        _workflow_executor,
        "execute_parsed_response_async",
        _execute_async,
    )

    result = asyncio.run(executor.run_async(_state()))

    assert result.snapshot.state.session.stop_reason is HarnessStopReason.COMPLETED


class _EmptyDriver(_StaticDriver):
    def __init__(self, *, workspace: Path) -> None:
        super().__init__(workspace=workspace)

    def select_task_ids(self, *, state: HarnessState) -> list[str]:
        del state
        return []


class _MismatchedConflictStore(InMemoryHarnessStateStore):
    def save_session(
        self,
        state: HarnessState,
        *,
        expected_revision: str | None = None,
        artifacts: StoredHarnessArtifacts | None = None,
    ):
        if expected_revision is not None:
            current = self.load_session(state.session.session_id)
            if current is not None and current.revision == expected_revision:
                divergent = current.state.model_copy(
                    update={
                        "session": current.state.session.model_copy(
                            update={
                                "retry_count": current.state.session.retry_count + 1,
                            }
                        )
                    },
                    deep=True,
                )
                super().save_session(
                    divergent,
                    expected_revision=expected_revision,
                    artifacts=current.artifacts if artifacts is None else artifacts,
                )
                raise HarnessStateConflictError("divergent conflict")
        return super().save_session(
            state,
            expected_revision=expected_revision,
            artifacts=artifacts,
        )


def _prior_turn_state(*, max_tool_invocations: int = 2) -> HarnessState:
    prior = _state(max_turns=4, max_tool_invocations=max_tool_invocations)
    return prior.model_copy(
        update={
            "session": prior.session.model_copy(update={"current_turn_index": 1}),
            "turns": [
                HarnessTurn(
                    turn_index=1,
                    started_at="2026-01-01T00:00:00Z",
                    selected_task_ids=["task-1"],
                    workflow_result=_success_workflow_result(),
                    decision=TurnDecision(
                        action=TurnDecisionAction.CONTINUE,
                        selected_task_ids=["task-1"],
                    ),
                    ended_at="2026-01-01T00:00:05Z",
                )
            ],
        },
        deep=True,
    )


def test_harness_executor_marks_completed_when_no_tasks_selected(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_EmptyDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
    )

    result = executor.run(_state())

    assert result.snapshot.state.session.stop_reason is HarnessStopReason.NO_PROGRESS
    assert result.snapshot.state.turns[-1].decision is not None
    assert (
        result.snapshot.state.turns[-1].decision.stop_reason
        is HarnessStopReason.NO_PROGRESS
    )


def test_harness_executor_run_async_marks_no_progress_when_no_tasks_selected(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_EmptyDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
    )

    result = asyncio.run(executor.run_async(_state()))

    assert result.snapshot.state.session.stop_reason is HarnessStopReason.NO_PROGRESS
    assert result.snapshot.state.turns[-1].no_progress_signals


def test_harness_executor_respects_elapsed_budget_limit(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    state = create_root_task(
        schema_version=CURRENT_HARNESS_STATE_SCHEMA_VERSION,
        session_id="session-elapsed",
        root_task_id="task-1",
        title="Root task",
        intent="Complete the request.",
        budget_policy=BudgetPolicy(max_turns=4, max_elapsed_seconds=1),
        started_at="2026-01-01T00:00:00Z",
    )
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
    )

    result = executor.run(
        state,
        now=__import__("datetime").datetime(
            2026, 1, 1, 0, 0, 2, tzinfo=__import__("datetime").UTC
        ),
    )

    assert (
        result.snapshot.state.session.stop_reason is HarnessStopReason.BUDGET_EXHAUSTED
    )


def test_harness_executor_forces_budget_stop_after_tool_invocation_limit(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
    )

    result = executor.run(_prior_turn_state(max_tool_invocations=1))

    assert (
        result.snapshot.state.session.stop_reason is HarnessStopReason.BUDGET_EXHAUSTED
    )
    assert len(result.snapshot.state.turns) == 2


def test_harness_executor_resume_requires_approval_resolution(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(
            workspace=tmp_path,
            payloads=[
                ParsedModelResponse(
                    invocations=[
                        {"tool_name": "list_directory", "arguments": {"path": "."}}
                    ]
                )
            ],
        ),
        applier=_RootTaskApplier(),
    )
    waiting = executor.run(_state())

    with pytest.raises(ValueError, match="approval_resolution is required"):
        executor.resume(waiting.snapshot.session_id)


def test_harness_executor_resume_async_waiting_approval_roundtrip(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed_response = ParsedModelResponse(
        invocations=[{"tool_name": "list_directory", "arguments": {"path": "."}}]
    )
    approval_request = ApprovalRequest(
        approval_id="approval-async-1",
        invocation_index=1,
        request=parsed_response.invocations[0],
        tool_name="list_directory",
        tool_version="0.1.0",
        policy_reason="approval required",
        requested_at="2026-01-01T00:00:00Z",
        expires_at="2026-01-01T00:05:00Z",
    )

    async def _execute_async(
        model_response: ParsedModelResponse,
        context: ToolContext,
    ) -> WorkflowTurnResult:
        del context
        return WorkflowTurnResult(
            parsed_response=model_response,
            outcomes=[
                WorkflowInvocationOutcome(
                    invocation_index=1,
                    request=model_response.invocations[0],
                    status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                    approval_request=approval_request,
                )
            ],
        )

    async def _resume_async(record, resolution: str, *, now=None) -> WorkflowTurnResult:
        del record, resolution, now
        return _success_workflow_result()

    monkeypatch.setattr(
        _workflow_executor, "execute_parsed_response_async", _execute_async
    )
    monkeypatch.setattr(
        _workflow_executor, "resume_persisted_approval_async", _resume_async
    )

    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(
            workspace=tmp_path,
            payloads=[parsed_response],
        ),
        applier=_RootTaskApplier(),
    )

    waiting = asyncio.run(
        executor.run_async(_state(), now=datetime(2026, 1, 1, 0, 1, tzinfo=UTC))
    )
    resumed = asyncio.run(
        executor.resume_async(
            waiting.snapshot.session_id,
            approval_resolution=ApprovalResolution.APPROVE,
            now=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
        )
    )

    assert waiting.resumed.disposition is ResumeDisposition.WAITING_FOR_APPROVAL
    assert resumed.snapshot.state.session.stop_reason is HarnessStopReason.COMPLETED


def test_harness_executor_resume_budget_gate_blocks_approval_resume(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed_response = ParsedModelResponse(
        invocations=[
            {"tool_name": "list_directory", "arguments": {"path": "."}},
            {"tool_name": "harness_work", "arguments": {"value": "ok"}},
        ]
    )
    approval_request = ApprovalRequest(
        approval_id="approval-budget-1",
        invocation_index=1,
        request=parsed_response.invocations[0],
        tool_name="list_directory",
        tool_version="0.1.0",
        policy_reason="approval required",
        requested_at="2026-01-01T00:00:00Z",
        expires_at="2026-01-01T00:05:00Z",
    )
    pending_approval = PendingApprovalRecord(
        approval_request=approval_request,
        parsed_response=parsed_response,
        base_context=ToolContext(invocation_id="approval-budget"),
        pending_index=1,
    )
    resume_calls = {"count": 0}
    monkeypatch.setattr(
        _workflow_executor,
        "resume_persisted_approval",
        lambda *args, **kwargs: resume_calls.update(count=resume_calls["count"] + 1),
    )

    store = InMemoryHarnessStateStore()
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
    )
    waiting_state = _prior_turn_state(max_tool_invocations=2).model_copy(
        update={
            "session": _prior_turn_state(max_tool_invocations=2).session.model_copy(
                update={"current_turn_index": 2}
            ),
            "turns": [
                *_prior_turn_state(max_tool_invocations=2).turns,
                HarnessTurn(
                    turn_index=2,
                    started_at="2026-01-01T00:01:00Z",
                    selected_task_ids=["task-1"],
                    workflow_result=WorkflowTurnResult(
                        parsed_response=parsed_response,
                        outcomes=[
                            WorkflowInvocationOutcome(
                                invocation_index=1,
                                request=parsed_response.invocations[0],
                                status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                                approval_request=approval_request,
                            )
                        ],
                    ),
                ),
            ],
            "pending_approvals": [pending_approval],
        },
        deep=True,
    )
    waiting = store.save_session(waiting_state)

    resumed = executor.resume(
        waiting.session_id,
        approval_resolution=ApprovalResolution.APPROVE,
        now=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
    )

    assert resume_calls["count"] == 0
    assert (
        resumed.snapshot.state.session.stop_reason is HarnessStopReason.BUDGET_EXHAUSTED
    )


def test_harness_executor_resume_async_budget_gate_blocks_approval_resume(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed_response = ParsedModelResponse(
        invocations=[
            {"tool_name": "list_directory", "arguments": {"path": "."}},
            {"tool_name": "harness_work", "arguments": {"value": "ok"}},
        ]
    )
    approval_request = ApprovalRequest(
        approval_id="approval-budget-async-1",
        invocation_index=1,
        request=parsed_response.invocations[0],
        tool_name="list_directory",
        tool_version="0.1.0",
        policy_reason="approval required",
        requested_at="2026-01-01T00:00:00Z",
        expires_at="2026-01-01T00:05:00Z",
    )
    pending_approval = PendingApprovalRecord(
        approval_request=approval_request,
        parsed_response=parsed_response,
        base_context=ToolContext(invocation_id="approval-budget-async"),
        pending_index=1,
    )
    resume_calls = {"count": 0}

    async def _resume_async(*args, **kwargs):
        del args, kwargs
        resume_calls["count"] += 1
        return _success_workflow_result()

    monkeypatch.setattr(
        _workflow_executor, "resume_persisted_approval_async", _resume_async
    )

    store = InMemoryHarnessStateStore()
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
    )
    waiting_state = _prior_turn_state(max_tool_invocations=2).model_copy(
        update={
            "session": _prior_turn_state(max_tool_invocations=2).session.model_copy(
                update={"current_turn_index": 2}
            ),
            "turns": [
                *_prior_turn_state(max_tool_invocations=2).turns,
                HarnessTurn(
                    turn_index=2,
                    started_at="2026-01-01T00:01:00Z",
                    selected_task_ids=["task-1"],
                    workflow_result=WorkflowTurnResult(
                        parsed_response=parsed_response,
                        outcomes=[
                            WorkflowInvocationOutcome(
                                invocation_index=1,
                                request=parsed_response.invocations[0],
                                status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                                approval_request=approval_request,
                            )
                        ],
                    ),
                ),
            ],
            "pending_approvals": [pending_approval],
        },
        deep=True,
    )
    waiting = store.save_session(waiting_state)

    resumed = asyncio.run(
        executor.resume_async(
            waiting.session_id,
            approval_resolution=ApprovalResolution.APPROVE,
            now=datetime(2026, 1, 1, 0, 2, tzinfo=UTC),
        )
    )

    assert resume_calls["count"] == 0
    assert (
        resumed.snapshot.state.session.stop_reason is HarnessStopReason.BUDGET_EXHAUSTED
    )


def test_harness_executor_resume_async_requires_approval_resolution(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(
            workspace=tmp_path,
            payloads=[
                ParsedModelResponse(
                    invocations=[
                        {"tool_name": "list_directory", "arguments": {"path": "."}}
                    ]
                )
            ],
        ),
        applier=_RootTaskApplier(),
    )
    waiting = executor.run(_state())

    with pytest.raises(ValueError, match="approval_resolution is required"):
        asyncio.run(executor.resume_async(waiting.snapshot.session_id))


def test_harness_executor_rejects_non_expire_resolution_for_expired_approval(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed_response = ParsedModelResponse(
        invocations=[{"tool_name": "list_directory", "arguments": {"path": "."}}]
    )
    approval_request = ApprovalRequest(
        approval_id="approval-expired-1",
        invocation_index=1,
        request=parsed_response.invocations[0],
        tool_name="list_directory",
        tool_version="0.1.0",
        policy_reason="approval required",
        requested_at="2026-01-01T00:00:00Z",
        expires_at="2026-01-01T00:05:00Z",
    )

    monkeypatch.setattr(
        _workflow_executor,
        "execute_parsed_response",
        lambda model_response, context: WorkflowTurnResult(
            parsed_response=model_response,
            outcomes=[
                WorkflowInvocationOutcome(
                    invocation_index=1,
                    request=model_response.invocations[0],
                    status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                    approval_request=approval_request,
                )
            ],
        ),
    )

    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path, payloads=[parsed_response]),
        applier=_RootTaskApplier(),
    )
    waiting = executor.run(_state(), now=datetime(2026, 1, 1, 0, 1, tzinfo=UTC))

    with pytest.raises(ValueError, match="Expired approvals may only resume"):
        executor.resume(
            waiting.snapshot.session_id,
            approval_resolution=ApprovalResolution.DENY,
            now=datetime(2026, 1, 1, 0, 10, tzinfo=UTC),
        )


def test_harness_executor_async_provider_retry_exhaustion(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path, payloads=[RuntimeError("boom")]),
        applier=_RootTaskApplier(),
        retry_policy=HarnessRetryPolicy(max_provider_retries=0),
    )

    result = asyncio.run(executor.run_async(_state()))

    assert result.snapshot.state.session.stop_reason is HarnessStopReason.NO_PROGRESS
    assert "Provider retry budget exhausted" in (
        result.snapshot.state.tasks[0].status_summary or ""
    )


def test_harness_executor_resume_surfaces_interrupted_turn_until_replay_allowed(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(
            workspace=tmp_path,
            payloads=[
                RuntimeError("boom"),
                ParsedModelResponse(
                    invocations=[
                        {"tool_name": "harness_work", "arguments": {"value": "ok"}}
                    ]
                ),
            ],
        ),
        applier=_RootTaskApplier(),
        retry_policy=HarnessRetryPolicy(max_provider_retries=1),
    )
    snapshot = executor.run(
        _state(),
        now=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
    ).snapshot
    interrupted = snapshot.model_copy(
        update={
            "state": snapshot.state.model_copy(
                update={
                    "session": snapshot.state.session.model_copy(
                        update={"ended_at": None, "stop_reason": None}
                    ),
                    "turns": [
                        snapshot.state.turns[-1].model_copy(
                            update={"decision": None, "ended_at": None}
                        )
                    ],
                },
                deep=True,
            )
        },
        deep=True,
    )
    executor._store.save_session(  # type: ignore[attr-defined]
        interrupted.state,
        expected_revision=snapshot.revision,
        artifacts=interrupted.artifacts,
    )

    resumed = executor.resume(interrupted.session_id)

    assert resumed.resumed.disposition is ResumeDisposition.INTERRUPTED

    replayed = executor.resume(
        interrupted.session_id,
        allow_interrupted_turn_replay=True,
    )

    assert replayed.snapshot.state.session.stop_reason is HarnessStopReason.COMPLETED


def test_harness_executor_drive_replays_interrupted_tail_when_allowed(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryHarnessStateStore()
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
    )
    snapshot = store.save_session(_state())
    resumed_states = iter(
        [
            ResumeDisposition.INTERRUPTED,
            ResumeDisposition.RUNNABLE,
            ResumeDisposition.TERMINAL,
        ]
    )
    drop_calls = {"count": 0}
    run_calls = {"count": 0}

    monkeypatch.setattr(
        "llm_tools.harness_api.executor.resume_session",
        lambda snapshot, now=None: ResumedHarnessSession(
            snapshot=snapshot,
            disposition=next(resumed_states),
        ),
    )
    monkeypatch.setattr(
        executor,
        "_drop_interrupted_tail",
        lambda *, snapshot: (
            drop_calls.update(count=drop_calls["count"] + 1) or snapshot
        ),
    )
    monkeypatch.setattr(
        executor,
        "_run_one_turn",
        lambda *, snapshot, now: (
            run_calls.update(count=run_calls["count"] + 1) or (snapshot, False)
        ),
    )

    result = executor._drive(
        snapshot=snapshot,
        approval_resolution=None,
        allow_interrupted_turn_replay=True,
        now=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
    )

    assert result.snapshot.session_id == snapshot.session_id
    assert drop_calls["count"] == 1
    assert run_calls["count"] == 1


def test_harness_executor_drive_async_replays_interrupted_tail_when_allowed(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = InMemoryHarnessStateStore()
    executor = HarnessExecutor(
        store=store,
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
    )
    snapshot = store.save_session(_state())
    resumed_states = iter(
        [
            ResumeDisposition.INTERRUPTED,
            ResumeDisposition.RUNNABLE,
            ResumeDisposition.TERMINAL,
        ]
    )
    drop_calls = {"count": 0}
    run_calls = {"count": 0}

    monkeypatch.setattr(
        "llm_tools.harness_api.executor.resume_session",
        lambda snapshot, now=None: ResumedHarnessSession(
            snapshot=snapshot,
            disposition=next(resumed_states),
        ),
    )
    monkeypatch.setattr(
        executor,
        "_drop_interrupted_tail",
        lambda *, snapshot: (
            drop_calls.update(count=drop_calls["count"] + 1) or snapshot
        ),
    )

    async def _run_one_turn_async(*, snapshot, now):
        del now
        run_calls["count"] += 1
        return snapshot, False

    monkeypatch.setattr(executor, "_run_one_turn_async", _run_one_turn_async)

    result = asyncio.run(
        executor._drive_async(
            snapshot=snapshot,
            approval_resolution=None,
            allow_interrupted_turn_replay=True,
            now=datetime(2026, 1, 1, 0, 1, tzinfo=UTC),
        )
    )

    assert result.snapshot.session_id == snapshot.session_id
    assert drop_calls["count"] == 1
    assert run_calls["count"] == 1


def test_harness_executor_pre_dispatch_budget_gate_stops_before_execution(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    executed = {"count": 0}

    def _execute(
        parsed_response: ParsedModelResponse, context: ToolContext
    ) -> WorkflowTurnResult:
        del parsed_response, context
        executed["count"] += 1
        return _success_workflow_result()

    monkeypatch.setattr(_workflow_executor, "execute_parsed_response", _execute)
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(
            workspace=tmp_path,
            payloads=[
                ParsedModelResponse(
                    invocations=[
                        {"tool_name": "harness_work", "arguments": {"value": "1"}},
                        {"tool_name": "harness_work", "arguments": {"value": "2"}},
                    ]
                )
            ],
        ),
        applier=_RootTaskApplier(),
    )

    result = executor.run(_prior_turn_state(max_tool_invocations=2))

    assert executed["count"] == 0
    assert (
        result.snapshot.state.session.stop_reason is HarnessStopReason.BUDGET_EXHAUSTED
    )


def test_harness_executor_async_retryable_tool_error_exhaustion(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _execute_async(
        parsed_response: ParsedModelResponse,
        context: ToolContext,
    ) -> WorkflowTurnResult:
        del parsed_response, context
        return _retryable_workflow_result()

    monkeypatch.setattr(
        _workflow_executor, "execute_parsed_response_async", _execute_async
    )
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
        retry_policy=HarnessRetryPolicy(max_retryable_tool_retries=0),
    )

    result = asyncio.run(executor.run_async(_state()))

    assert result.snapshot.state.session.stop_reason is HarnessStopReason.ERROR


def test_harness_executor_async_budget_stop_after_tool_invocation_limit(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async def _execute_async(
        parsed_response: ParsedModelResponse,
        context: ToolContext,
    ) -> WorkflowTurnResult:
        del parsed_response, context
        return _success_workflow_result()

    monkeypatch.setattr(
        _workflow_executor, "execute_parsed_response_async", _execute_async
    )
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
    )

    result = asyncio.run(executor.run_async(_prior_turn_state(max_tool_invocations=1)))

    assert (
        result.snapshot.state.session.stop_reason is HarnessStopReason.BUDGET_EXHAUSTED
    )


def test_harness_executor_conflict_retry_raises_for_divergent_state(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    executor = HarnessExecutor(
        store=_MismatchedConflictStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
        retry_policy=HarnessRetryPolicy(max_persistence_retries=1),
    )

    with pytest.raises(HarnessStateConflictError, match="divergent conflict"):
        executor.run(_state())


def test_harness_executor_resume_rejects_unknown_session(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
    )

    with pytest.raises(ValueError, match="Unknown session id"):
        executor.resume("missing-session")

    with pytest.raises(ValueError, match="Unknown session id"):
        asyncio.run(executor.resume_async("missing-session"))


def test_harness_executor_helper_methods_cover_error_and_false_paths(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
    )
    turn = HarnessTurn(
        turn_index=1,
        started_at="2026-01-01T00:00:00Z",
        selected_task_ids=["task-1"],
        workflow_result=WorkflowTurnResult(
            parsed_response=ParsedModelResponse(final_response="done")
        ),
        decision=TurnDecision(action=TurnDecisionAction.CONTINUE),
        ended_at="2026-01-01T00:00:01Z",
    )
    state = _state().model_copy(
        update={
            "session": _state().session.model_copy(update={"current_turn_index": 1}),
            "turns": [turn],
        },
        deep=True,
    )

    assert executor._count_persisted_tool_invocations(state) == 0
    assert executor._has_retryable_tool_error(_success_workflow_result()) is False
    assert executor._approval_stop_reason(ApprovalResolution.APPROVE) is None
    assert executor._parse_timestamp("2026-01-01T00:00:00").tzinfo is not None

    with pytest.raises(
        ValueError,
        match="approval_requested workflow outcomes must include approval_request",
    ):
        executor._approval_request_from_result(
            WorkflowTurnResult.model_construct(
                parsed_response=ParsedModelResponse(
                    invocations=[
                        {"tool_name": "list_directory", "arguments": {"path": "."}}
                    ]
                ),
                outcomes=[
                    WorkflowInvocationOutcome.model_construct(
                        invocation_index=1,
                        request={
                            "tool_name": "list_directory",
                            "arguments": {"path": "."},
                        },
                        status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                        approval_request=None,
                        tool_result=None,
                    )
                ],
            )
        )


def test_harness_executor_helper_methods_cover_elapsed_budget_and_retryable_paths(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(workspace=tmp_path),
        applier=_RootTaskApplier(),
    )
    executed_turn = HarnessTurn(
        turn_index=2,
        started_at="2026-01-01T00:00:00Z",
        selected_task_ids=["task-1"],
        workflow_result=_success_workflow_result(),
        decision=TurnDecision(action=TurnDecisionAction.CONTINUE),
        ended_at="2026-01-01T00:00:01Z",
    )
    base_state = _state(max_turns=10, max_tool_invocations=1)
    state = base_state.model_copy(
        update={
            "session": base_state.session.model_copy(update={"current_turn_index": 1}),
            "turns": [
                HarnessTurn(
                    turn_index=1,
                    started_at="2026-01-01T00:00:00Z",
                    selected_task_ids=["task-1"],
                    decision=TurnDecision(action=TurnDecisionAction.CONTINUE),
                    ended_at="2026-01-01T00:00:00Z",
                ),
                executed_turn,
            ],
        },
        deep=True,
    )

    assert executor._count_persisted_tool_invocations(state) == 1
    assert executor._has_retryable_tool_error(_retryable_workflow_result()) is True
    assert (
        executor._pre_turn_budget_stop(
            state.model_copy(
                update={
                    "session": state.session.model_copy(
                        update={
                            "budget_policy": BudgetPolicy(
                                max_turns=10,
                                max_elapsed_seconds=1,
                            )
                        }
                    )
                },
                deep=True,
            ),
            now=executor._parse_timestamp("2026-01-01T00:00:02Z"),
        )
        is HarnessStopReason.BUDGET_EXHAUSTED
    )

    approval_request = ApprovalRequest(
        approval_id="approval-helper-1",
        invocation_index=2,
        request={"tool_name": "harness_work", "arguments": {"value": "ok"}},
        tool_name="harness_work",
        tool_version="0.1.0",
        policy_reason="approval required",
        requested_at="2026-01-01T00:00:00Z",
        expires_at="2026-01-01T00:05:00Z",
    )
    pending_approval = PendingApprovalRecord(
        approval_request=approval_request,
        parsed_response=ParsedModelResponse(
            invocations=[
                {"tool_name": "harness_work", "arguments": {"value": "1"}},
                {"tool_name": "harness_work", "arguments": {"value": "2"}},
            ]
        ),
        base_context=ToolContext(invocation_id="approval-helper"),
        pending_index=2,
    )

    assert (
        executor._resume_would_exceed_budget(
            state=state, pending_approval=pending_approval
        )
        is True
    )
    assert (
        executor._exceeds_tool_invocation_budget(state, _success_workflow_result())
        is True
    )
    assert (
        executor._empty_selection_signals(
            state=_state(),
            turn_index=1,
            now=executor._parse_timestamp("2026-01-01T00:00:01Z"),
        )[0].summary
        == "Planner found no actionable tasks while non-terminal work remains."
    )
    blocked_state = _state().model_copy(
        update={
            "tasks": [
                _state()
                .tasks[0]
                .model_copy(update={"status": TaskLifecycleStatus.BLOCKED})
            ]
        },
        deep=True,
    )
    assert (
        executor._empty_selection_signals(
            state=blocked_state,
            turn_index=1,
            now=executor._parse_timestamp("2026-01-01T00:00:01Z"),
        )[0].task_id
        == "task-1"
    )
    assert (
        len(
            executor._retry_exhaustion_signals(
                state=_state(),
                task_ids=["task-1", "missing"],
                turn_index=1,
                now=executor._parse_timestamp("2026-01-01T00:00:01Z"),
            )
        )
        == 1
    )
    assert (
        executor._post_turn_no_progress_signals(
            previous_state=state,
            updated_state=state.model_copy(
                update={
                    "tasks": [
                        state.tasks[0].model_copy(
                            update={"status": TaskLifecycleStatus.COMPLETED}
                        )
                    ]
                },
                deep=True,
            ),
            turn=executed_turn,
            decision=TurnDecision(
                action=TurnDecisionAction.CONTINUE,
                selected_task_ids=["task-1"],
            ),
            now=executor._parse_timestamp("2026-01-01T00:00:02Z"),
        )
        == []
    )


def test_harness_executor_scrubs_completed_turn_when_protection_requests_purge(
    tmp_path: Path,
    _workflow_executor: WorkflowExecutor,
) -> None:
    executor = HarnessExecutor(
        store=InMemoryHarnessStateStore(),
        workflow_executor=_workflow_executor,
        driver=_StaticDriver(
            workspace=tmp_path,
            payloads=[ParsedModelResponse(final_response="secret")],
            context_metadata={
                "protection_review": {
                    "purge_requested": True,
                    "safe_message": "safe",
                }
            },
        ),
        applier=_RootTaskApplier(),
    )

    result = executor.run(_state())

    assert result.snapshot.state.turns[-1].workflow_result is not None
    assert (
        result.snapshot.state.turns[-1].workflow_result.parsed_response.final_response
        == "safe"
    )
