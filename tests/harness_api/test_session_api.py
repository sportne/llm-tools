"""End-to-end tests for the public harness session API."""

from __future__ import annotations

from pathlib import Path

from llm_tools.apps.chat_runtime import build_chat_executor
from llm_tools.harness_api import (
    ApprovalResolution,
    BudgetPolicy,
    HarnessRetryPolicy,
    HarnessSessionCreateRequest,
    HarnessSessionInspectRequest,
    HarnessSessionListRequest,
    HarnessSessionResumeRequest,
    HarnessSessionRunRequest,
    HarnessSessionService,
    HarnessSessionStopRequest,
    HarnessSessionSummary,
    HarnessStopReason,
    InMemoryHarnessStateStore,
    ScriptedParsedResponseProvider,
    StoredHarnessArtifacts,
    TaskLifecycleStatus,
    TurnDecision,
    TurnDecisionAction,
    VerificationOutcome,
    VerificationStatus,
)
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import (
    ErrorCode,
    SideEffectClass,
    ToolContext,
    ToolError,
    ToolPolicy,
    ToolResult,
)
from llm_tools.workflow_api import (
    WorkflowInvocationOutcome,
    WorkflowInvocationStatus,
    WorkflowTurnResult,
)


class _SingleTurnDriver:
    def select_task_ids(self, *, state):
        del state
        return ["task-1"]

    def build_context(self, *, state, selected_task_ids, turn_index):
        del state, selected_task_ids
        return ToolContext(invocation_id=f"turn-{turn_index}")

    def run_turn(self, *, state, selected_task_ids, context):
        del state, selected_task_ids, context
        return ParsedModelResponse(final_response="keep going")

    async def run_turn_async(self, *, state, selected_task_ids, context):
        return self.run_turn(
            state=state,
            selected_task_ids=selected_task_ids,
            context=context,
        )


class _ContinueApplier:
    def apply_turn(self, *, state, turn):
        return state, TurnDecision(
            action=TurnDecisionAction.CONTINUE,
            selected_task_ids=list(turn.selected_task_ids),
            summary="continue",
        )


def _service(
    responses: list[ParsedModelResponse],
    *,
    workspace: Path,
    retry_policy: HarnessRetryPolicy | None = None,
    require_approval_for_read: bool = False,
) -> HarnessSessionService:
    store = InMemoryHarnessStateStore()
    _, workflow_executor = build_chat_executor(
        policy=ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
            require_approval_for=(
                {SideEffectClass.LOCAL_READ} if require_approval_for_read else set()
            ),
            allow_network=False,
            allow_filesystem=True,
            allow_subprocess=False,
        )
    )
    return HarnessSessionService(
        store=store,
        workflow_executor=workflow_executor,
        provider=ScriptedParsedResponseProvider(responses),
        workspace=str(workspace),
        retry_policy=retry_policy,
    )


def test_session_service_runs_simple_scripted_session_to_completion(
    tmp_path: Path,
) -> None:
    service = _service(
        [ParsedModelResponse(final_response="done")],
        workspace=tmp_path,
        require_approval_for_read=True,
    )
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Root task",
            intent="Complete the request.",
            budget_policy=BudgetPolicy(max_turns=3),
            session_id="session-success",
        )
    )

    result = service.run_session(
        HarnessSessionRunRequest(session_id=created.session_id)
    )
    inspection = service.inspect_session(
        HarnessSessionInspectRequest(
            session_id=result.snapshot.session_id,
            include_replay=True,
        )
    )

    assert result.snapshot.state.session.stop_reason is HarnessStopReason.COMPLETED
    assert inspection.summary.completed_task_ids == ["task-1"]
    assert inspection.replay is not None
    assert len(inspection.replay.steps) == 1
    assert inspection.snapshot.artifacts.trace is not None


def test_session_service_handles_approval_wait_then_approve(tmp_path: Path) -> None:
    service = _service(
        [
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "list_directory", "arguments": {"path": "."}}
                ]
            )
        ],
        workspace=tmp_path,
        require_approval_for_read=True,
    )
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Approval task",
            intent="Needs approval.",
            budget_policy=BudgetPolicy(max_turns=3),
            session_id="session-approval-approve",
        )
    )

    waiting = service.run_session(
        HarnessSessionRunRequest(session_id=created.session_id)
    )
    approved = service.resume_session(
        HarnessSessionResumeRequest(
            session_id=waiting.snapshot.session_id,
            approval_resolution=ApprovalResolution.APPROVE,
        )
    )

    assert waiting.resumed.disposition.value == "waiting_for_approval"
    assert waiting.snapshot.state.turns[0].pending_approval_request is not None
    assert "request" not in waiting.snapshot.state.turns[
        0
    ].pending_approval_request.model_dump(mode="json")
    assert approved.snapshot.state.session.stop_reason is HarnessStopReason.COMPLETED
    assert approved.snapshot.state.turns[0].pending_approval_request is not None
    assert "request" not in approved.snapshot.state.turns[
        0
    ].pending_approval_request.model_dump(mode="json")

    inspection = service.inspect_session(
        HarnessSessionInspectRequest(
            session_id=approved.snapshot.session_id,
            include_replay=True,
        )
    )

    trace = inspection.snapshot.artifacts.trace
    assert trace is not None
    assert trace.turns[0].pending_approval_id is not None
    assert trace.turns[0].invocation_traces[0].policy_snapshot is not None
    assert trace.turns[0].invocation_traces[0].policy_snapshot.requires_approval is True
    assert [outcome.status for outcome in trace.turns[0].invocation_traces] == [
        WorkflowInvocationStatus.APPROVAL_REQUESTED,
        WorkflowInvocationStatus.EXECUTED,
    ]
    assert inspection.replay is not None
    assert inspection.replay.steps[0].workflow_outcome_statuses == [
        WorkflowInvocationStatus.APPROVAL_REQUESTED,
        WorkflowInvocationStatus.EXECUTED,
    ]


def test_session_service_handles_approval_deny(tmp_path: Path) -> None:
    service = _service(
        [
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "list_directory", "arguments": {"path": "."}}
                ]
            )
        ],
        workspace=tmp_path,
        require_approval_for_read=True,
    )
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Approval deny",
            intent="Needs approval.",
            budget_policy=BudgetPolicy(max_turns=3),
            session_id="session-approval-deny",
        )
    )

    waiting = service.run_session(
        HarnessSessionRunRequest(session_id=created.session_id)
    )
    denied = service.resume_session(
        HarnessSessionResumeRequest(
            session_id=waiting.snapshot.session_id,
            approval_resolution=ApprovalResolution.DENY,
        )
    )

    assert (
        denied.snapshot.state.session.stop_reason is HarnessStopReason.APPROVAL_DENIED
    )
    assert denied.snapshot.state.tasks[0].status is TaskLifecycleStatus.BLOCKED
    assert denied.snapshot.state.turns[0].pending_approval_request is not None
    assert "request" not in denied.snapshot.state.turns[
        0
    ].pending_approval_request.model_dump(mode="json")

    inspection = service.inspect_session(
        HarnessSessionInspectRequest(
            session_id=denied.snapshot.session_id,
            include_replay=True,
        )
    )

    assert inspection.snapshot.artifacts.trace is not None
    assert inspection.snapshot.artifacts.trace.turns[0].workflow_outcome_statuses == [
        WorkflowInvocationStatus.APPROVAL_REQUESTED,
        WorkflowInvocationStatus.APPROVAL_DENIED,
    ]
    assert inspection.replay is not None
    assert inspection.replay.steps[0].workflow_outcome_statuses == [
        WorkflowInvocationStatus.APPROVAL_REQUESTED,
        WorkflowInvocationStatus.APPROVAL_DENIED,
    ]


def test_session_service_stop_finalizes_incomplete_tail_turn(tmp_path: Path) -> None:
    service = _service(
        [
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "list_directory", "arguments": {"path": "."}}
                ]
            )
        ],
        workspace=tmp_path,
        require_approval_for_read=True,
    )
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Stop me",
            intent="Wait for approval.",
            budget_policy=BudgetPolicy(max_turns=3),
            session_id="session-stop",
        )
    )
    service.run_session(HarnessSessionRunRequest(session_id=created.session_id))

    inspection = service.stop_session(
        HarnessSessionStopRequest(session_id=created.session_id)
    )

    assert inspection.snapshot.state.session.stop_reason is HarnessStopReason.CANCELED
    assert inspection.snapshot.state.turns[-1].decision is not None
    assert inspection.snapshot.artifacts.trace is not None
    assert (
        inspection.snapshot.artifacts.trace.turns[-1].decision_stop_reason
        is HarnessStopReason.CANCELED
    )


def test_session_service_surfaces_retryable_tool_retries(
    tmp_path: Path, monkeypatch
) -> None:
    service = _service(
        [
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "list_directory", "arguments": {"path": "."}}
                ]
            ),
            ParsedModelResponse(final_response="done"),
        ],
        workspace=tmp_path,
        retry_policy=HarnessRetryPolicy(max_retryable_tool_retries=1),
    )
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Retry session",
            intent="Retry tool work.",
            budget_policy=BudgetPolicy(max_turns=3),
            session_id="session-retry",
        )
    )
    results = [
        WorkflowTurnResult(
            parsed_response=ParsedModelResponse(
                invocations=[
                    {"tool_name": "list_directory", "arguments": {"path": "."}}
                ]
            ),
            outcomes=[
                WorkflowInvocationOutcome(
                    invocation_index=1,
                    request={"tool_name": "list_directory", "arguments": {"path": "."}},
                    status=WorkflowInvocationStatus.EXECUTED,
                    tool_result=ToolResult(
                        ok=False,
                        tool_name="list_directory",
                        tool_version="0.1.0",
                        error=ToolError(
                            code=ErrorCode.EXECUTION_FAILED,
                            message="temporary failure",
                            retryable=True,
                        ),
                    ),
                )
            ],
        ),
        WorkflowTurnResult(
            parsed_response=ParsedModelResponse(final_response="done"),
            outcomes=[],
        ),
    ]
    monkeypatch.setattr(
        service._executor._workflow_executor,
        "execute_parsed_response",
        lambda parsed_response, context: results.pop(0),
    )

    result = service.run_session(
        HarnessSessionRunRequest(session_id=created.session_id)
    )

    assert result.snapshot.state.session.retry_count == 1
    assert result.snapshot.state.session.stop_reason is HarnessStopReason.COMPLETED


def test_session_service_supports_injected_driver_and_applier_for_budget_stop(
    tmp_path: Path,
) -> None:
    store = InMemoryHarnessStateStore()
    _, workflow_executor = build_chat_executor()
    service = HarnessSessionService(
        store=store,
        workflow_executor=workflow_executor,
        driver=_SingleTurnDriver(),
        applier=_ContinueApplier(),
    )
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Budget session",
            intent="Use injected contracts.",
            budget_policy=BudgetPolicy(max_turns=1),
            session_id="session-budget",
        )
    )

    result = service.run_session(
        HarnessSessionRunRequest(session_id=created.session_id)
    )

    assert (
        result.snapshot.state.session.stop_reason is HarnessStopReason.BUDGET_EXHAUSTED
    )


def test_session_service_inspect_rebuilds_artifacts_from_canonical_state(
    tmp_path: Path,
) -> None:
    service = _service([ParsedModelResponse(final_response="done")], workspace=tmp_path)
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Canonical",
            intent="Normalize inspection output.",
            budget_policy=BudgetPolicy(max_turns=3),
            session_id="session-canonical-inspect",
        )
    )
    result = service.run_session(
        HarnessSessionRunRequest(session_id=created.session_id)
    )

    trace = result.snapshot.artifacts.trace
    assert trace is not None
    tampered = result.snapshot.model_copy(
        update={
            "artifacts": StoredHarnessArtifacts(
                summary=HarnessSessionSummary(
                    session_id=result.snapshot.session_id,
                    current_turn_index=99,
                    total_turns=99,
                    stop_reason=HarnessStopReason.ERROR,
                    completed_task_ids=["fake-task"],
                    active_task_ids=["fake-task"],
                    pending_approval_ids=["approval-fake"],
                    verification_status_counts={"failed": 9},
                    latest_decision_summary="tampered summary",
                ),
                trace={
                    "session_id": result.snapshot.session_id,
                    "final_stop_reason": "error",
                    "turns": [
                        {
                            "turn_index": 1,
                            "started_at": trace.turns[0].started_at,
                            "selected_task_ids": ["task-1"],
                            "workflow_outcome_statuses": ["approval_requested"],
                            "invocation_traces": [
                                {
                                    "invocation_index": 1,
                                    "status": "executed",
                                    "tool_name": "write_file",
                                    "redacted_arguments": {"secret": "tampered"},
                                }
                            ],
                            "decision_action": "stop",
                            "decision_stop_reason": "error",
                            "decision_summary": "tampered trace",
                        }
                    ],
                },
            )
        },
        deep=True,
    )
    service._store._snapshots[result.snapshot.session_id] = tampered

    inspection = service.inspect_session(
        HarnessSessionInspectRequest(
            session_id=result.snapshot.session_id,
            include_replay=True,
        )
    )

    assert inspection.summary.completed_task_ids == ["task-1"]
    assert (
        inspection.summary.latest_decision_summary == "Model returned a final response."
    )
    assert inspection.snapshot.artifacts.summary == inspection.summary
    assert inspection.snapshot.artifacts.trace is not None
    assert (
        inspection.snapshot.artifacts.trace.final_stop_reason
        is HarnessStopReason.COMPLETED
    )
    assert inspection.snapshot.artifacts.trace.turns[0].decision_summary == (
        "Model returned a final response."
    )
    assert inspection.snapshot.artifacts.trace.turns[0].invocation_traces == []
    assert inspection.replay is not None
    assert inspection.replay.final_stop_reason is HarnessStopReason.COMPLETED
    assert inspection.replay.steps[0].workflow_outcome_statuses == []


def test_session_service_inspect_preserves_turn_verification_history(
    tmp_path: Path,
) -> None:
    service = _service([ParsedModelResponse(final_response="done")], workspace=tmp_path)
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Canonical verification",
            intent="Preserve turn verification snapshots.",
            budget_policy=BudgetPolicy(max_turns=3),
            session_id="session-canonical-verification",
        )
    )
    result = service.run_session(
        HarnessSessionRunRequest(session_id=created.session_id)
    )

    mutated = result.snapshot.model_copy(
        update={
            "state": result.snapshot.state.model_copy(
                update={
                    "tasks": [
                        result.snapshot.state.tasks[0].model_copy(
                            update={
                                "verification": VerificationOutcome(
                                    status=VerificationStatus.PASSED,
                                    checked_at="2026-01-01T00:00:00Z",
                                )
                            }
                        )
                    ],
                    "turns": [
                        result.snapshot.state.turns[0].model_copy(
                            update={
                                "verification_status_by_task_id": {"task-1": "not_run"}
                            }
                        )
                    ],
                },
                deep=True,
            )
        },
        deep=True,
    )
    service._store._snapshots[result.snapshot.session_id] = mutated

    inspection = service.inspect_session(
        HarnessSessionInspectRequest(session_id=result.snapshot.session_id)
    )

    assert inspection.summary.verification_status_counts == {"passed": 1}
    assert inspection.snapshot.artifacts.trace is not None
    assert inspection.snapshot.artifacts.trace.turns[
        0
    ].verification_status_by_task_id == {"task-1": "not_run"}


def test_session_service_inspect_omits_historical_verification_when_snapshot_missing(
    tmp_path: Path,
) -> None:
    service = _service([ParsedModelResponse(final_response="done")], workspace=tmp_path)
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Canonical verification omission",
            intent="Do not backfill historical verification state.",
            budget_policy=BudgetPolicy(max_turns=3),
            session_id="session-canonical-verification-omission",
        )
    )
    result = service.run_session(
        HarnessSessionRunRequest(session_id=created.session_id)
    )

    mutated = result.snapshot.model_copy(
        update={
            "state": result.snapshot.state.model_copy(
                update={
                    "tasks": [
                        result.snapshot.state.tasks[0].model_copy(
                            update={
                                "verification": VerificationOutcome(
                                    status=VerificationStatus.PASSED,
                                    checked_at="2026-01-01T00:00:00Z",
                                )
                            }
                        )
                    ],
                    "turns": [
                        result.snapshot.state.turns[0].model_copy(
                            update={"verification_status_by_task_id": {}}
                        )
                    ],
                },
                deep=True,
            )
        },
        deep=True,
    )
    service._store._snapshots[result.snapshot.session_id] = mutated

    inspection = service.inspect_session(
        HarnessSessionInspectRequest(session_id=result.snapshot.session_id)
    )

    assert inspection.summary.verification_status_counts == {"passed": 1}
    assert inspection.snapshot.artifacts.trace is not None
    assert (
        inspection.snapshot.artifacts.trace.turns[0].verification_status_by_task_id
        == {}
    )


def test_session_service_list_rebuilds_summary_from_canonical_state(
    tmp_path: Path,
) -> None:
    service = _service([ParsedModelResponse(final_response="done")], workspace=tmp_path)
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Canonical list",
            intent="Normalize list output.",
            budget_policy=BudgetPolicy(max_turns=3),
            session_id="session-canonical-list",
        )
    )
    result = service.run_session(
        HarnessSessionRunRequest(session_id=created.session_id)
    )

    tampered = result.snapshot.model_copy(
        update={
            "artifacts": StoredHarnessArtifacts(
                summary=HarnessSessionSummary(
                    session_id=result.snapshot.session_id,
                    current_turn_index=0,
                    total_turns=0,
                    stop_reason=HarnessStopReason.ERROR,
                    completed_task_ids=[],
                    active_task_ids=["task-1"],
                    verification_status_counts={"failed": 1},
                    latest_decision_summary="tampered list summary",
                )
            )
        },
        deep=True,
    )
    service._store._snapshots[result.snapshot.session_id] = tampered

    listed = service.list_sessions(HarnessSessionListRequest(include_replay=False))

    assert len(listed.sessions) == 1
    assert listed.sessions[0].summary.completed_task_ids == ["task-1"]
    assert listed.sessions[0].summary.latest_decision_summary == (
        "Model returned a final response."
    )
    assert listed.sessions[0].snapshot.artifacts.summary == listed.sessions[0].summary
