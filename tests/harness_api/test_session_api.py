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
    HarnessSessionResumeRequest,
    HarnessSessionRunRequest,
    HarnessSessionService,
    HarnessSessionStopRequest,
    HarnessStopReason,
    InMemoryHarnessStateStore,
    ScriptedParsedResponseProvider,
    TaskLifecycleStatus,
    TurnDecision,
    TurnDecisionAction,
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
    assert approved.snapshot.state.session.stop_reason is HarnessStopReason.COMPLETED
    trace = approved.snapshot.artifacts.trace
    assert trace is not None
    assert trace.turns[0].pending_approval_id is not None
    assert trace.turns[0].invocation_traces[0].policy_snapshot is not None
    assert trace.turns[0].invocation_traces[0].policy_snapshot.requires_approval is True


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
