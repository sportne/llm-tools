"""Additional branch coverage for harness session surfaces."""

from __future__ import annotations

import asyncio
from typing import cast

import pytest

from llm_tools.apps.chat_runtime import build_chat_executor
from llm_tools.harness_api import (
    BudgetPolicy,
    DefaultHarnessTurnDriver,
    HarnessSessionCreateRequest,
    HarnessSessionInspectRequest,
    HarnessSessionResumeRequest,
    HarnessSessionRunRequest,
    HarnessSessionService,
    InMemoryHarnessStateStore,
    MinimalHarnessTurnApplier,
    ReplanningTrigger,
    ScriptedParsedResponseProvider,
    TaskLifecycleStatus,
    TaskSelection,
    TurnDecisionAction,
    create_root_task,
)
from llm_tools.harness_api.models import HarnessTurn
from llm_tools.harness_api.planning import HarnessPlanner
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import ErrorCode, ToolContext, ToolError, ToolResult
from llm_tools.workflow_api import (
    WorkflowInvocationOutcome,
    WorkflowInvocationStatus,
    WorkflowTurnResult,
)


class _TriggerPlanner(HarnessPlanner):
    def select_tasks(self, *, state):
        del state
        return TaskSelection(selected_task_ids=["task-1"])

    def detect_replanning_triggers(
        self,
        *,
        previous_state,
        current_state,
        previous_selected_task_ids,
        decision,
    ):
        del previous_state, current_state, previous_selected_task_ids, decision
        return [ReplanningTrigger.SELECT_TASKS_REQUESTED]


class _AsyncProvider:
    def run(self, **kwargs):
        del kwargs
        return ParsedModelResponse(final_response="sync")

    async def run_async(self, **kwargs):
        del kwargs
        return ParsedModelResponse(final_response="async")


def test_scripted_provider_run_async_and_exhaustion() -> None:
    provider = ScriptedParsedResponseProvider(
        [ParsedModelResponse(final_response="done")]
    )
    state = create_root_task(
        schema_version="3",
        session_id="session-provider",
        root_task_id="task-1",
        title="Provider task",
        intent="Run provider.",
        budget_policy=BudgetPolicy(max_turns=2),
        started_at="2026-01-01T00:00:00Z",
    )

    async_result = asyncio.run(
        provider.run_async(
            state=state,
            selected_task_ids=["task-1"],
            context=ToolContext(invocation_id="turn-1"),
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=cast(object, None),
        )
    )
    assert async_result.final_response == "done"

    exhausted_state = state.model_copy(
        update={"session": state.session.model_copy(update={"current_turn_index": 1})}
    )
    with pytest.raises(RuntimeError, match="No scripted parsed response remains"):
        provider.run(
            state=exhausted_state,
            selected_task_ids=["task-1"],
            context=ToolContext(invocation_id="turn-2"),
            adapter=ActionEnvelopeAdapter(),
            prepared_interaction=cast(object, None),
        )


def test_default_turn_driver_records_replanning_metadata_and_async_result() -> None:
    _, workflow_executor = build_chat_executor()
    driver = DefaultHarnessTurnDriver(
        workflow_executor=workflow_executor,
        provider=_AsyncProvider(),
        planner=_TriggerPlanner(),
        workspace=".",
    )
    state = create_root_task(
        schema_version="3",
        session_id="session-driver",
        root_task_id="task-1",
        title="Driver task",
        intent="Drive one turn.",
        budget_policy=BudgetPolicy(max_turns=2),
        started_at="2026-01-01T00:00:00Z",
    )

    selected = driver.select_task_ids(state=state)
    first_context = driver.build_context(
        state=state,
        selected_task_ids=selected,
        turn_index=1,
    )
    assert first_context.metadata["harness_planning"]["replanning_triggers"] == []

    second_context = driver.build_context(
        state=state,
        selected_task_ids=selected,
        turn_index=2,
    )
    assert second_context.metadata["harness_planning"]["replanning_triggers"] == [
        "select_tasks_requested"
    ]

    parsed = asyncio.run(
        driver.run_turn_async(
            state=state,
            selected_task_ids=selected,
            context=second_context,
        )
    )
    assert parsed.final_response == "async"


def test_minimal_turn_applier_handles_missing_workflow_result() -> None:
    applier = MinimalHarnessTurnApplier()
    state = create_root_task(
        schema_version="3",
        session_id="session-no-workflow",
        root_task_id="task-1",
        title="No workflow",
        intent="Missing workflow result.",
        budget_policy=BudgetPolicy(max_turns=2),
        started_at="2026-01-01T00:00:00Z",
    )
    turn = HarnessTurn(
        turn_index=1,
        started_at="2026-01-01T00:00:00Z",
        selected_task_ids=["task-1"],
    )

    _, decision = applier.apply_turn(state=state, turn=turn)

    assert decision.action is TurnDecisionAction.STOP
    assert decision.summary == "Harness turn completed without a workflow_result."


def test_minimal_turn_applier_fails_tasks_on_error() -> None:
    applier = MinimalHarnessTurnApplier()
    state = create_root_task(
        schema_version="3",
        session_id="session-failure",
        root_task_id="task-1",
        title="Failure task",
        intent="Fail the task.",
        budget_policy=BudgetPolicy(max_turns=2),
        started_at="2026-01-01T00:00:00Z",
    )
    turn = HarnessTurn(
        turn_index=1,
        started_at="2026-01-01T00:00:00Z",
        selected_task_ids=["task-1"],
        workflow_result=WorkflowTurnResult(
            parsed_response=ParsedModelResponse(
                invocations=[{"tool_name": "noop", "arguments": {}}]
            ),
            outcomes=[
                WorkflowInvocationOutcome(
                    invocation_index=1,
                    request={"tool_name": "noop", "arguments": {}},
                    status=WorkflowInvocationStatus.EXECUTED,
                    tool_result=ToolResult(
                        ok=False,
                        tool_name="noop",
                        tool_version="0.1.0",
                        error=ToolError(
                            code=ErrorCode.EXECUTION_FAILED,
                            message="boom",
                        ),
                    ),
                )
            ],
        ),
    )

    updated_state, decision = applier.apply_turn(state=state, turn=turn)

    assert updated_state.tasks[0].status is TaskLifecycleStatus.FAILED
    assert decision.action is TurnDecisionAction.STOP
    assert decision.summary == "boom"


def test_session_service_requires_provider_without_driver() -> None:
    _, workflow_executor = build_chat_executor()
    with pytest.raises(ValueError, match="provider is required"):
        HarnessSessionService(
            store=InMemoryHarnessStateStore(),
            workflow_executor=workflow_executor,
        )


def test_session_service_unknown_session_errors() -> None:
    _, workflow_executor = build_chat_executor()
    service = HarnessSessionService(
        store=InMemoryHarnessStateStore(),
        workflow_executor=workflow_executor,
        provider=ScriptedParsedResponseProvider([]),
        workspace=".",
    )

    with pytest.raises(ValueError, match="Unknown session id"):
        service.run_session(HarnessSessionRunRequest(session_id="missing"))
    with pytest.raises(ValueError, match="Unknown session id"):
        asyncio.run(
            service.run_session_async(HarnessSessionRunRequest(session_id="missing"))
        )
    with pytest.raises(ValueError, match="Unknown session id"):
        service.inspect_session(HarnessSessionInspectRequest(session_id="missing"))


def test_session_service_resume_async_runs_session() -> None:
    store = InMemoryHarnessStateStore()
    _, workflow_executor = build_chat_executor()
    service = HarnessSessionService(
        store=store,
        workflow_executor=workflow_executor,
        provider=ScriptedParsedResponseProvider(
            [ParsedModelResponse(final_response="done")]
        ),
        workspace=".",
    )
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Async resume",
            intent="Resume asynchronously.",
            budget_policy=BudgetPolicy(max_turns=2),
            session_id="session-resume-async",
        )
    )

    result = asyncio.run(
        service.resume_session_async(
            HarnessSessionResumeRequest(session_id=created.session_id)
        )
    )

    assert result.snapshot.state.session.stop_reason is not None
