"""Default harness driver and applier implementations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from llm_tools.harness_api.context import (
    DefaultHarnessContextBuilder,
    HarnessContextBuilder,
)
from llm_tools.harness_api.executor_approvals import HarnessRetryPolicy
from llm_tools.harness_api.models import (
    HarnessState,
    HarnessStopReason,
    HarnessTurn,
    TaskLifecycleStatus,
    TurnDecision,
    TurnDecisionAction,
)
from llm_tools.harness_api.planning import DeterministicHarnessPlanner, HarnessPlanner
from llm_tools.harness_api.tasks import (
    block_task,
    complete_task,
    fail_task,
    start_task,
)
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import ToolContext
from llm_tools.workflow_api import WorkflowExecutor
from llm_tools.workflow_api.executor import PreparedModelInteraction


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


__all__ = [
    "DefaultHarnessTurnDriver",
    "HarnessModelTurnProvider",
    "MinimalHarnessTurnApplier",
    "ScriptedParsedResponseProvider",
]
