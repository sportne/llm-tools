"""Best-effort protection scrubbing helpers for persisted harness state."""

from __future__ import annotations

from typing import Any

from llm_tools.harness_api.models import HarnessState
from llm_tools.tool_api import ProtectionProvenanceSnapshot, ToolResult
from llm_tools.workflow_api import (
    WorkflowInvocationOutcome,
    WorkflowTurnResult,
    collect_provenance_from_tool_results,
)

DEFAULT_PURGED_RESPONSE = "[WITHHELD BY PROTECTION]"


def collect_state_provenance(
    state: HarnessState | object,
) -> ProtectionProvenanceSnapshot:
    """Collect tool-result provenance from a harness state snapshot."""
    if not isinstance(state, HarnessState):
        return ProtectionProvenanceSnapshot()

    tool_results: list[ToolResult] = []
    for turn in state.turns:
        workflow_result = turn.workflow_result
        if workflow_result is None:
            continue
        for outcome in workflow_result.outcomes:
            if outcome.tool_result is not None:
                tool_results.append(outcome.tool_result)
    return collect_provenance_from_tool_results(tool_results)


def scrub_state_for_protection(
    state: HarnessState,
    *,
    safe_message: Any | None = None,
) -> HarnessState:
    """Scrub persisted workflow payloads before saving a protected session."""
    if not state.turns:
        return state
    changed = False
    scrubbed_turns = []
    for index, turn in enumerate(state.turns):
        workflow_result = turn.workflow_result
        if workflow_result is None:
            scrubbed_turns.append(turn)
            continue
        changed = True
        scrubbed_turns.append(
            turn.model_copy(
                update={
                    "workflow_result": scrub_workflow_result(
                        workflow_result,
                        safe_message=(
                            safe_message if index == len(state.turns) - 1 else None
                        ),
                    )
                }
            )
        )
    if not changed:
        return state
    return state.model_copy(
        update={
            "turns": scrubbed_turns,
            "pending_approvals": [],
        },
        deep=True,
    )


def scrub_workflow_result(
    workflow_result: WorkflowTurnResult,
    *,
    safe_message: Any | None = None,
) -> WorkflowTurnResult:
    """Return a workflow result with scrubbed tool payloads and final response."""
    parsed_response = workflow_result.parsed_response
    if parsed_response.final_response is None and not any(
        outcome.tool_result is not None for outcome in workflow_result.outcomes
    ):
        return workflow_result
    scrubbed_response = parsed_response
    if parsed_response.final_response is not None:
        replacement = (
            safe_message if safe_message is not None else DEFAULT_PURGED_RESPONSE
        )
        scrubbed_response = parsed_response.model_copy(
            update={"final_response": replacement}
        )
    scrubbed_outcomes = [
        _scrub_workflow_outcome(outcome) for outcome in workflow_result.outcomes
    ]
    return workflow_result.model_copy(
        update={
            "parsed_response": scrubbed_response,
            "outcomes": scrubbed_outcomes,
        }
    )


def _scrub_workflow_outcome(
    outcome: WorkflowInvocationOutcome,
) -> WorkflowInvocationOutcome:
    if outcome.tool_result is None:
        return outcome
    return outcome.model_copy(
        update={"tool_result": _scrub_tool_result(outcome.tool_result)}
    )


def _scrub_tool_result(tool_result: ToolResult) -> ToolResult:
    metadata = dict(tool_result.metadata)
    execution_record = metadata.get("execution_record")
    if isinstance(execution_record, dict):
        scrubbed_record = dict(execution_record)
        scrubbed_record.pop("validated_output", None)
        scrubbed_record.pop("redacted_output", None)
        scrubbed_record["logs"] = []
        scrubbed_record["artifacts"] = []
        metadata["execution_record"] = scrubbed_record
    return tool_result.model_copy(
        update={
            "output": None,
            "logs": [],
            "artifacts": [],
            "metadata": metadata,
        },
        deep=True,
    )


__all__ = [
    "collect_state_provenance",
    "DEFAULT_PURGED_RESPONSE",
    "scrub_state_for_protection",
    "scrub_workflow_result",
]
