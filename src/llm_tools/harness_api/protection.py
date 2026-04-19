"""Best-effort protection scrubbing helpers for persisted harness state."""

from __future__ import annotations

from typing import Any

from llm_tools.harness_api.models import HarnessState
from llm_tools.tool_api import ProtectionProvenanceSnapshot, ToolResult
from llm_tools.workflow_api import (
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
    """Replace the tail turn final response with a safe placeholder before save."""
    if not state.turns:
        return state
    tail_turn = state.turns[-1]
    if tail_turn.workflow_result is None:
        return state
    scrubbed_result = scrub_workflow_result(
        tail_turn.workflow_result,
        safe_message=safe_message,
    )
    scrubbed_turn = tail_turn.model_copy(update={"workflow_result": scrubbed_result})
    return state.model_copy(
        update={
            "turns": [*state.turns[:-1], scrubbed_turn],
            "pending_approvals": [],
        },
        deep=True,
    )


def scrub_workflow_result(
    workflow_result: WorkflowTurnResult,
    *,
    safe_message: Any | None = None,
) -> WorkflowTurnResult:
    """Return a workflow result whose final response no longer carries raw text."""
    parsed_response = workflow_result.parsed_response
    if parsed_response.final_response is None:
        return workflow_result
    replacement = safe_message if safe_message is not None else DEFAULT_PURGED_RESPONSE
    scrubbed_response = parsed_response.model_copy(
        update={"final_response": replacement}
    )
    return workflow_result.model_copy(update={"parsed_response": scrubbed_response})


__all__ = [
    "collect_state_provenance",
    "DEFAULT_PURGED_RESPONSE",
    "scrub_state_for_protection",
    "scrub_workflow_result",
]
