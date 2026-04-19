"""Best-effort protection scrubbing helpers for persisted harness state."""

from __future__ import annotations

from typing import Any

from llm_tools.harness_api.models import HarnessState
from llm_tools.workflow_api import WorkflowTurnResult

DEFAULT_PURGED_RESPONSE = "[WITHHELD BY PROTECTION]"


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
    "DEFAULT_PURGED_RESPONSE",
    "scrub_state_for_protection",
    "scrub_workflow_result",
]
