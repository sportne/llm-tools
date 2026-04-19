"""Golden trace and replay regression tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from llm_tools.apps.chat_runtime import build_chat_executor
from llm_tools.harness_api import (
    ApprovalResolution,
    BudgetPolicy,
    HarnessSessionCreateRequest,
    HarnessSessionResumeRequest,
    HarnessSessionRunRequest,
    HarnessSessionService,
    HarnessSessionTrace,
    HarnessStopReason,
    HarnessTurn,
    HarnessTurnTrace,
    InMemoryHarnessStateStore,
    ScriptedParsedResponseProvider,
    TurnDecisionAction,
    replay_session,
)
from llm_tools.harness_api.replay import build_turn_trace
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import SideEffectClass, ToolInvocationRequest, ToolPolicy
from llm_tools.workflow_api import ApprovalRequest, WorkflowInvocationStatus

_GOLDEN_DIR = Path(__file__).with_name("golden")


def test_success_trace_matches_golden_fixture() -> None:
    _, workflow_executor = build_chat_executor()
    service = HarnessSessionService(
        store=InMemoryHarnessStateStore(),
        workflow_executor=workflow_executor,
        provider=ScriptedParsedResponseProvider(
            [ParsedModelResponse(final_response="done")]
        ),
        workspace=".",
    )
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Root task",
            intent="Complete",
            budget_policy=BudgetPolicy(max_turns=3),
            session_id="golden-success",
            started_at="2026-01-01T00:00:00Z",
        )
    )
    result = service.run_session(
        HarnessSessionRunRequest(session_id=created.session_id)
    )

    payload = {
        "trace": _normalize(result.snapshot.artifacts.trace.model_dump(mode="json")),
        "replay": _normalize(replay_session(result.snapshot).model_dump(mode="json")),
    }

    assert json.dumps(payload, sort_keys=True) == json.dumps(
        _load_golden("success_trace.json"), sort_keys=True
    )


def test_approval_trace_matches_golden_fixture() -> None:
    _, workflow_executor = build_chat_executor(
        policy=ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
            require_approval_for={SideEffectClass.LOCAL_READ},
            allow_network=False,
            allow_filesystem=True,
            allow_subprocess=False,
        )
    )
    service = HarnessSessionService(
        store=InMemoryHarnessStateStore(),
        workflow_executor=workflow_executor,
        provider=ScriptedParsedResponseProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        {"tool_name": "list_directory", "arguments": {"path": "."}}
                    ]
                )
            ]
        ),
        workspace=".",
    )
    created = service.create_session(
        HarnessSessionCreateRequest(
            title="Approval task",
            intent="Need approval",
            budget_policy=BudgetPolicy(max_turns=3),
            session_id="golden-approval",
            started_at="2026-01-01T00:00:00Z",
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

    payload = {
        "trace": _normalize(approved.snapshot.artifacts.trace.model_dump(mode="json")),
        "replay": _normalize(replay_session(approved.snapshot).model_dump(mode="json")),
    }

    assert payload["replay"]["final_stop_reason"] == "completed"
    assert payload["replay"]["steps"][0]["workflow_outcome_statuses"] == [
        "approval_requested",
        "executed",
    ]
    assert payload["trace"]["final_stop_reason"] == "completed"
    invocation_traces = payload["trace"]["turns"][0]["invocation_traces"]
    assert [trace["status"] for trace in invocation_traces] == [
        "approval_requested",
        "executed",
    ]
    assert invocation_traces[0]["policy_snapshot"]["reason"] == "approval required"
    assert invocation_traces[1]["policy_snapshot"]["reason"] == "approved"
    assert invocation_traces[0]["redacted_arguments"] == {}
    assert invocation_traces[1]["redacted_arguments"] == {
        "path": ".",
        "recursive": False,
        "max_depth": None,
    }


def _load_golden(filename: str) -> dict[str, Any]:
    return json.loads((_GOLDEN_DIR / filename).read_text(encoding="utf-8"))


def _normalize(value: Any) -> Any:
    if isinstance(value, dict):
        normalized: dict[str, Any] = {}
        for key, item in value.items():
            if key in {
                "saved_at",
                "started_at",
                "ended_at",
                "requested_at",
                "expires_at",
                "recorded_at",
                "checked_at",
            }:
                normalized[key] = f"<{key}>"
            elif key in {"approval_id", "pending_approval_id"}:
                normalized[key] = "<approval_id>" if item is not None else None
            else:
                normalized[key] = _normalize(item)
        return normalized
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def test_turn_trace_validators_reject_invalid_shapes() -> None:
    with pytest.raises(ValueError, match="selected_task_ids must be unique"):
        HarnessTurnTrace(
            turn_index=1,
            started_at="2026-01-01T00:00:00Z",
            selected_task_ids=["task-1", "task-1"],
        )

    with pytest.raises(ValueError, match="decision_stop_reason is only allowed"):
        HarnessTurnTrace(
            turn_index=1,
            started_at="2026-01-01T00:00:00Z",
            selected_task_ids=["task-1"],
            decision_action=TurnDecisionAction.CONTINUE,
            decision_stop_reason=HarnessStopReason.ERROR,
        )


def test_session_trace_validators_reject_duplicate_turn_indices() -> None:
    turn = HarnessTurnTrace(
        turn_index=1,
        started_at="2026-01-01T00:00:00Z",
        selected_task_ids=["task-1"],
    )

    with pytest.raises(ValueError, match="unique ascending indices"):
        HarnessSessionTrace(session_id="session-1", turns=[turn, turn])


def test_build_turn_trace_preserves_pending_approval_without_workflow_result() -> None:
    turn = HarnessTurn(
        turn_index=1,
        started_at="2026-01-01T00:00:00Z",
        selected_task_ids=["task-1"],
        pending_approval_request=ApprovalRequest(
            approval_id="approval-1",
            invocation_index=1,
            request=ToolInvocationRequest(
                tool_name="list_directory",
                arguments={"path": "."},
            ),
            tool_name="list_directory",
            tool_version="0.1.0",
            policy_reason="approval required",
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T01:00:00Z",
        ),
    )

    trace = build_turn_trace(
        turn=turn,
        context=None,
        tasks_state=None,
    )

    assert turn.pending_approval_request is not None
    assert "request" not in turn.pending_approval_request.model_dump(mode="json")
    assert trace.pending_approval_id == "approval-1"
    assert trace.workflow_outcome_statuses == [
        WorkflowInvocationStatus.APPROVAL_REQUESTED
    ]
    assert len(trace.invocation_traces) == 1
    assert (
        trace.invocation_traces[0].status is WorkflowInvocationStatus.APPROVAL_REQUESTED
    )
    assert trace.invocation_traces[0].redacted_arguments == {}
