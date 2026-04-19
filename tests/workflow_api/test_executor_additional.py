"""Additional coverage tests for workflow executor edge paths."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime, timedelta

import pytest

import llm_tools.tool_api.runtime as runtime_module
from llm_tools.harness_api import PendingApprovalRecord
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import SideEffectClass, ToolContext, ToolPolicy, ToolRegistry
from llm_tools.tools.filesystem import register_filesystem_tools
from llm_tools.workflow_api import WorkflowExecutor
from llm_tools.workflow_api.models import ApprovalRequest, WorkflowInvocationStatus


@pytest.fixture(autouse=True)
def _inline_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(runtime_module.asyncio, "to_thread", _run_inline)


def _build_executor() -> WorkflowExecutor:
    registry = ToolRegistry()
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
            approval_timeout_seconds=1,
        ),
    )


def _persisted_record(tmp_path: str) -> PendingApprovalRecord:
    parsed_response = ParsedModelResponse(
        invocations=[
            {"tool_name": "list_directory", "arguments": {"path": "."}},
            {
                "tool_name": "write_file",
                "arguments": {"path": "after.txt", "content": "continued"},
            },
        ]
    )
    return PendingApprovalRecord(
        approval_request=ApprovalRequest(
            approval_id="approval-async-record",
            invocation_index=1,
            request=parsed_response.invocations[0],
            tool_name="list_directory",
            tool_version="0.1.0",
            policy_reason="approval required",
            policy_metadata={"tool_name": "list_directory"},
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:05:00Z",
        ),
        parsed_response=parsed_response,
        base_context=ToolContext(
            invocation_id="persisted-async",
            workspace=str(tmp_path),
        ),
        pending_index=1,
    )


def test_workflow_executor_export_tools_and_pending_approval_errors(
    tmp_path: str,
) -> None:
    executor = _build_executor()
    exported = executor.export_tools(
        ActionEnvelopeAdapter(),
        context=ToolContext(invocation_id="export", workspace=str(tmp_path)),
    )
    assert isinstance(exported, dict)
    assert "ActionEnvelope" in exported["title"]

    final_only = executor.execute_parsed_response(
        ParsedModelResponse(final_response="done"),
        ToolContext(invocation_id="final-only"),
    )
    assert final_only.outcomes == []

    with pytest.raises(ValueError, match="Unknown approval id"):
        executor.cancel_pending_approval("missing")
    with pytest.raises(ValueError, match="Unknown approval id"):
        executor.resolve_pending_approval("missing", approved=True)

    record = _persisted_record(tmp_path)
    with pytest.raises(ValueError, match="Unsupported approval resolution"):
        executor.resume_persisted_approval(record, "later")


def test_workflow_executor_async_pending_paths(tmp_path: str) -> None:
    async def run() -> None:
        executor = _build_executor()

        final_only = await executor.execute_parsed_response_async(
            ParsedModelResponse(final_response="async done"),
            ToolContext(invocation_id="async-final-only"),
        )
        assert final_only.outcomes == []

        seeded = await executor.execute_model_output_async(
            ActionEnvelopeAdapter(),
            {
                "actions": [
                    {"tool_name": "list_directory", "arguments": {"path": "."}},
                    {
                        "tool_name": "write_file",
                        "arguments": {"path": "after.txt", "content": "continued"},
                    },
                ]
            },
            ToolContext(invocation_id="async-seeded", workspace=str(tmp_path)),
        )
        approval_id = seeded.outcomes[0].approval_request.approval_id  # type: ignore[union-attr]

        with pytest.raises(ValueError, match="Unknown approval id"):
            await executor.resolve_pending_approval_async("missing", approved=True)

        approved = await executor.resolve_pending_approval_async(
            approval_id,
            approved=True,
        )
        assert [outcome.status for outcome in approved.outcomes] == [
            WorkflowInvocationStatus.EXECUTED,
            WorkflowInvocationStatus.EXECUTED,
        ]

        timeout_executor = _build_executor()
        await timeout_executor.execute_model_output_async(
            ActionEnvelopeAdapter(),
            {
                "actions": [
                    {"tool_name": "list_directory", "arguments": {"path": "."}},
                    {
                        "tool_name": "write_file",
                        "arguments": {"path": "after.txt", "content": "continued"},
                    },
                ]
            },
            ToolContext(invocation_id="async-timeout", workspace=str(tmp_path)),
        )
        timed_out = await timeout_executor.finalize_expired_approvals_async(
            now=datetime.now(UTC) + timedelta(seconds=5)
        )
        assert len(timed_out) == 1
        assert (
            timed_out[0].outcomes[0].status
            is WorkflowInvocationStatus.APPROVAL_TIMED_OUT
        )

        denied = await executor.resume_persisted_approval_async(
            _persisted_record(tmp_path),
            "deny",
        )
        assert denied.outcomes[0].status is WorkflowInvocationStatus.APPROVAL_DENIED
        assert denied.outcomes[1].status is WorkflowInvocationStatus.EXECUTED

        expired = await executor.resume_persisted_approval_async(
            _persisted_record(tmp_path),
            "expire",
        )
        assert expired.outcomes[0].status is WorkflowInvocationStatus.APPROVAL_TIMED_OUT
        assert expired.outcomes[1].status is WorkflowInvocationStatus.EXECUTED

        unknown_tool = await executor.execute_parsed_response_async(
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "missing_tool", "arguments": {"path": "README.md"}}
                ]
            ),
            ToolContext(invocation_id="async-missing-tool", workspace=str(tmp_path)),
        )
        assert unknown_tool.outcomes[0].status is WorkflowInvocationStatus.EXECUTED
        assert unknown_tool.outcomes[0].tool_result is not None
        assert unknown_tool.outcomes[0].tool_result.ok is False

    asyncio.run(run())
