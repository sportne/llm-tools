"""End-to-end integration tests for the workflow executor."""

from __future__ import annotations

import asyncio
import subprocess
import sys
from datetime import UTC, datetime, timedelta
from types import ModuleType

import pytest
from pydantic import BaseModel, ValidationError

import llm_tools.tool_api.runtime as runtime_module
import llm_tools.tools.git.tools as git_tools
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import (
    ErrorCode,
    SideEffectClass,
    Tool,
    ToolContext,
    ToolPolicy,
    ToolRegistry,
    ToolResult,
    ToolSpec,
)
from llm_tools.tools.atlassian import register_atlassian_tools
from llm_tools.tools.filesystem import register_filesystem_tools
from llm_tools.tools.git import register_git_tools
from llm_tools.tools.text import register_text_tools
from llm_tools.workflow_api import WorkflowExecutor, WorkflowTurnResult
from llm_tools.workflow_api.models import (
    ApprovalRequest,
    WorkflowInvocationOutcome,
    WorkflowInvocationStatus,
)


@pytest.fixture(autouse=True)
def _inline_to_thread(monkeypatch: pytest.MonkeyPatch) -> None:
    async def _run_inline(func: object, /, *args: object, **kwargs: object) -> object:
        return func(*args, **kwargs)

    monkeypatch.setattr(runtime_module.asyncio, "to_thread", _run_inline)


class _AsyncEchoInput(BaseModel):
    value: str


class _AsyncEchoOutput(BaseModel):
    value: str


class _AsyncEchoTool(Tool[_AsyncEchoInput, _AsyncEchoOutput]):
    spec = ToolSpec(
        name="async_echo",
        description="Async echo tool for workflow tests.",
        side_effects=SideEffectClass.NONE,
    )
    input_model = _AsyncEchoInput
    output_model = _AsyncEchoOutput

    async def ainvoke(
        self, context: ToolContext, args: _AsyncEchoInput
    ) -> _AsyncEchoOutput:
        context.logs.append("async-echo")
        return _AsyncEchoOutput(value=f"{context.invocation_id}:{args.value}")


def _executor(registry: ToolRegistry, *, allow_write: bool = False) -> WorkflowExecutor:
    allowed_side_effects = {
        SideEffectClass.NONE,
        SideEffectClass.LOCAL_READ,
        SideEffectClass.EXTERNAL_READ,
    }
    if allow_write:
        allowed_side_effects.add(SideEffectClass.LOCAL_WRITE)

    return WorkflowExecutor(
        registry,
        policy=ToolPolicy(allowed_side_effects=allowed_side_effects),
    )


def _executed_tool_results(result: WorkflowTurnResult) -> list[ToolResult]:
    return [
        outcome.tool_result
        for outcome in result.outcomes
        if outcome.status is WorkflowInvocationStatus.EXECUTED
        and outcome.tool_result is not None
    ]


def _approval_request(index: int = 1) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=f"approval-{index}",
        invocation_index=index,
        request={"tool_name": "read_file", "arguments": {"path": "README.md"}},
        tool_name="read_file",
        tool_version="0.1.0",
        policy_reason="approval required",
        policy_metadata={"tool_name": "read_file"},
        requested_at="2026-01-01T00:00:00Z",
        expires_at="2026-01-01T00:05:00Z",
    )


def test_workflow_turn_result_rejects_mismatched_final_response_and_outcomes() -> None:
    with pytest.raises(ValidationError):
        WorkflowTurnResult(
            parsed_response=ParsedModelResponse(final_response="done"),
            outcomes=[
                WorkflowInvocationOutcome(
                    invocation_index=1,
                    request={"tool_name": "read_file", "arguments": {}},
                    status=WorkflowInvocationStatus.EXECUTED,
                    tool_result=ToolResult(
                        ok=True,
                        tool_name="read_file",
                        tool_version="0.1.0",
                        output={},
                    ),
                )
            ],
        )


def test_workflow_turn_result_rejects_outcome_indices_outside_parsed_invocations() -> (
    None
):
    with pytest.raises(ValidationError):
        WorkflowTurnResult(
            parsed_response=ParsedModelResponse(
                invocations=[
                    {
                        "tool_name": "read_file",
                        "arguments": {},
                    },
                    {
                        "tool_name": "list_directory",
                        "arguments": {},
                    },
                ]
            ),
            outcomes=[
                WorkflowInvocationOutcome(
                    invocation_index=3,
                    request={"tool_name": "read_file", "arguments": {}},
                    status=WorkflowInvocationStatus.EXECUTED,
                    tool_result=ToolResult(
                        ok=True,
                        tool_name="read_file",
                        tool_version="0.1.0",
                        output={},
                    ),
                )
            ],
        )


def test_workflow_invocation_outcome_validators_reject_invalid_payload_shapes() -> None:
    with pytest.raises(ValidationError):
        WorkflowInvocationOutcome(
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {}},
            status=WorkflowInvocationStatus.EXECUTED,
        )

    with pytest.raises(ValidationError):
        WorkflowInvocationOutcome(
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {}},
            status=WorkflowInvocationStatus.EXECUTED,
            tool_result=ToolResult(
                ok=True,
                tool_name="read_file",
                tool_version="0.1.0",
                output={},
            ),
            approval_request=_approval_request(),
        )

    with pytest.raises(ValidationError):
        WorkflowInvocationOutcome(
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {}},
            status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
        )

    with pytest.raises(ValidationError):
        WorkflowInvocationOutcome(
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {}},
            status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
            approval_request=_approval_request(),
            tool_result=ToolResult(
                ok=False,
                tool_name="read_file",
                tool_version="0.1.0",
                error={"code": "policy_denied", "message": "denied"},
            ),
        )


def test_workflow_turn_result_rejects_non_ascending_outcome_indices() -> None:
    with pytest.raises(ValidationError):
        WorkflowTurnResult(
            parsed_response=ParsedModelResponse(
                invocations=[
                    {"tool_name": "read_file", "arguments": {}},
                    {"tool_name": "list_directory", "arguments": {}},
                ]
            ),
            outcomes=[
                WorkflowInvocationOutcome(
                    invocation_index=2,
                    request={"tool_name": "list_directory", "arguments": {}},
                    status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                    approval_request=_approval_request(index=2),
                ),
                WorkflowInvocationOutcome(
                    invocation_index=1,
                    request={"tool_name": "read_file", "arguments": {}},
                    status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                    approval_request=_approval_request(index=1),
                ),
            ],
        )


def test_workflow_executor_prepares_registered_tools_for_model_interaction() -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    register_text_tools(registry)
    executor = _executor(registry)

    prepared = executor.prepare_model_interaction(ActionEnvelopeAdapter())

    assert prepared.tool_names == [
        "read_file",
        "write_file",
        "list_directory",
        "find_files",
        "get_file_info",
        "search_text",
    ]
    assert isinstance(prepared.schema, dict)
    assert "ActionEnvelope" in prepared.schema["title"]


def test_workflow_executor_export_filters_out_policy_denied_tools_with_context(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry, allow_write=False)

    prepared = executor.prepare_model_interaction(
        ActionEnvelopeAdapter(),
        context=ToolContext(invocation_id="export-1", workspace=str(tmp_path)),
    )

    assert prepared.tool_names == [
        "read_file",
        "list_directory",
        "find_files",
        "get_file_info",
    ]


def test_workflow_executor_export_can_include_approval_required_tools(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    policy = ToolPolicy(
        allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
        require_approval_for={SideEffectClass.LOCAL_READ},
    )
    executor = WorkflowExecutor(registry, policy=policy)

    hidden_export = executor.prepare_model_interaction(
        ActionEnvelopeAdapter(),
        context=ToolContext(invocation_id="export-2", workspace=str(tmp_path)),
    )
    shown_export = executor.prepare_model_interaction(
        ActionEnvelopeAdapter(),
        context=ToolContext(invocation_id="export-3", workspace=str(tmp_path)),
        include_requires_approval=True,
    )

    assert hidden_export.tool_names == []
    assert shown_export.tool_names == [
        "read_file",
        "list_directory",
        "find_files",
        "get_file_info",
    ]


def test_workflow_executor_returns_final_response_without_tool_execution(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry)

    result = executor.execute_parsed_response(
        ParsedModelResponse(final_response="All set."),
        ToolContext(
            invocation_id="turn-1",
            workspace=str(tmp_path),
            logs=["turn-log"],
            artifacts=["turn-artifact"],
        ),
    )

    assert result.parsed_response.final_response == "All set."
    assert result.outcomes == []


def test_workflow_executor_executes_single_parsed_invocation(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry, allow_write=True)
    context = ToolContext(invocation_id="turn-2", workspace=str(tmp_path))

    write_result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {
                    "tool_name": "write_file",
                    "arguments": {
                        "path": "docs/note.txt",
                        "content": "hello",
                        "create_parents": True,
                    },
                }
            ]
        },
        context,
    )

    executed_results = _executed_tool_results(write_result)
    assert len(executed_results) == 1
    assert executed_results[0].ok is True
    record = executed_results[0].metadata["execution_record"]
    assert record["invocation_id"] == "turn-2"


def test_workflow_executor_executes_multiple_invocations_sequentially(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry, allow_write=True)

    setup_result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {
                    "tool_name": "write_file",
                    "arguments": {
                        "path": "docs/note.txt",
                        "content": "hello world",
                        "create_parents": True,
                    },
                }
            ]
        },
        ToolContext(invocation_id="setup", workspace=str(tmp_path)),
    )
    assert _executed_tool_results(setup_result)[0].ok is True

    result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {"tool_name": "read_file", "arguments": {"path": "docs/note.txt"}},
                {
                    "tool_name": "list_directory",
                    "arguments": {"path": ".", "recursive": True},
                },
            ]
        },
        ToolContext(
            invocation_id="turn-3",
            workspace=str(tmp_path),
            logs=["base-log"],
            artifacts=["base-artifact"],
        ),
    )

    executed_results = _executed_tool_results(result)
    assert [tool_result.tool_name for tool_result in executed_results] == [
        "read_file",
        "list_directory",
    ]
    first_record = executed_results[0].metadata["execution_record"]
    second_record = executed_results[1].metadata["execution_record"]
    assert first_record["invocation_id"] == "turn-3:1"
    assert second_record["invocation_id"] == "turn-3:2"
    assert executed_results[0].artifacts != []
    assert executed_results[1].artifacts == []
    assert "base-artifact" not in executed_results[0].artifacts
    assert "base-log" not in executed_results[0].logs
    assert "base-artifact" not in executed_results[1].artifacts
    assert "base-log" not in executed_results[1].logs


def test_workflow_executor_executes_action_envelope_path(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry, allow_write=True)

    setup = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {
                    "tool_name": "write_file",
                    "arguments": {
                        "path": "docs/note.txt",
                        "content": "hello openai",
                        "create_parents": True,
                    },
                }
            ]
        },
        ToolContext(invocation_id="setup-openai", workspace=str(tmp_path)),
    )
    assert _executed_tool_results(setup)[0].ok is True

    result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {"tool_name": "read_file", "arguments": {"path": "docs/note.txt"}}
            ]
        },
        ToolContext(invocation_id="turn-4", workspace=str(tmp_path)),
    )

    executed_results = _executed_tool_results(result)
    assert executed_results[0].ok is True
    assert executed_results[0].output["content"] == "hello openai"


def test_workflow_executor_executes_action_envelope_final_response_path(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry)

    result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {"actions": [], "final_response": "No tool needed."},
        ToolContext(invocation_id="turn-5", workspace=str(tmp_path)),
    )

    assert result.parsed_response.final_response == "No tool needed."
    assert result.outcomes == []


def test_workflow_executor_executes_action_envelope_json_string_paths(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry, allow_write=True)

    setup = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {
                    "tool_name": "write_file",
                    "arguments": {
                        "path": "docs/note.txt",
                        "content": "hello prompt",
                        "create_parents": True,
                    },
                }
            ]
        },
        ToolContext(invocation_id="setup-prompt", workspace=str(tmp_path)),
    )
    assert _executed_tool_results(setup)[0].ok is True

    action_result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        '{"actions":[{"tool_name":"read_file","arguments":{"path":"docs/note.txt"}}],"final_response":null}',
        ToolContext(invocation_id="turn-6", workspace=str(tmp_path)),
    )
    final_result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        '{"actions": [], "final_response": "Already answered."}',
        ToolContext(invocation_id="turn-7", workspace=str(tmp_path)),
    )

    assert _executed_tool_results(action_result)[0].output["content"] == "hello prompt"
    assert final_result.parsed_response.final_response == "Already answered."


def test_workflow_executor_normalizes_unknown_tool_failure(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry)

    result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {"actions": [{"tool_name": "missing_tool", "arguments": {}}]},
        ToolContext(invocation_id="turn-8", workspace=str(tmp_path)),
    )

    first = _executed_tool_results(result)[0]
    assert first.error is not None
    assert first.error.code is ErrorCode.TOOL_NOT_FOUND


def test_workflow_executor_normalizes_policy_denied_failure(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry)

    result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {
                    "tool_name": "write_file",
                    "arguments": {
                        "path": "docs/note.txt",
                        "content": "denied",
                        "create_parents": True,
                    },
                }
            ]
        },
        ToolContext(invocation_id="turn-9", workspace=str(tmp_path)),
    )

    first = _executed_tool_results(result)[0]
    assert first.error is not None
    assert first.error.code is ErrorCode.POLICY_DENIED


def test_workflow_executor_emits_approval_requested_and_pauses_at_first_pending(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = WorkflowExecutor(
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

    result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {"tool_name": "list_directory", "arguments": {"path": "."}},
                {
                    "tool_name": "write_file",
                    "arguments": {"path": "note.txt", "content": "after approval"},
                },
            ]
        },
        ToolContext(invocation_id="turn-approval-1", workspace=str(tmp_path)),
    )

    assert len(result.outcomes) == 1
    assert result.outcomes[0].status is WorkflowInvocationStatus.APPROVAL_REQUESTED
    assert result.outcomes[0].approval_request is not None
    assert result.outcomes[0].approval_request.invocation_index == 1
    assert len(executor.list_pending_approvals()) == 1


def test_workflow_executor_approve_resumes_and_continues_execution(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = WorkflowExecutor(
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

    initial = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {"tool_name": "list_directory", "arguments": {"path": "."}},
                {
                    "tool_name": "write_file",
                    "arguments": {"path": "resume.txt", "content": "approved"},
                },
            ]
        },
        ToolContext(invocation_id="turn-approval-2", workspace=str(tmp_path)),
    )
    approval_id = initial.outcomes[0].approval_request.approval_id  # type: ignore[union-attr]

    resumed = executor.resolve_pending_approval(approval_id, approved=True)
    statuses = [outcome.status for outcome in resumed.outcomes]

    assert statuses == [
        WorkflowInvocationStatus.EXECUTED,
        WorkflowInvocationStatus.EXECUTED,
    ]
    executed = _executed_tool_results(resumed)
    assert executed[0].tool_name == "list_directory"
    assert executed[1].tool_name == "write_file"
    assert executed[1].ok is True
    assert executor.list_pending_approvals() == []


def test_workflow_executor_deny_marks_outcome_and_continues(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = WorkflowExecutor(
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

    initial = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {"tool_name": "list_directory", "arguments": {"path": "."}},
                {
                    "tool_name": "write_file",
                    "arguments": {"path": "denied.txt", "content": "still runs"},
                },
            ]
        },
        ToolContext(invocation_id="turn-approval-3", workspace=str(tmp_path)),
    )
    approval_id = initial.outcomes[0].approval_request.approval_id  # type: ignore[union-attr]

    denied = executor.resolve_pending_approval(approval_id, approved=False)

    assert denied.outcomes[0].status is WorkflowInvocationStatus.APPROVAL_DENIED
    executed = _executed_tool_results(denied)
    assert len(executed) == 1
    assert executed[0].tool_name == "write_file"
    assert executed[0].ok is True


def test_workflow_executor_finalizes_expired_approvals(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = WorkflowExecutor(
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

    executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {"tool_name": "list_directory", "arguments": {"path": "."}},
                {
                    "tool_name": "write_file",
                    "arguments": {"path": "timeout.txt", "content": "after timeout"},
                },
            ]
        },
        ToolContext(invocation_id="turn-approval-4", workspace=str(tmp_path)),
    )
    finalized = executor.finalize_expired_approvals(
        now=datetime.now(UTC) + timedelta(seconds=5)
    )

    assert len(finalized) == 1
    result = finalized[0]
    assert result.outcomes[0].status is WorkflowInvocationStatus.APPROVAL_TIMED_OUT
    executed = _executed_tool_results(result)
    assert len(executed) == 1
    assert executed[0].tool_name == "write_file"
    assert executor.list_pending_approvals() == []


def test_workflow_executor_can_queue_multiple_pending_approvals_sequentially(
    tmp_path: str,
) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = WorkflowExecutor(
        registry,
        policy=ToolPolicy(
            allowed_side_effects={SideEffectClass.NONE, SideEffectClass.LOCAL_READ},
            require_approval_for={SideEffectClass.LOCAL_READ},
        ),
    )

    seed = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {"tool_name": "list_directory", "arguments": {"path": "."}},
                {"tool_name": "read_file", "arguments": {"path": "a.txt"}},
            ]
        },
        ToolContext(invocation_id="turn-approval-5", workspace=str(tmp_path)),
    )
    first_id = seed.outcomes[0].approval_request.approval_id  # type: ignore[union-attr]
    first_resume = executor.resolve_pending_approval(first_id, approved=True)

    assert first_resume.outcomes[0].status is WorkflowInvocationStatus.EXECUTED
    assert (
        first_resume.outcomes[1].status is WorkflowInvocationStatus.APPROVAL_REQUESTED
    )
    pending = executor.list_pending_approvals()
    assert len(pending) == 1
    assert pending[0].invocation_index == 2


def test_workflow_executor_normalizes_input_validation_failure(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry)

    result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {
                    "tool_name": "list_directory",
                    "arguments": {"recursive": {"value": True}},
                }
            ]
        },
        ToolContext(invocation_id="turn-10", workspace=str(tmp_path)),
    )

    first = _executed_tool_results(result)[0]
    assert first.error is not None
    assert first.error.code is ErrorCode.INPUT_VALIDATION_ERROR


def test_workflow_executor_propagates_adapter_parse_failure(tmp_path: str) -> None:
    registry = ToolRegistry()
    register_filesystem_tools(registry)
    executor = _executor(registry)

    with pytest.raises(ValueError):
        executor.execute_model_output(
            ActionEnvelopeAdapter(),
            "not json",
            ToolContext(invocation_id="turn-11", workspace=str(tmp_path)),
        )


def test_workflow_executor_executes_git_and_jira_paths(
    tmp_path: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeJira:
        def __init__(self, **kwargs: str) -> None:
            self.kwargs = kwargs

        def enhanced_jql(self, jql: str, *, limit: int) -> dict[str, object]:
            del jql, limit
            return {
                "issues": [
                    {
                        "key": "DEMO-1",
                        "fields": {
                            "summary": "Issue",
                            "status": {"name": "Open"},
                            "issuetype": {"name": "Task"},
                            "assignee": {"displayName": "Alice"},
                        },
                    }
                ]
            }

    def fake_run(
        args: list[str],
        *,
        cwd: str,
        capture_output: bool,
        text: bool,
        check: bool,
    ) -> subprocess.CompletedProcess[str]:
        del cwd, capture_output, text, check
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="ok\n")

    fake_module = ModuleType("atlassian")
    fake_module.Jira = FakeJira
    monkeypatch.setitem(sys.modules, "atlassian", fake_module)
    monkeypatch.setattr(git_tools.subprocess, "run", fake_run)

    registry = ToolRegistry()
    register_git_tools(registry)
    register_atlassian_tools(registry)
    executor = _executor(registry)

    git_result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {"actions": [{"tool_name": "run_git_status", "arguments": {"path": "."}}]},
        ToolContext(invocation_id="turn-12", workspace=str(tmp_path)),
    )
    jira_result = executor.execute_model_output(
        ActionEnvelopeAdapter(),
        {
            "actions": [
                {"tool_name": "search_jira", "arguments": {"jql": "project = DEMO"}}
            ]
        },
        ToolContext(
            invocation_id="turn-13",
            env={
                "JIRA_BASE_URL": "https://example.atlassian.net",
                "JIRA_USERNAME": "user@example.com",
                "JIRA_API_TOKEN": "token",
            },
        ),
    )

    assert _executed_tool_results(git_result)[0].output["status_text"] == "ok\n"
    assert _executed_tool_results(jira_result)[0].output["issues"][0]["key"] == "DEMO-1"


def test_workflow_executor_async_executes_sync_tooling_paths(tmp_path: str) -> None:
    async def run() -> None:
        registry = ToolRegistry()
        register_filesystem_tools(registry)
        executor = _executor(registry, allow_write=True)

        setup = await executor.execute_model_output_async(
            ActionEnvelopeAdapter(),
            {
                "actions": [
                    {
                        "tool_name": "write_file",
                        "arguments": {
                            "path": "docs/note.txt",
                            "content": "async hello",
                            "create_parents": True,
                        },
                    }
                ]
            },
            ToolContext(invocation_id="async-setup", workspace=str(tmp_path)),
        )
        result = await executor.execute_model_output_async(
            ActionEnvelopeAdapter(),
            {
                "actions": [
                    {"tool_name": "read_file", "arguments": {"path": "docs/note.txt"}}
                ]
            },
            ToolContext(invocation_id="async-turn", workspace=str(tmp_path)),
        )

        assert _executed_tool_results(setup)[0].ok is True
        executed = _executed_tool_results(result)[0]
        assert executed.ok is True
        assert executed.output["content"] == "async hello"

    asyncio.run(run())


def test_workflow_executor_async_handles_async_only_tools() -> None:
    async def run() -> None:
        registry = ToolRegistry()
        registry.register(_AsyncEchoTool())
        executor = _executor(registry)

        result = await executor.execute_parsed_response_async(
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "async_echo", "arguments": {"value": "hello"}}
                ]
            ),
            ToolContext(invocation_id="async-only"),
        )

        executed = _executed_tool_results(result)[0]
        assert executed.ok is True
        assert executed.output["value"] == "async-only:hello"
        assert executed.logs == ["async-echo"]

    asyncio.run(run())


def test_workflow_executor_async_approval_roundtrip(tmp_path: str) -> None:
    async def run() -> None:
        registry = ToolRegistry()
        register_filesystem_tools(registry)
        executor = WorkflowExecutor(
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

        initial = await executor.execute_model_output_async(
            ActionEnvelopeAdapter(),
            {
                "actions": [
                    {"tool_name": "list_directory", "arguments": {"path": "."}},
                    {
                        "tool_name": "write_file",
                        "arguments": {"path": "after.txt", "content": "approved"},
                    },
                ]
            },
            ToolContext(invocation_id="async-approval", workspace=str(tmp_path)),
        )

        assert initial.outcomes[0].status is WorkflowInvocationStatus.APPROVAL_REQUESTED
        approval_id = initial.outcomes[0].approval_request.approval_id  # type: ignore[union-attr]
        resumed = await executor.resolve_pending_approval_async(
            approval_id, approved=True
        )

        statuses = [outcome.status for outcome in resumed.outcomes]
        assert statuses == [
            WorkflowInvocationStatus.EXECUTED,
            WorkflowInvocationStatus.EXECUTED,
        ]

    asyncio.run(run())
