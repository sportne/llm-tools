"""Textual UI tests for the workbench shell."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("textual")

from tests.apps._imports import import_textual_workbench_modules
from textual.widgets import Button, Input, Static

from llm_tools.apps.textual_workbench.models import (
    ApprovalFinalizeResult,
    ApprovalResolutionResult,
    DirectExecutionResult,
    ExportToolsResult,
    ModelTurnExecutionResult,
)
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolResult
from llm_tools.workflow_api import WorkflowTurnResult
from llm_tools.workflow_api.models import (
    ApprovalRequest,
    WorkflowInvocationOutcome,
    WorkflowInvocationStatus,
)

_WORKBENCH_MODULES = import_textual_workbench_modules()
JsonArgumentsTextArea = _WORKBENCH_MODULES.app.JsonArgumentsTextArea
PromptComposerTextArea = _WORKBENCH_MODULES.app.PromptComposerTextArea
TextualWorkbenchApp = _WORKBENCH_MODULES.app.TextualWorkbenchApp


def _approval_request(index: int = 1) -> ApprovalRequest:
    return ApprovalRequest(
        approval_id=f"approval-{index}",
        invocation_index=index,
        request={"tool_name": "list_directory", "arguments": {"path": "."}},
        tool_name="list_directory",
        tool_version="0.1.0",
        policy_reason="approval required",
        policy_metadata={"tool_name": "list_directory"},
        requested_at="2026-01-01T00:00:00Z",
        expires_at="2026-01-01T00:05:00Z",
    )


async def _run_default_controller_assertions(tmp_path: Path) -> None:
    app = TextualWorkbenchApp()
    app._config_state = app._config_state.model_copy(
        update={"workspace": str(tmp_path)}
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#provider-preset-value", Static).renderable == "ollama"
        assert app.query_one("#mode-value", Static).renderable == "auto"
        tools_text = str(app.query_one("#registered-tools-box", Static).renderable)
        assert "read_file" in tools_text
        assert "run_git_status" in tools_text
        assert "search_text" in tools_text
        assert "search_jira" not in tools_text
        assert (
            app.query_one("#toggle-filesystem-tools", Button).label.plain
            == "Filesystem tools: ON"
        )
        assert (
            app.query_one("#toggle-atlassian-tools", Button).label.plain
            == "Atlassian tools: OFF"
        )

        app._run_export_tools_worker = lambda: None  # type: ignore[method-assign]
        app._start_export_tools()
        assert app.query_one("#export-tools-button", Button).disabled is True
        assert app.query_one("#run-model-turn-button", Button).disabled is True
        assert app.query_one("#execute-direct-tool-button", Button).disabled is True
        app._handle_export_tools_success(
            ExportToolsResult(exported_tools={"demo": True})
        )
        assert app.query_one("#export-tools-button", Button).disabled is False

        app._config_state = app._config_state.model_copy(
            update={"allow_local_write": True}
        )
        app._sync_ui_from_state()
        direct_result = app._controller.execute_direct_tool(
            app._config_state,
            tool_name="write_file",
            arguments_text='{"path":"workbench.txt","content":"hello from workbench"}',
        )
        app._handle_direct_tool_success(direct_result)
        direct_result_text = str(app.query_one("#direct-result-box", Static).renderable)
        execution_record = str(
            app.query_one("#execution-record-box", Static).renderable
        )
        transcript = str(app.query_one("#transcript-body", Static).renderable)
        assert "write_file" in direct_result_text
        assert "execution_record" not in direct_result_text
        assert "write_file" in execution_record
        assert "Directly executed tool 'write_file'." in transcript

        parsed_final = ParsedModelResponse(final_response="done")
        app._handle_model_turn_success(
            ModelTurnExecutionResult(
                exported_tools={"demo": True},
                parsed_response=parsed_final,
                workflow_result=WorkflowTurnResult(parsed_response=parsed_final),
            )
        )
        parsed_response_text = str(
            app.query_one("#parsed-response-box", Static).renderable
        )
        workflow_result_text = str(
            app.query_one("#workflow-result-box", Static).renderable
        )
        assert "final_response" in parsed_response_text
        assert "done" in parsed_response_text
        assert "final_response" in workflow_result_text

        parsed_invocations = ParsedModelResponse(
            invocations=[{"tool_name": "read_file", "arguments": {"path": "demo.txt"}}]
        )
        app._handle_model_turn_success(
            ModelTurnExecutionResult(
                exported_tools={"demo": True},
                parsed_response=parsed_invocations,
                workflow_result=WorkflowTurnResult(
                    parsed_response=parsed_invocations,
                    outcomes=[
                        WorkflowInvocationOutcome(
                            invocation_index=1,
                            request={
                                "tool_name": "read_file",
                                "arguments": {"path": "demo.txt"},
                            },
                            status=WorkflowInvocationStatus.EXECUTED,
                            tool_result=ToolResult(
                                ok=True,
                                tool_name="read_file",
                                tool_version="0.1.0",
                                output={"content": "demo"},
                            ),
                        )
                    ],
                ),
            )
        )

        app._config_state = app._config_state.model_copy(
            update={"require_approval_for_local_read": True}
        )
        app._sync_ui_from_state()
        pending = app._controller.execute_direct_tool(
            app._config_state,
            tool_name="list_directory",
            arguments_text='{"path":"."}',
        )
        app._handle_direct_tool_success(pending)
        queue_text = str(app.query_one("#approval-queue-box", Static).renderable)
        approval_id = app.query_one("#approval-id-input", Input).value.strip()
        assert "approval_id" in queue_text
        assert approval_id != ""

        resolved = app._controller.resolve_pending_approval(
            app._config_state,
            approval_id=approval_id,
            approved=True,
        )
        app._handle_resolve_approval_success(resolved, approved=True)
        assert str(app.query_one("#approval-queue-box", Static).renderable) == "[]"

        app._handle_worker_error("boom")
        app._handle_finalize_expired_success(
            ApprovalFinalizeResult(workflow_results=[])
        )
        transcript = str(app.query_one("#transcript-body", Static).renderable)
        assert "Model returned a final response without tool execution." in transcript
        assert (
            "Model returned tool invocations and the workflow executed them."
            in transcript
        )
        assert "requires approval and is queued" in transcript
        assert "Approved pending approval request." in transcript
        assert "Error: boom" in transcript
        assert "Finalized 0 expired approval request(s)." in transcript

        app._handle_export_tools_success(
            ExportToolsResult(session_rebuilt=True, exported_tools={"demo": True})
        )
        app._handle_model_turn_success(
            ModelTurnExecutionResult(
                session_rebuilt=True,
                exported_tools={"demo": True},
                parsed_response=parsed_invocations,
                workflow_result=None,
            )
        )
        app._handle_model_turn_success(
            ModelTurnExecutionResult(
                exported_tools={"demo": True},
                parsed_response=parsed_invocations,
                workflow_result=WorkflowTurnResult(
                    parsed_response=parsed_invocations,
                    outcomes=[
                        WorkflowInvocationOutcome(
                            invocation_index=1,
                            request={
                                "tool_name": "list_directory",
                                "arguments": {"path": "."},
                            },
                            status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                            approval_request=_approval_request(),
                        )
                    ],
                ),
            )
        )
        app._handle_direct_tool_success(
            DirectExecutionResult(
                session_rebuilt=True,
                workflow_result=WorkflowTurnResult(
                    parsed_response=parsed_invocations,
                    outcomes=[
                        WorkflowInvocationOutcome(
                            invocation_index=1,
                            request={
                                "tool_name": "list_directory",
                                "arguments": {"path": "."},
                            },
                            status=WorkflowInvocationStatus.APPROVAL_DENIED,
                            approval_request=_approval_request(),
                        )
                    ],
                ),
                tool_result=None,
            )
        )
        app._handle_resolve_approval_success(
            ApprovalResolutionResult(
                session_rebuilt=True,
                workflow_result=WorkflowTurnResult(
                    parsed_response=ParsedModelResponse(final_response="done")
                ),
            ),
            approved=True,
        )
        app._run_state = app._run_state.model_copy(
            update={
                "last_workflow_result": WorkflowTurnResult(
                    parsed_response=ParsedModelResponse(final_response="carry-over")
                )
            }
        )
        app._handle_finalize_expired_success(
            ApprovalFinalizeResult(session_rebuilt=True, workflow_results=[])
        )
        app._handle_finalize_expired_success(
            ApprovalFinalizeResult(
                workflow_results=[
                    WorkflowTurnResult(
                        parsed_response=parsed_invocations,
                        outcomes=[
                            WorkflowInvocationOutcome(
                                invocation_index=1,
                                request={
                                    "tool_name": "list_directory",
                                    "arguments": {"path": "."},
                                },
                                status=WorkflowInvocationStatus.APPROVAL_TIMED_OUT,
                                approval_request=_approval_request(),
                            )
                        ],
                    )
                ]
            )
        )
        transcript = str(app.query_one("#transcript-body", Static).renderable)
        assert "workflow session was rebuilt" in transcript
        assert "execution-after-parse is disabled" in transcript
        assert "at least one invocation requires approval" in transcript
        assert "Direct tool invocation completed without execution." in transcript
        assert "Finalized 1 expired approval request(s)." in transcript

        latest = TextualWorkbenchApp._latest_tool_result(
            WorkflowTurnResult(
                parsed_response=ParsedModelResponse(
                    invocations=[
                        {"tool_name": "list_directory", "arguments": {"path": "."}},
                        {"tool_name": "read_file", "arguments": {"path": "README.md"}},
                    ]
                ),
                outcomes=[
                    WorkflowInvocationOutcome(
                        invocation_index=1,
                        request={
                            "tool_name": "list_directory",
                            "arguments": {"path": "."},
                        },
                        status=WorkflowInvocationStatus.APPROVAL_REQUESTED,
                        approval_request=_approval_request(),
                    ),
                    WorkflowInvocationOutcome(
                        invocation_index=2,
                        request={
                            "tool_name": "read_file",
                            "arguments": {"path": "README.md"},
                        },
                        status=WorkflowInvocationStatus.EXECUTED,
                        tool_result=ToolResult(
                            ok=True,
                            tool_name="read_file",
                            tool_version="0.1.0",
                            output={"content": "ok"},
                        ),
                    ),
                ],
            )
        )
        assert latest.tool_name == "read_file"
        app.exit()
        await pilot.pause()


def test_textual_workbench_default_controller_paths(tmp_path: Path) -> None:
    asyncio.run(_run_default_controller_assertions(tmp_path))


async def _run_dispatch_and_start_method_assertions() -> None:
    app = TextualWorkbenchApp()
    calls: list[tuple[object, ...]] = []

    async with app.run_test() as pilot:
        await pilot.pause()
        app.handle_tool_name_change()
        app.handle_button_press(SimpleNamespace(button=SimpleNamespace(id=None)))
        app.handle_button_press(
            SimpleNamespace(button=SimpleNamespace(id="provider-preset-button"))
        )
        app.handle_button_press(
            SimpleNamespace(button=SimpleNamespace(id="mode-button"))
        )
        app.handle_button_press(
            SimpleNamespace(button=SimpleNamespace(id="toggle-atlassian-tools"))
        )
        assert "search_jira" in str(
            app.query_one("#registered-tools-box", Static).renderable
        )

        app.query_one("#prompt-input", PromptComposerTextArea).text = "Prompt"
        app.query_one("#tool-name-input", Input).value = "read_file"
        app.query_one(
            "#tool-args-input", JsonArgumentsTextArea
        ).text = '{"path":"README.md"}'
        app.query_one("#approval-id-input", Input).value = "approval-1"
        app._run_state = app._run_state.model_copy(
            update={"pending_approvals": [_approval_request()]}
        )

        app._read_inputs_into_config = lambda: calls.append(("read",))  # type: ignore[method-assign]
        app._start_export_tools = lambda: calls.append(("export",))  # type: ignore[method-assign]
        app._run_model_turn_worker = (  # type: ignore[method-assign]
            lambda prompt: calls.append(("run", prompt))
        )
        app._run_direct_tool_worker = (  # type: ignore[method-assign]
            lambda tool_name, arguments_text: calls.append(
                ("direct", tool_name, arguments_text)
            )
        )
        app._run_resolve_approval_worker = (  # type: ignore[method-assign]
            lambda approval_id, approved: calls.append(
                ("resolve", approval_id, approved)
            )
        )
        app._run_finalize_expired_worker = lambda: calls.append(  # type: ignore[method-assign]
            ("finalize",)
        )

        app.handle_button_press(
            SimpleNamespace(button=SimpleNamespace(id="export-tools-button"))
        )
        app.handle_button_press(
            SimpleNamespace(button=SimpleNamespace(id="run-model-turn-button"))
        )
        app._run_state = app._run_state.model_copy(update={"busy": False})
        app.handle_button_press(
            SimpleNamespace(button=SimpleNamespace(id="execute-direct-tool-button"))
        )
        app._run_state = app._run_state.model_copy(update={"busy": False})
        app.handle_button_press(
            SimpleNamespace(button=SimpleNamespace(id="approve-approval-button"))
        )
        app._run_state = app._run_state.model_copy(update={"busy": False})
        app.handle_button_press(
            SimpleNamespace(button=SimpleNamespace(id="deny-approval-button"))
        )
        app._run_state = app._run_state.model_copy(update={"busy": False})
        app.handle_button_press(
            SimpleNamespace(
                button=SimpleNamespace(id="finalize-expired-approvals-button")
            )
        )
        app._run_state = app._run_state.model_copy(update={"busy": False})
        app._start_model_turn()
        app._run_state = app._run_state.model_copy(update={"busy": False})
        app._start_direct_tool_execution()
        app._run_state = app._run_state.model_copy(update={"busy": False})
        app._start_resolve_pending_approval(approved=True)
        app._run_state = app._run_state.model_copy(update={"busy": False})
        app._start_finalize_expired_approvals()
        app.handle_approval_id_change()

        assert ("export",) in calls
        assert ("run", "Prompt") in calls
        assert ("direct", "read_file", '{"path":"README.md"}') in calls
        assert ("resolve", "approval-1", True) in calls
        assert ("resolve", "approval-1", False) in calls
        assert ("finalize",) in calls
        assert app.query_one("#approval-id-input", Input).value == "approval-1"
        assert "approval-1" in str(
            app.query_one("#approval-detail-box", Static).renderable
        )

        app.query_one("#approval-id-input", Input).value = ""
        app._run_state = app._run_state.model_copy(update={"busy": False})
        app._start_resolve_pending_approval(approved=True)
        assert "Error: An approval id is required." in str(
            app.query_one("#transcript-body", Static).renderable
        )

        app._run_state = app._run_state.model_copy(update={"busy": True})
        app.action_export_tools()
        app.action_run_model_turn()
        app.action_execute_direct_tool()
        app.action_approve_pending()
        app.action_deny_pending()
        app.action_finalize_expired()
        app._start_model_turn()
        app._start_direct_tool_execution()
        app._start_resolve_pending_approval(approved=True)
        app._start_finalize_expired_approvals()
        assert app._run_state.busy is True
        app.exit()
        await pilot.pause()


def test_textual_workbench_dispatch_and_start_methods() -> None:
    asyncio.run(_run_dispatch_and_start_method_assertions())


def test_textual_workbench_worker_wrappers_and_entrypoints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    parsed = ParsedModelResponse(final_response="done")
    success_app = TextualWorkbenchApp()
    threaded_calls: list[tuple[str, object]] = []
    direct_calls: list[tuple[str, object]] = []

    monkeypatch.setattr(
        success_app,
        "call_from_thread",
        lambda func, payload: threaded_calls.append((func.__name__, payload)),
    )
    monkeypatch.setattr(
        success_app,
        "_handle_model_turn_success",
        lambda result: direct_calls.append(("model", result)),
    )
    monkeypatch.setattr(
        success_app,
        "_handle_direct_tool_success",
        lambda result: direct_calls.append(("direct", result)),
    )
    monkeypatch.setattr(
        success_app,
        "_handle_resolve_approval_success",
        lambda result, approved: direct_calls.append(("resolve", (result, approved))),
    )
    monkeypatch.setattr(
        success_app,
        "_handle_finalize_expired_success",
        lambda result: direct_calls.append(("finalize", result)),
    )
    monkeypatch.setattr(
        success_app._controller,
        "export_tools",
        lambda _config: ExportToolsResult(exported_tools={"demo": True}),
    )

    async def _model_success(*_args: object, **_kwargs: object) -> object:
        return ModelTurnExecutionResult(
            exported_tools={"demo": True},
            parsed_response=parsed,
            workflow_result=WorkflowTurnResult(parsed_response=parsed),
        )

    async def _direct_success(*_args: object, **_kwargs: object) -> object:
        return DirectExecutionResult(
            workflow_result=WorkflowTurnResult(parsed_response=parsed),
            tool_result=ToolResult(
                ok=True,
                tool_name="read_file",
                tool_version="0.1.0",
                output={"content": "ok"},
            ),
        )

    async def _resolve_success(*_args: object, **_kwargs: object) -> object:
        return ApprovalResolutionResult(
            workflow_result=WorkflowTurnResult(parsed_response=parsed)
        )

    async def _finalize_success(*_args: object, **_kwargs: object) -> object:
        return ApprovalFinalizeResult(workflow_results=[])

    monkeypatch.setattr(success_app._controller, "run_model_turn_async", _model_success)
    monkeypatch.setattr(
        success_app._controller, "execute_direct_tool_async", _direct_success
    )
    monkeypatch.setattr(
        success_app._controller,
        "resolve_pending_approval_async",
        _resolve_success,
    )
    monkeypatch.setattr(
        success_app._controller,
        "finalize_expired_approvals_async",
        _finalize_success,
    )

    success_app._run_export_tools_worker.__wrapped__(success_app)

    async def _exercise_success_workers() -> None:
        await success_app._run_model_turn_worker.__wrapped__(success_app, "Prompt")
        await success_app._run_direct_tool_worker.__wrapped__(
            success_app, "read_file", '{"path":"README.md"}'
        )
        await success_app._run_resolve_approval_worker.__wrapped__(
            success_app, "approval-1", True
        )
        await success_app._run_finalize_expired_worker.__wrapped__(success_app)

    asyncio.run(_exercise_success_workers())

    assert threaded_calls[0][0] == "_handle_export_tools_success"
    assert direct_calls[0][0] == "model"
    assert direct_calls[1][0] == "direct"
    assert direct_calls[2][0] == "resolve"
    assert direct_calls[3][0] == "finalize"

    error_app = TextualWorkbenchApp()
    error_calls: list[tuple[str, object]] = []
    monkeypatch.setattr(
        error_app,
        "call_from_thread",
        lambda func, payload: error_calls.append((func.__name__, payload)),
    )
    monkeypatch.setattr(
        error_app,
        "_handle_worker_error",
        lambda message: error_calls.append(("error", message)),
    )

    def _raise_export(_config: object) -> object:
        raise ValueError("export boom")

    async def _raise_model(*_args: object, **_kwargs: object) -> object:
        raise ValueError("model boom")

    async def _raise_direct(*_args: object, **_kwargs: object) -> object:
        raise ValueError("direct boom")

    async def _raise_resolve(*_args: object, **_kwargs: object) -> object:
        raise ValueError("resolve boom")

    async def _raise_finalize(*_args: object, **_kwargs: object) -> object:
        raise ValueError("finalize boom")

    monkeypatch.setattr(error_app._controller, "export_tools", _raise_export)
    monkeypatch.setattr(error_app._controller, "run_model_turn_async", _raise_model)
    monkeypatch.setattr(
        error_app._controller, "execute_direct_tool_async", _raise_direct
    )
    monkeypatch.setattr(
        error_app._controller, "resolve_pending_approval_async", _raise_resolve
    )
    monkeypatch.setattr(
        error_app._controller, "finalize_expired_approvals_async", _raise_finalize
    )

    error_app._run_export_tools_worker.__wrapped__(error_app)

    async def _exercise_error_workers() -> None:
        await error_app._run_model_turn_worker.__wrapped__(error_app, "Prompt")
        await error_app._run_direct_tool_worker.__wrapped__(
            error_app, "read_file", '{"path":"README.md"}'
        )
        await error_app._run_resolve_approval_worker.__wrapped__(
            error_app, "approval-1", False
        )
        await error_app._run_finalize_expired_worker.__wrapped__(error_app)

    asyncio.run(_exercise_error_workers())

    assert any(payload == "export boom" for _, payload in error_calls)
    assert ("error", "model boom") in error_calls
    assert ("error", "direct boom") in error_calls
    assert ("error", "resolve boom") in error_calls
    assert ("error", "finalize boom") in error_calls

    run_calls: list[str] = []
    monkeypatch.setattr(
        _WORKBENCH_MODULES.app.TextualWorkbenchApp,
        "run",
        lambda self: run_calls.append("run"),
    )
    _WORKBENCH_MODULES.app.run_workbench_app()
    monkeypatch.setattr(
        _WORKBENCH_MODULES.app,
        "run_workbench_app",
        lambda: run_calls.append("main"),
    )
    assert _WORKBENCH_MODULES.app.main() == 0
    assert run_calls == ["run", "main"]
