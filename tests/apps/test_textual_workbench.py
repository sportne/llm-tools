"""Textual UI tests for the workbench shell."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("textual")

from textual.widgets import Button, Input, Static

from llm_tools.apps.textual_workbench.app import TextualWorkbenchApp
from llm_tools.apps.textual_workbench.controller import WorkbenchController
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


class _SlowExportController(WorkbenchController):
    def export_tools(self, config):  # type: ignore[override]
        del config
        time.sleep(0.1)
        return ExportToolsResult(exported_tools=[{"demo": True}])


class _FakeModelTurnController(WorkbenchController):
    async def run_model_turn_async(  # type: ignore[override]
        self, config, *, prompt
    ):
        del config, prompt
        parsed = ParsedModelResponse(final_response="done")
        return ModelTurnExecutionResult(
            exported_tools={"demo": True},
            parsed_response=parsed,
            workflow_result=WorkflowTurnResult(parsed_response=parsed),
        )


class _FakeToolCallController(WorkbenchController):
    async def run_model_turn_async(  # type: ignore[override]
        self, config, *, prompt
    ):
        del config, prompt
        parsed = ParsedModelResponse(
            invocations=[{"tool_name": "read_file", "arguments": {"path": "demo.txt"}}]
        )
        return ModelTurnExecutionResult(
            exported_tools={"demo": True},
            parsed_response=parsed,
            workflow_result=WorkflowTurnResult(
                parsed_response=parsed,
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


class _ErrorController(WorkbenchController):
    def export_tools(self, config):  # type: ignore[override]
        del config
        raise ValueError("boom")


class _WorkerErrorController(WorkbenchController):
    async def run_model_turn_async(  # type: ignore[override]
        self, config, *, prompt
    ):
        del config, prompt
        raise ValueError("model worker boom")

    async def execute_direct_tool_async(  # type: ignore[override]
        self, config, *, tool_name, arguments_text
    ):
        del config, tool_name, arguments_text
        raise ValueError("direct worker boom")

    async def resolve_pending_approval_async(  # type: ignore[override]
        self, config, *, approval_id, approved
    ):
        del config, approval_id, approved
        raise ValueError("resolve worker boom")

    async def finalize_expired_approvals_async(self, config):  # type: ignore[override]
        del config
        raise ValueError("finalize worker boom")


class _FinalizeSuccessController(WorkbenchController):
    async def finalize_expired_approvals_async(self, config):  # type: ignore[override]
        del config
        return ApprovalFinalizeResult(workflow_results=[])


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


async def _run_startup_assertions(tmp_path: Path) -> None:
    app = TextualWorkbenchApp()
    app._config_state = app._config_state.model_copy(
        update={"workspace": str(tmp_path)}
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        assert app.query_one("#provider-preset-value", Static).renderable == "ollama"
        assert app.query_one("#mode-value", Static).renderable == "native_tool_calling"
        tools_text = str(app.query_one("#registered-tools-box", Static).renderable)
        assert "read_file" in tools_text
        assert "run_git_status" in tools_text
        assert "directory_text_search" in tools_text
        assert "search_jira" not in tools_text
        assert (
            app.query_one("#toggle-filesystem-tools", Button).label.plain
            == "Filesystem tools: ON"
        )
        assert (
            app.query_one("#toggle-atlassian-tools", Button).label.plain
            == "Atlassian tools: OFF"
        )
        app.handle_tool_name_change()
        app.exit()
        await pilot.pause()


def test_textual_workbench_launches_with_default_shell_state(tmp_path: Path) -> None:
    asyncio.run(_run_startup_assertions(tmp_path))


async def _run_busy_state_assertions() -> None:
    app = TextualWorkbenchApp(controller=_SlowExportController())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_export_tools()
        await pilot.pause(0.02)
        assert app.query_one("#export-tools-button", Button).disabled is True
        assert app.query_one("#run-model-turn-button", Button).disabled is True
        assert app.query_one("#execute-direct-tool-button", Button).disabled is True
        await pilot.pause(0.15)
        assert app.query_one("#exported-tools-box", Static).renderable != ""
        app.exit()
        await pilot.pause()


def test_textual_workbench_busy_state_disables_actions() -> None:
    asyncio.run(_run_busy_state_assertions())


async def _run_direct_execution_assertions(tmp_path: Path) -> None:
    app = TextualWorkbenchApp()
    app._config_state = app._config_state.model_copy(
        update={"workspace": str(tmp_path), "allow_local_write": True}
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app._sync_ui_from_state()
        app.query_one("#tool-name-input", Input).value = "write_file"
        app.query_one("#tool-args-input").load_text(
            '{"path":"workbench.txt","content":"hello from workbench"}'
        )
        app.action_execute_direct_tool()
        await pilot.pause(0.1)
        direct_result = str(app.query_one("#direct-result-box", Static).renderable)
        execution_record = str(
            app.query_one("#execution-record-box", Static).renderable
        )
        transcript = str(app.query_one("#transcript-body", Static).renderable)
        assert "write_file" in direct_result
        assert "execution_record" not in direct_result
        assert "write_file" in execution_record
        assert "Directly executed tool 'write_file'." in transcript
        app.exit()
        await pilot.pause()


def test_textual_workbench_updates_result_panes_after_direct_execution(
    tmp_path: Path,
) -> None:
    asyncio.run(_run_direct_execution_assertions(tmp_path))


async def _run_provider_assertions() -> None:
    app = TextualWorkbenchApp(controller=_FakeModelTurnController())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt-input").load_text("Say hi")
        app.action_run_model_turn()
        await pilot.pause(0.1)
        parsed_response = str(app.query_one("#parsed-response-box", Static).renderable)
        workflow_result = str(app.query_one("#workflow-result-box", Static).renderable)
        transcript = str(app.query_one("#transcript-body", Static).renderable)
        assert "final_response" in parsed_response
        assert "done" in parsed_response
        assert "final_response" in workflow_result
        assert "Model returned a final response without tool execution." in transcript
        app.exit()
        await pilot.pause()


def test_textual_workbench_updates_result_panes_after_provider_execution() -> None:
    asyncio.run(_run_provider_assertions())


async def _run_button_and_error_assertions() -> None:
    app = TextualWorkbenchApp(controller=_ErrorController())
    async with app.run_test() as pilot:
        await pilot.pause()
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

        app.handle_button_press(
            SimpleNamespace(button=app.query_one("#export-tools-button", Button))
        )
        await pilot.pause(0.05)
        transcript = str(app.query_one("#transcript-body", Static).renderable)
        assert "Error: boom" in transcript
        app.exit()
        await pilot.pause()


def test_textual_workbench_button_handlers_and_error_path() -> None:
    asyncio.run(_run_button_and_error_assertions())


async def _run_tool_invocation_branch_assertions() -> None:
    app = TextualWorkbenchApp(controller=_FakeToolCallController())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt-input").load_text("Use a tool")
        app.action_run_model_turn()
        await pilot.pause(0.1)
        transcript = str(app.query_one("#transcript-body", Static).renderable)
        assert (
            "Model returned tool invocations and the workflow executed them."
            in transcript
        )
        app.exit()
        await pilot.pause()


def test_textual_workbench_covers_tool_invocation_result_branch() -> None:
    asyncio.run(_run_tool_invocation_branch_assertions())


async def _run_approval_queue_assertions(tmp_path: Path) -> None:
    app = TextualWorkbenchApp()
    app._config_state = app._config_state.model_copy(
        update={
            "workspace": str(tmp_path),
            "require_approval_for_local_read": True,
        }
    )
    async with app.run_test() as pilot:
        await pilot.pause()
        app._sync_ui_from_state()
        app.query_one("#tool-name-input", Input).value = "list_directory"
        app.query_one("#tool-args-input").load_text('{"path":"."}')
        app.action_execute_direct_tool()
        await pilot.pause(0.1)

        queue_text = str(app.query_one("#approval-queue-box", Static).renderable)
        transcript = str(app.query_one("#transcript-body", Static).renderable)
        approval_id = app.query_one("#approval-id-input", Input).value.strip()
        assert "approval_id" in queue_text
        assert "requires approval and is queued" in transcript
        assert approval_id != ""

        app.action_approve_pending()
        await pilot.pause(0.1)
        transcript = str(app.query_one("#transcript-body", Static).renderable)
        queue_text = str(app.query_one("#approval-queue-box", Static).renderable)
        assert "Approved pending approval request." in transcript
        assert queue_text == "[]"
        app.exit()
        await pilot.pause()


def test_textual_workbench_approval_queue_round_trip(tmp_path: Path) -> None:
    asyncio.run(_run_approval_queue_assertions(tmp_path))


async def _run_dispatch_and_busy_guard_assertions() -> None:
    app = TextualWorkbenchApp()
    called: list[str] = []

    app._start_model_turn = lambda: called.append("run")  # type: ignore[method-assign]
    app._start_direct_tool_execution = lambda: called.append("direct")  # type: ignore[method-assign]
    app._start_resolve_pending_approval = (  # type: ignore[method-assign]
        lambda *, approved: called.append("approve" if approved else "deny")
    )
    app._start_finalize_expired_approvals = (  # type: ignore[method-assign]
        lambda: called.append("finalize")
    )

    async with app.run_test() as pilot:
        await pilot.pause()
        app.handle_button_press(
            SimpleNamespace(button=SimpleNamespace(id="run-model-turn-button"))
        )
        app.handle_button_press(
            SimpleNamespace(button=SimpleNamespace(id="execute-direct-tool-button"))
        )
        app.handle_button_press(
            SimpleNamespace(button=SimpleNamespace(id="approve-approval-button"))
        )
        app.handle_button_press(
            SimpleNamespace(button=SimpleNamespace(id="deny-approval-button"))
        )
        app.handle_button_press(
            SimpleNamespace(
                button=SimpleNamespace(id="finalize-expired-approvals-button")
            )
        )
        assert called == ["run", "direct", "approve", "deny", "finalize"]

        app._run_state = app._run_state.model_copy(update={"busy": True})
        app.action_export_tools()
        app.action_run_model_turn()
        app.action_execute_direct_tool()
        app.action_approve_pending()
        app.action_deny_pending()
        app.action_finalize_expired()
        assert app._run_state.busy is True
        app.exit()
        await pilot.pause()


def test_textual_workbench_dispatch_and_busy_guards() -> None:
    asyncio.run(_run_dispatch_and_busy_guard_assertions())


async def _run_start_methods_busy_shortcircuit_assertions() -> None:
    app = TextualWorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()
        app._run_state = app._run_state.model_copy(update={"busy": True})
        app._start_export_tools()
        app._start_model_turn()
        app._start_direct_tool_execution()
        app._start_resolve_pending_approval(approved=True)
        app._start_finalize_expired_approvals()
        assert app._run_state.busy is True
        app.exit()
        await pilot.pause()


def test_textual_workbench_start_methods_short_circuit_when_busy() -> None:
    asyncio.run(_run_start_methods_busy_shortcircuit_assertions())


async def _run_worker_error_path_assertions() -> None:
    app = TextualWorkbenchApp(controller=_WorkerErrorController())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.query_one("#prompt-input").load_text("hi")
        app.action_run_model_turn()
        await pilot.pause(0.08)

        app.query_one("#tool-name-input", Input).value = "list_directory"
        app.query_one("#tool-args-input").load_text('{"path":"."}')
        app.action_execute_direct_tool()
        await pilot.pause(0.08)

        app.query_one("#approval-id-input", Input).value = ""
        app.action_deny_pending()
        await pilot.pause(0.02)

        app.query_one("#approval-id-input", Input).value = "approval-1"
        app.action_approve_pending()
        await pilot.pause(0.08)

        app.action_finalize_expired()
        await pilot.pause(0.08)

        transcript = str(app.query_one("#transcript-body", Static).renderable)
        assert "Error: model worker boom" in transcript
        assert "Error: direct worker boom" in transcript
        assert "Error: An approval id is required." in transcript
        assert "Error: resolve worker boom" in transcript
        assert "Error: finalize worker boom" in transcript
        app.exit()
        await pilot.pause()


def test_textual_workbench_worker_error_paths() -> None:
    asyncio.run(_run_worker_error_path_assertions())


async def _run_finalize_success_worker_assertions() -> None:
    app = TextualWorkbenchApp(controller=_FinalizeSuccessController())
    async with app.run_test() as pilot:
        await pilot.pause()
        app.action_finalize_expired()
        await pilot.pause(0.08)
        transcript = str(app.query_one("#transcript-body", Static).renderable)
        assert "Finalized 0 expired approval request(s)." in transcript
        app.exit()
        await pilot.pause()


def test_textual_workbench_finalize_worker_success_path() -> None:
    asyncio.run(_run_finalize_success_worker_assertions())


async def _run_handler_branch_assertions() -> None:
    app = TextualWorkbenchApp()
    async with app.run_test() as pilot:
        await pilot.pause()

        app._handle_export_tools_success(
            ExportToolsResult(session_rebuilt=True, exported_tools={"demo": True})
        )
        parsed_invocations = ParsedModelResponse(
            invocations=[{"tool_name": "list_directory", "arguments": {"path": "."}}]
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
        assert "Approved pending approval request." in transcript
        assert "Finalized 0 expired approval request(s)." in transcript
        assert "Finalized 1 expired approval request(s)." in transcript

        assert (
            TextualWorkbenchApp._latest_tool_result(
                WorkflowTurnResult(
                    parsed_response=ParsedModelResponse(
                        invocations=[
                            {"tool_name": "list_directory", "arguments": {"path": "."}},
                            {
                                "tool_name": "read_file",
                                "arguments": {"path": "README.md"},
                            },
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
            ).tool_name
            == "read_file"
        )
        app.exit()
        await pilot.pause()


def test_textual_workbench_handler_branches_and_helpers() -> None:
    asyncio.run(_run_handler_branch_assertions())
