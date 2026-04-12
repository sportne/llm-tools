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
from llm_tools.apps.textual_workbench.models import ModelTurnExecutionResult
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolResult
from llm_tools.workflow_api import WorkflowTurnResult


class _SlowExportController(WorkbenchController):
    def export_tools(self, config):  # type: ignore[override]
        del config
        time.sleep(0.1)
        return [{"demo": True}]


class _FakeModelTurnController(WorkbenchController):
    def run_model_turn(self, config, *, prompt):  # type: ignore[override]
        del config, prompt
        parsed = ParsedModelResponse(final_response="done")
        return ModelTurnExecutionResult(
            exported_tools={"demo": True},
            parsed_response=parsed,
            workflow_result=WorkflowTurnResult(parsed_response=parsed),
        )


class _FakeToolCallController(WorkbenchController):
    def run_model_turn(self, config, *, prompt):  # type: ignore[override]
        del config, prompt
        parsed = ParsedModelResponse(
            invocations=[{"tool_name": "read_file", "arguments": {"path": "demo.txt"}}]
        )
        return ModelTurnExecutionResult(
            exported_tools={"demo": True},
            parsed_response=parsed,
            workflow_result=WorkflowTurnResult(
                parsed_response=parsed,
                tool_results=[
                    ToolResult(
                        ok=True,
                        tool_name="read_file",
                        tool_version="0.1.0",
                        output={"content": "demo"},
                    )
                ],
            ),
        )


class _ErrorController(WorkbenchController):
    def export_tools(self, config):  # type: ignore[override]
        del config
        raise ValueError("boom")


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
