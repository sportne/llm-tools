"""Textual workbench app for inspecting and exercising llm-tools."""

from __future__ import annotations

from typing import Any

from textual import on, work
from textual.app import App, ComposeResult
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.widgets import Button, Input, Static

from llm_tools.apps.textual_workbench.controller import WorkbenchController
from llm_tools.apps.textual_workbench.models import (
    DirectExecutionResult,
    ModelTurnExecutionResult,
    WorkbenchRunState,
)
from llm_tools.apps.textual_workbench.presentation import (
    append_event,
    build_tool_details_payload,
    extract_execution_record,
    pretty_json,
    sanitize_tool_result,
)
from llm_tools.apps.textual_workbench.screens import (
    JsonArgumentsTextArea,
    PromptComposerTextArea,
)


class TextualWorkbenchApp(App[None]):
    """Developer-facing one-turn workbench for llm-tools."""

    CSS = """
    Screen {
        background: #101317;
        color: #e8eef4;
    }

    #workbench-layout {
        height: 1fr;
        min-height: 0;
    }

    .pane {
        width: 1fr;
        min-height: 0;
        padding: 1;
        layout: vertical;
    }

    .pane-title {
        padding-bottom: 1;
        text-style: bold;
    }

    .section-title {
        padding-top: 1;
        text-style: bold;
    }

    .value-box {
        min-height: 1;
        padding: 0 1;
        background: #18202a;
        color: #d7e4f0;
    }

    .toggle-button {
        margin-top: 1;
    }

    .action-button {
        margin-top: 1;
    }

    .inspector-box {
        height: auto;
        padding: 0 1;
        background: #161b22;
        color: #d7e4f0;
        min-height: 2;
    }

    #transcript-scroll {
        height: 1fr;
        border: round #304050;
        margin-top: 1;
    }

    #transcript-body {
        padding: 1;
    }

    #prompt-input {
        height: 6;
    }

    #tool-args-input {
        height: 6;
    }

    #left-pane,
    #right-pane {
        overflow-y: auto;
    }

    #footer-bar {
        height: 1;
        padding: 0 1;
        background: #1a2330;
        color: #d4e6ff;
    }
    """

    BINDINGS = [
        ("ctrl+e", "export_tools", "Export tools"),
        ("ctrl+r", "run_model_turn", "Run model turn"),
        ("ctrl+d", "execute_direct_tool", "Execute direct tool"),
    ]

    def __init__(self, controller: WorkbenchController | None = None) -> None:
        super().__init__()
        self._controller = controller or WorkbenchController()
        self._config_state = self._controller.default_config()
        self._run_state = WorkbenchRunState()
        self._event_lines: list[str] = []

    def compose(self) -> ComposeResult:
        with Horizontal(id="workbench-layout"):
            with VerticalScroll(classes="pane", id="left-pane"):
                yield Static("Workbench Config", classes="pane-title")
                yield Static("Workspace", classes="section-title")
                yield Input(self._config_state.workspace, id="workspace-input")
                yield Static("Provider preset", classes="section-title")
                yield Static("", id="provider-preset-value", classes="value-box")
                yield Button(
                    "Cycle provider preset",
                    id="provider-preset-button",
                    classes="toggle-button",
                )
                yield Static("Mode", classes="section-title")
                yield Static("", id="mode-value", classes="value-box")
                yield Button(
                    "Cycle mode",
                    id="mode-button",
                    classes="toggle-button",
                )
                yield Static("Base URL", classes="section-title")
                yield Input(self._config_state.base_url, id="base-url-input")
                yield Static("Model", classes="section-title")
                yield Input(self._config_state.model, id="model-input")
                yield Static("API key", classes="section-title")
                yield Input(
                    self._config_state.api_key,
                    id="api-key-input",
                    password=True,
                )
                yield Static("Built-in tools", classes="section-title")
                yield Button("", id="toggle-filesystem-tools", classes="toggle-button")
                yield Button("", id="toggle-git-tools", classes="toggle-button")
                yield Button("", id="toggle-text-tools", classes="toggle-button")
                yield Button("", id="toggle-atlassian-tools", classes="toggle-button")
                yield Static("Policy", classes="section-title")
                yield Button("", id="toggle-allow-local-write", classes="toggle-button")
                yield Button(
                    "", id="toggle-allow-external-read", classes="toggle-button"
                )
                yield Button(
                    "", id="toggle-allow-external-write", classes="toggle-button"
                )
                yield Button("", id="toggle-allow-network", classes="toggle-button")
                yield Button("", id="toggle-allow-filesystem", classes="toggle-button")
                yield Button("", id="toggle-allow-subprocess", classes="toggle-button")
                yield Button(
                    "", id="toggle-execute-after-parse", classes="toggle-button"
                )
                yield Button(
                    "Export Tools",
                    id="export-tools-button",
                    classes="action-button",
                )
                yield Button(
                    "Run Model Turn",
                    id="run-model-turn-button",
                    classes="action-button",
                )
            with Vertical(classes="pane", id="center-pane"):
                yield Static("Turn and Execution", classes="pane-title")
                yield Static("Prompt", classes="section-title")
                yield PromptComposerTextArea("", id="prompt-input")
                yield Static("Direct tool name", classes="section-title")
                yield Input("", id="tool-name-input")
                yield Static("Direct tool JSON arguments", classes="section-title")
                yield JsonArgumentsTextArea("{}", id="tool-args-input")
                yield Button(
                    "Execute Direct Tool",
                    id="execute-direct-tool-button",
                    classes="action-button",
                )
                yield Static("Events", classes="section-title")
                with VerticalScroll(id="transcript-scroll"):
                    yield Static("", id="transcript-body")
            with VerticalScroll(classes="pane", id="right-pane"):
                yield Static("Inspector", classes="pane-title")
                yield Static("Registered tools", classes="section-title")
                yield Static("", id="registered-tools-box", classes="inspector-box")
                yield Static("Selected tool", classes="section-title")
                yield Static("", id="selected-tool-box", classes="inspector-box")
                yield Static("Exported payload", classes="section-title")
                yield Static("", id="exported-tools-box", classes="inspector-box")
                yield Static("Parsed response", classes="section-title")
                yield Static("", id="parsed-response-box", classes="inspector-box")
                yield Static("Workflow result", classes="section-title")
                yield Static("", id="workflow-result-box", classes="inspector-box")
                yield Static("Selected tool result", classes="section-title")
                yield Static("", id="direct-result-box", classes="inspector-box")
                yield Static("Execution record", classes="section-title")
                yield Static("", id="execution-record-box", classes="inspector-box")
        yield Static("", id="footer-bar")

    def on_mount(self) -> None:
        self._sync_ui_from_state()
        self._append_event("Textual workbench ready.")

    def action_export_tools(self) -> None:
        self._start_export_tools()

    def action_run_model_turn(self) -> None:
        self._start_model_turn()

    def action_execute_direct_tool(self) -> None:
        self._start_direct_tool_execution()

    @on(Button.Pressed)
    def handle_button_press(self, event: Button.Pressed) -> None:
        button_id = event.button.id
        if button_id is None:
            return

        if self._handle_cycle_button(button_id):
            return

        if self._handle_toggle_button(button_id):
            return

        if button_id == "export-tools-button":
            self._start_export_tools()
            return
        if button_id == "run-model-turn-button":
            self._start_model_turn()
            return
        if button_id == "execute-direct-tool-button":
            self._start_direct_tool_execution()

    @on(Input.Changed, "#tool-name-input")
    def handle_tool_name_change(self) -> None:
        self._sync_registry_inspector()

    def _handle_cycle_button(self, button_id: str) -> bool:
        if button_id == "provider-preset-button":
            self._config_state = self._controller.cycle_provider_preset(
                self._config_state
            )
            self._sync_ui_from_state()
            return True

        if button_id == "mode-button":
            self._config_state = self._controller.cycle_mode(self._config_state)
            self._sync_ui_from_state()
            return True

        return False

    def _handle_toggle_button(self, button_id: str) -> bool:
        toggle_fields = {
            "toggle-filesystem-tools": "enable_filesystem_tools",
            "toggle-git-tools": "enable_git_tools",
            "toggle-text-tools": "enable_text_tools",
            "toggle-atlassian-tools": "enable_atlassian_tools",
            "toggle-allow-local-write": "allow_local_write",
            "toggle-allow-external-read": "allow_external_read",
            "toggle-allow-external-write": "allow_external_write",
            "toggle-allow-network": "allow_network",
            "toggle-allow-filesystem": "allow_filesystem",
            "toggle-allow-subprocess": "allow_subprocess",
            "toggle-execute-after-parse": "execute_after_parse",
        }
        field_name = toggle_fields.get(button_id)
        if field_name is None:
            return False

        self._flip_config_bool(field_name)
        return True

    def _flip_config_bool(self, field_name: str) -> None:
        value = getattr(self._config_state, field_name)
        self._config_state = self._config_state.model_copy(
            update={field_name: not value}
        )
        self._sync_ui_from_state()

    def _start_export_tools(self) -> None:
        if self._run_state.busy:
            return
        self._read_inputs_into_config()
        self._set_busy(True, "Exporting tools")
        self._run_export_tools_worker()

    def _start_model_turn(self) -> None:
        if self._run_state.busy:
            return
        self._read_inputs_into_config()
        self._set_busy(True, "Running model turn")
        self._run_model_turn_worker(
            self.query_one("#prompt-input", PromptComposerTextArea).text
        )

    def _start_direct_tool_execution(self) -> None:
        if self._run_state.busy:
            return
        self._read_inputs_into_config()
        self._set_busy(True, "Executing direct tool")
        self._run_direct_tool_worker(
            self.query_one("#tool-name-input", Input).value,
            self.query_one("#tool-args-input", JsonArgumentsTextArea).text,
        )

    @work(thread=True, exclusive=True)
    def _run_export_tools_worker(self) -> None:
        try:
            exported = self._controller.export_tools(self._config_state)
        except Exception as exc:
            self.call_from_thread(self._handle_worker_error, str(exc))
            return

        self.call_from_thread(self._handle_export_tools_success, exported)

    @work(thread=True, exclusive=True)
    def _run_model_turn_worker(self, prompt: str) -> None:
        try:
            result = self._controller.run_model_turn(self._config_state, prompt=prompt)
        except Exception as exc:
            self.call_from_thread(self._handle_worker_error, str(exc))
            return

        self.call_from_thread(self._handle_model_turn_success, result)

    @work(thread=True, exclusive=True)
    def _run_direct_tool_worker(self, tool_name: str, arguments_text: str) -> None:
        try:
            result = self._controller.execute_direct_tool(
                self._config_state,
                tool_name=tool_name,
                arguments_text=arguments_text,
            )
        except Exception as exc:
            self.call_from_thread(self._handle_worker_error, str(exc))
            return

        self.call_from_thread(self._handle_direct_tool_success, result)

    def _handle_export_tools_success(self, exported: Any) -> None:
        self._run_state = self._run_state.model_copy(
            update={
                "busy": False,
                "current_status_text": "Export complete",
                "last_exported_tools": exported,
            }
        )
        self._append_event("Exported tools for the current adapter mode.")
        self._sync_ui_from_state()

    def _handle_model_turn_success(self, result: ModelTurnExecutionResult) -> None:
        workflow_result = result.workflow_result
        last_tool_result = None
        if workflow_result is not None and workflow_result.tool_results:
            last_tool_result = workflow_result.tool_results[-1]
        self._run_state = self._run_state.model_copy(
            update={
                "busy": False,
                "current_status_text": "Model turn complete",
                "last_exported_tools": result.exported_tools,
                "last_parsed_response": result.parsed_response,
                "last_workflow_result": workflow_result,
                "last_direct_tool_result": last_tool_result,
            }
        )
        if result.parsed_response.final_response is not None:
            self._append_event(
                "Model returned a final response without tool execution."
            )
        else:
            self._append_event(
                "Model returned tool invocations and the workflow executed them."
            )
        self._sync_ui_from_state()

    def _handle_direct_tool_success(self, result: DirectExecutionResult) -> None:
        self._run_state = self._run_state.model_copy(
            update={
                "busy": False,
                "current_status_text": "Direct tool execution complete",
                "last_direct_tool_result": result.tool_result,
            }
        )
        self._append_event(f"Directly executed tool '{result.tool_result.tool_name}'.")
        self._sync_ui_from_state()

    def _handle_worker_error(self, message: str) -> None:
        self._run_state = self._run_state.model_copy(
            update={
                "busy": False,
                "current_status_text": "Error",
            }
        )
        self._append_event(f"Error: {message}")
        self._sync_ui_from_state()

    def _read_inputs_into_config(self) -> None:
        self._config_state = self._config_state.model_copy(
            update={
                "workspace": self.query_one("#workspace-input", Input).value,
                "base_url": self.query_one("#base-url-input", Input).value,
                "model": self.query_one("#model-input", Input).value,
                "api_key": self.query_one("#api-key-input", Input).value,
            }
        )

    def _sync_ui_from_state(self) -> None:
        self.query_one("#provider-preset-value", Static).update(
            self._config_state.provider_preset.value
        )
        self.query_one("#mode-value", Static).update(self._config_state.mode.value)
        self.query_one("#toggle-filesystem-tools", Button).label = self._toggle_label(
            "Filesystem tools",
            self._config_state.enable_filesystem_tools,
        )
        self.query_one("#toggle-git-tools", Button).label = self._toggle_label(
            "Git tools",
            self._config_state.enable_git_tools,
        )
        self.query_one("#toggle-text-tools", Button).label = self._toggle_label(
            "Text tools",
            self._config_state.enable_text_tools,
        )
        self.query_one("#toggle-atlassian-tools", Button).label = self._toggle_label(
            "Atlassian tools",
            self._config_state.enable_atlassian_tools,
        )
        self.query_one("#toggle-allow-local-write", Button).label = self._toggle_label(
            "Allow local write",
            self._config_state.allow_local_write,
        )
        self.query_one(
            "#toggle-allow-external-read", Button
        ).label = self._toggle_label(
            "Allow external read",
            self._config_state.allow_external_read,
        )
        self.query_one(
            "#toggle-allow-external-write", Button
        ).label = self._toggle_label(
            "Allow external write",
            self._config_state.allow_external_write,
        )
        self.query_one("#toggle-allow-network", Button).label = self._toggle_label(
            "Allow network",
            self._config_state.allow_network,
        )
        self.query_one("#toggle-allow-filesystem", Button).label = self._toggle_label(
            "Allow filesystem",
            self._config_state.allow_filesystem,
        )
        self.query_one("#toggle-allow-subprocess", Button).label = self._toggle_label(
            "Allow subprocess",
            self._config_state.allow_subprocess,
        )
        self.query_one(
            "#toggle-execute-after-parse", Button
        ).label = self._toggle_label(
            "Execute after parse",
            self._config_state.execute_after_parse,
        )
        self._sync_registry_inspector()
        self._sync_result_inspector()
        self._sync_footer()
        self._sync_busy_buttons()

    def _sync_registry_inspector(self) -> None:
        tool_names = self._controller.list_tool_names(self._config_state)
        self.query_one("#registered-tools-box", Static).update("\n".join(tool_names))
        tool_name_input = self.query_one("#tool-name-input", Input)
        if tool_names and tool_name_input.value.strip() == "":
            tool_name_input.value = tool_names[0]
        spec_payload, input_schema = self._controller.get_tool_details(
            self._config_state,
            tool_name_input.value,
        )
        self.query_one("#selected-tool-box", Static).update(
            pretty_json(
                build_tool_details_payload(
                    spec_payload=spec_payload,
                    input_schema=input_schema,
                )
            )
        )

    def _sync_result_inspector(self) -> None:
        self.query_one("#exported-tools-box", Static).update(
            pretty_json(self._run_state.last_exported_tools)
        )
        parsed_response = self._run_state.last_parsed_response
        self.query_one("#parsed-response-box", Static).update(
            pretty_json(
                None
                if parsed_response is None
                else parsed_response.model_dump(mode="json")
            )
        )
        workflow_result = self._run_state.last_workflow_result
        self.query_one("#workflow-result-box", Static).update(
            pretty_json(
                None
                if workflow_result is None
                else workflow_result.model_dump(mode="json")
            )
        )
        direct_tool_result = self._run_state.last_direct_tool_result
        self.query_one("#direct-result-box", Static).update(
            pretty_json(sanitize_tool_result(direct_tool_result))
        )
        self.query_one("#execution-record-box", Static).update(
            pretty_json(extract_execution_record(direct_tool_result))
        )

    def _sync_footer(self) -> None:
        busy_text = "busy" if self._run_state.busy else "idle"
        footer = (
            f"workspace: {self._config_state.workspace or '-'}"
            f" | provider: {self._config_state.provider_preset.value}"
            f" | model: {self._config_state.model or '-'}"
            f" | state: {busy_text}"
            " | Ctrl+E export | Ctrl+R run turn | Ctrl+D direct tool"
        )
        self.query_one("#footer-bar", Static).update(footer)

    def _sync_busy_buttons(self) -> None:
        busy = self._run_state.busy
        for button_id in (
            "#export-tools-button",
            "#run-model-turn-button",
            "#execute-direct-tool-button",
        ):
            self.query_one(button_id, Button).disabled = busy

    def _append_event(self, message: str) -> None:
        transcript_text = append_event(self._event_lines, message)
        self.query_one("#transcript-body", Static).update(transcript_text)

    def _set_busy(self, busy: bool, status_text: str) -> None:
        self._run_state = self._run_state.model_copy(
            update={"busy": busy, "current_status_text": status_text}
        )
        self._append_event(status_text)
        self._sync_ui_from_state()

    @staticmethod
    def _toggle_label(label: str, enabled: bool) -> str:
        suffix = "ON" if enabled else "OFF"
        return f"{label}: {suffix}"


def run_workbench_app() -> None:
    """Launch the Textual workbench."""
    TextualWorkbenchApp().run()


def main() -> int:
    """Console entrypoint for the Textual workbench."""
    run_workbench_app()
    return 0
