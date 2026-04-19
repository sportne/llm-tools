"""Textual chat app shell for interactive repository chat."""

from __future__ import annotations

from os import getenv
from pathlib import Path

from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical, VerticalScroll
from textual.screen import Screen
from textual.widgets import Button, Static

from llm_tools.apps.chat_controls import (
    build_chat_control_state,
    build_startup_message,
)
from llm_tools.apps.protection_runtime import (
    build_protection_controller,
    build_protection_environment,
)
from llm_tools.apps.textual_chat.controller import (
    ChatScreenController,
    build_available_tool_specs,
    build_chat_context,
    build_chat_executor,
    build_chat_system_prompt_for_screen,
    create_provider,
)
from llm_tools.apps.textual_chat.models import (
    ChatCredentialPromptMetadata,
    TextualChatConfig,
)
from llm_tools.apps.textual_chat.presentation import (
    AssistantMarkdownEntry,
    TranscriptEntry,
)
from llm_tools.apps.textual_chat.screens import (
    ComposerTextArea,
    CredentialModal,
    InterruptConfirmModal,
    TranscriptCopyModal,
)
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.workflow_api import (
    ChatSessionState,
    ChatSessionTurnRunner,
    ChatWorkflowApprovalEvent,
    ChatWorkflowApprovalResolvedEvent,
    ChatWorkflowApprovalState,
    ChatWorkflowInspectorEvent,
    ChatWorkflowStatusEvent,
    run_interactive_chat_session_turn,
)

__all__ = [
    "AssistantMarkdownEntry",
    "ChatApp",
    "ChatScreen",
    "ComposerTextArea",
    "CredentialModal",
    "InterruptConfirmModal",
    "TranscriptCopyModal",
    "TranscriptEntry",
    "run_chat_app",
]


class ChatScreen(Screen[None]):
    """Main chat screen with transcript, composer, status, and footer rows."""

    BINDINGS = [
        Binding(
            "f6", "open_transcript_copy", "Copy transcript", show=False, priority=True
        ),
        Binding(
            "f7", "toggle_inspector", "Toggle inspector", show=False, priority=True
        ),
    ]

    DEFAULT_CSS = """
    ChatScreen {
        background: #111317;
        color: #e6e6e6;
    }

    #chat-layout {
        layout: horizontal;
    }

    #chat-main {
        width: 1fr;
        min-width: 0;
        layout: vertical;
    }

    #transcript {
        height: 1fr;
        padding: 1 1 0 1;
    }

    .transcript-entry {
        width: 100%;
        margin: 0 0 1 0;
        padding: 0 1;
        background: #17191d;
    }

    .transcript-entry.user {
        background: #1a2028;
        border-left: tall #4f6f96;
    }

    .transcript-entry.assistant {
        background: #171b1f;
        border-left: tall #4f8a7d;
    }

    .transcript-entry.system {
        background: #16161a;
        color: #b8bec8;
        border-left: tall #5b5f66;
    }

    .transcript-entry.error {
        background: #261718;
        color: #f3d8d9;
        border-left: tall #b76363;
    }

    #status-row {
        height: auto;
        min-height: 3;
        margin: 0 1;
        background: #1a2330;
        align: left middle;
    }

    #status-text {
        width: 1fr;
        min-height: 1;
        padding: 0 1;
        color: #d4e6ff;
    }

    #approve-button,
    #deny-button {
        width: 10;
        margin: 0 1 0 0;
    }

    #composer-row {
        height: 11;
        margin: 0 1;
        layout: horizontal;
    }

    #composer {
        width: 1fr;
    }

    #composer-actions {
        width: 11;
        height: auto;
        margin-left: 1;
        layout: vertical;
    }

    #composer-actions Button {
        width: 100%;
        height: 3;
        min-height: 3;
        margin-bottom: 1;
    }

    #stop-button {
        margin-bottom: 0;
    }

    #footer-bar {
        height: 1;
        min-height: 1;
        margin: 0 1 1 1;
        padding: 0 1;
        color: #c7c7c7;
    }

    #inspector-pane {
        width: 44;
        min-width: 30;
        padding: 1 1 1 0;
    }

    .inspector-title {
        padding-bottom: 1;
        text-style: bold;
    }

    .inspector-section {
        padding-top: 1;
        text-style: bold;
    }

    .inspector-box {
        min-height: 2;
        padding: 0 1;
        background: #161b22;
        color: #d7e4f0;
    }
    """

    def __init__(
        self,
        *,
        root_path: Path,
        config: TextualChatConfig,
        provider: OpenAICompatibleProvider | None,
        credential_metadata_override: ChatCredentialPromptMetadata | None = None,
    ) -> None:
        super().__init__(id="chat-screen")
        self._root_path = root_path
        self._config = config
        self._provider = provider
        self._active_model_name = config.llm.model_name
        self._credential_metadata_override = credential_metadata_override
        self._create_provider = create_provider
        self._session_state = ChatSessionState()
        self._credential_secret: str | None = None
        self._busy = False
        self._active_assistant_entry: TranscriptEntry | None = None
        self._active_runner: ChatSessionTurnRunner | None = None
        self._available_tool_specs = build_available_tool_specs()
        control_state = build_chat_control_state(
            config,
            available_tool_names=set(self._available_tool_specs),
        )
        self._default_enabled_tools = set(control_state.default_enabled_tools)
        self._enabled_tools = set(control_state.enabled_tools)
        self._require_approval_for = set(control_state.require_approval_for)
        self._pending_approval: ChatWorkflowApprovalState | None = None
        self._approval_decision_in_flight = False
        self._inspector_open = control_state.inspector_open
        self._active_turn_number = 0
        self._inspector_provider_messages: list[dict[str, object]] = []
        self._inspector_parsed_responses: list[dict[str, object]] = []
        self._inspector_tool_executions: list[dict[str, object]] = []
        self._pending_interrupt_draft: str | None = None
        self._footer_session_tokens: int | None = None
        self._footer_active_context_tokens: int | None = None
        self._footer_confidence: float | None = None
        self._credential_prompt_completed = False
        self._controller = ChatScreenController(self)

    def compose(self) -> ComposeResult:
        with Horizontal(id="chat-layout"):
            with Vertical(id="chat-main"):
                yield VerticalScroll(id="transcript")
                with Horizontal(id="status-row"):
                    yield Static("", id="status-text")
                    yield Button(
                        "Approve",
                        id="approve-button",
                        variant="success",
                    )
                    yield Button(
                        "Deny",
                        id="deny-button",
                        variant="error",
                    )
                with Horizontal(id="composer-row"):
                    yield ComposerTextArea("", id="composer")
                    with Vertical(id="composer-actions"):
                        yield Button("Send", id="send-button", variant="primary")
                        yield Button(
                            "Stop", id="stop-button", variant="warning", disabled=True
                        )
                yield Static("", id="footer-bar")
            with VerticalScroll(id="inspector-pane"):
                yield Static("Inspector", classes="inspector-title")
                yield Static("Current Tool State", classes="inspector-section")
                yield Static("", id="tool-state-box", classes="inspector-box")
                yield Static("Pending Approval", classes="inspector-section")
                yield Static("", id="pending-approval-box", classes="inspector-box")
                yield Static("Model Messages", classes="inspector-section")
                yield Static("", id="provider-messages-box", classes="inspector-box")
                yield Static("Parsed Responses", classes="inspector-section")
                yield Static("", id="parsed-response-box", classes="inspector-box")
                yield Static("Tool Execution Records", classes="inspector-section")
                yield Static("", id="tool-execution-box", classes="inspector-box")

    def on_mount(self) -> None:
        self._controller.append_transcript(
            "system",
            build_startup_message(
                root_path=self._root_path,
                model_name=self._active_model_name,
                exit_hint="Type quit or exit to leave.",
            ),
        )
        metadata = (
            self._credential_metadata_override
            or self._config.llm.credential_prompt_metadata()
        )
        env_var = self._config.llm.api_key_env_var or "OPENAI_API_KEY"
        if self._provider is None and metadata.expects_api_key and not getenv(env_var):
            self.app.push_screen(
                CredentialModal(metadata),
                callback=self._controller.handle_credential_submit,
            )
        self._controller.refresh_footer()
        self._controller.refresh_inspector()
        self.query_one("#composer", ComposerTextArea).focus()

    def action_open_transcript_copy(self) -> None:
        self._controller.open_transcript_copy()

    def action_toggle_inspector(self) -> None:
        self._controller.toggle_inspector()

    @on(ComposerTextArea.SubmitRequested, "#composer")
    def handle_composer_submit(self) -> None:
        self._controller.submit_draft(
            self.query_one("#composer", ComposerTextArea).text
        )

    @on(Button.Pressed, "#send-button")
    def handle_send_button(self) -> None:
        self._controller.submit_draft(
            self.query_one("#composer", ComposerTextArea).text
        )

    @on(Button.Pressed, "#stop-button")
    def handle_stop_button(self) -> None:
        if not self._busy:
            return
        self._pending_interrupt_draft = None
        self._controller.cancel_active_turn(status_text="stopping")

    @on(Button.Pressed, "#approve-button")
    def handle_approve_button(self) -> None:
        self._controller.resolve_active_approval(approved=True)

    @on(Button.Pressed, "#deny-button")
    def handle_deny_button(self) -> None:
        self._controller.resolve_active_approval(approved=False)

    @work(thread=True, exclusive=True)
    def _run_turn_worker(self, user_message: str) -> None:
        try:
            provider = self._provider
            if provider is None:
                raise RuntimeError("Chat provider is not configured.")
            registry, executor = build_chat_executor(self)
            protection_controller = build_protection_controller(
                config=self._config.protection,
                provider=provider,
                environment=build_protection_environment(
                    app_name="textual_chat",
                    model_name=self._active_model_name,
                    workspace=str(self._root_path)
                    if self._root_path is not None
                    else None,
                    enabled_tools=self._enabled_tools,
                    allow_network=False,
                    allow_filesystem=True,
                    allow_subprocess=False,
                ),
            )
            runner = run_interactive_chat_session_turn(
                user_message=user_message,
                session_state=self._session_state,
                executor=executor,
                provider=provider,
                system_prompt=build_chat_system_prompt_for_screen(self, registry),
                base_context=build_chat_context(self),
                session_config=self._config.session,
                tool_limits=self._config.tool_limits,
                redaction_config=self._config.policy.redaction,
                temperature=self._config.llm.temperature,
                protection_controller=protection_controller,
            )
            self._active_runner = runner
            for event in runner:
                if isinstance(event, ChatWorkflowStatusEvent):
                    self.app.call_from_thread(
                        self._controller.handle_turn_status,
                        event.model_dump(mode="json"),
                    )
                    continue
                if isinstance(event, ChatWorkflowApprovalEvent):
                    self.app.call_from_thread(
                        self._controller.handle_turn_approval_requested,
                        event.model_dump(mode="json"),
                    )
                    continue
                if isinstance(event, ChatWorkflowApprovalResolvedEvent):
                    self.app.call_from_thread(
                        self._controller.handle_turn_approval_resolved,
                        event.model_dump(mode="json"),
                    )
                    continue
                if isinstance(event, ChatWorkflowInspectorEvent):
                    self.app.call_from_thread(
                        self._controller.handle_turn_inspector_event,
                        event.model_dump(mode="json"),
                    )
                    continue
                self.app.call_from_thread(
                    self._controller.handle_turn_result,
                    event.model_dump(mode="json"),
                )
        except Exception as exc:
            self.app.call_from_thread(self._controller.handle_turn_error, str(exc))
        finally:
            self._active_runner = None


class ChatApp(App[None]):
    """Textual shell for interactive repository chat."""

    def __init__(
        self,
        *,
        root_path: Path,
        config: TextualChatConfig,
        provider: OpenAICompatibleProvider | None,
        credential_metadata_override: ChatCredentialPromptMetadata | None = None,
    ) -> None:
        super().__init__()
        self._root_path = root_path
        self._config = config
        self._provider = provider
        self._credential_metadata_override = credential_metadata_override

    def on_mount(self) -> None:
        self.push_screen(
            ChatScreen(
                root_path=self._root_path,
                config=self._config,
                provider=self._provider,
                credential_metadata_override=self._credential_metadata_override,
            )
        )


def run_chat_app(
    *,
    root_path: Path,
    config: TextualChatConfig,
    provider: OpenAICompatibleProvider | None = None,
) -> int:
    """Launch the Textual repository chat app."""
    ChatApp(root_path=root_path, config=config, provider=provider).run()
    return 0
