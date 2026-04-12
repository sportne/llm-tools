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

from llm_tools.apps.textual_chat.controller import (
    ChatScreenController,
    build_chat_context,
    build_chat_executor,
    build_chat_system_prompt_for_screen,
    create_provider,
)
from llm_tools.apps.textual_chat.models import (
    ChatCredentialPromptMetadata,
    TextualChatConfig,
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

from llm_tools.apps.textual_chat.presentation import (  # noqa: E402
    AssistantMarkdownEntry,
    TranscriptEntry,
)


class ChatScreen(Screen[None]):
    """Main chat screen with transcript, composer, status, and footer rows."""

    BINDINGS = [
        Binding(
            "f6", "open_transcript_copy", "Copy transcript", show=False, priority=True
        )
    ]

    DEFAULT_CSS = """
    ChatScreen {
        background: #111317;
        color: #e6e6e6;
    }

    #chat-layout {
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

    #status-bar {
        height: 1;
        min-height: 1;
        margin: 0 1;
        padding: 0 1;
        color: #d4e6ff;
        background: #1a2330;
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
        self._pending_interrupt_draft: str | None = None
        self._footer_session_tokens: int | None = None
        self._footer_active_context_tokens: int | None = None
        self._footer_confidence: float | None = None
        self._credential_prompt_completed = False
        self._controller = ChatScreenController(self)

    def compose(self) -> ComposeResult:
        with Vertical(id="chat-layout"):
            yield VerticalScroll(id="transcript")
            yield Static("", id="status-bar")
            with Horizontal(id="composer-row"):
                yield ComposerTextArea("", id="composer")
                with Vertical(id="composer-actions"):
                    yield Button("Send", id="send-button", variant="primary")
                    yield Button(
                        "Stop", id="stop-button", variant="warning", disabled=True
                    )
            yield Static("", id="footer-bar")

    def on_mount(self) -> None:
        self._controller.append_transcript(
            "system",
            (
                f"Root: {self._root_path}\n"
                f"Model: {self._active_model_name}\n"
                "Use /help for guidance. Type quit or exit to leave."
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
        self.query_one("#composer", ComposerTextArea).focus()

    def action_open_transcript_copy(self) -> None:
        self._controller.open_transcript_copy()

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

    @work(thread=True, exclusive=True)
    def _run_turn_worker(self, user_message: str) -> None:
        try:
            provider = self._provider
            if provider is None:
                raise RuntimeError("Chat provider is not configured.")
            registry, executor = build_chat_executor()
            runner = run_interactive_chat_session_turn(
                user_message=user_message,
                session_state=self._session_state,
                executor=executor,
                provider=provider,
                system_prompt=build_chat_system_prompt_for_screen(self, registry),
                base_context=build_chat_context(self),
                session_config=self._config.session,
                tool_limits=self._config.tool_limits,
                temperature=self._config.llm.temperature,
            )
            self._active_runner = runner
            for event in runner:
                if isinstance(event, ChatWorkflowStatusEvent):
                    self.app.call_from_thread(
                        self._controller.handle_turn_status,
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
