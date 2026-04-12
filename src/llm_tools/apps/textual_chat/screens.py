"""Textual modal and composer widgets for the interactive chat UI."""

from __future__ import annotations

from textual import events, on
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.message import Message
from textual.screen import ModalScreen
from textual.widgets import Button, Input, Static, TextArea

from llm_tools.apps.textual_chat.models import ChatCredentialPromptMetadata


class CredentialModal(ModalScreen[str | None]):
    """Startup modal for optional session-only credential entry."""

    def __init__(self, metadata: ChatCredentialPromptMetadata) -> None:
        super().__init__(id="credential-modal")
        self._metadata = metadata

    def compose(self) -> ComposeResult:
        prompt_label = self._metadata.api_key_env_var or "API key"
        with Vertical(id="credential-modal-body"):
            yield Static(
                f"Enter {prompt_label} for this session only, or leave it empty.",
                id="credential-copy",
            )
            yield Input(
                placeholder=prompt_label,
                password=self._metadata.mask_input,
                id="credential-input",
            )
            with Horizontal(id="credential-actions"):
                yield Button("Continue", id="credential-submit", variant="primary")
                yield Button("Cancel", id="credential-cancel")

    @on(Button.Pressed, "#credential-submit")
    def handle_submit(self) -> None:
        self.dismiss(self.query_one("#credential-input", Input).value)

    @on(Button.Pressed, "#credential-cancel")
    def handle_cancel(self) -> None:
        self.dismiss(None)


class InterruptConfirmModal(ModalScreen[bool]):
    """Confirmation modal shown when the user sends while busy."""

    def compose(self) -> ComposeResult:
        with Vertical(id="interrupt-modal-body"):
            yield Static(
                "A chat turn is already running. Interrupt it and send the current draft now?",
                id="interrupt-copy",
            )
            with Horizontal(id="interrupt-actions"):
                yield Button("Interrupt", id="interrupt-confirm", variant="warning")
                yield Button("Cancel", id="interrupt-cancel")

    @on(Button.Pressed, "#interrupt-confirm")
    def handle_confirm(self) -> None:
        self.dismiss(True)

    @on(Button.Pressed, "#interrupt-cancel")
    def handle_cancel(self) -> None:
        self.dismiss(False)


class TranscriptCopyModal(ModalScreen[None]):
    """Modal that exposes the transcript in a selectable read-only text area."""

    def __init__(self, transcript_text: str) -> None:
        super().__init__(id="transcript-copy-modal")
        self._transcript_text = transcript_text

    def compose(self) -> ComposeResult:
        with Vertical(id="transcript-copy-body"):
            yield Static(
                "Select transcript text with the mouse, then use the copy buttons.",
                id="transcript-copy-help",
            )
            yield TextArea(self._transcript_text, id="transcript-copy-area", read_only=True)
            yield Static("", id="transcript-copy-status")
            with Horizontal(id="transcript-copy-actions"):
                yield Button("Copy Selected", id="transcript-copy-selection")
                yield Button("Copy All", id="transcript-copy-all", variant="primary")
                yield Button("Close", id="transcript-copy-close")

    def on_mount(self) -> None:
        self.query_one("#transcript-copy-area", TextArea).focus()

    def _set_status(self, text: str) -> None:
        self.query_one("#transcript-copy-status", Static).update(text)

    @on(Button.Pressed, "#transcript-copy-selection")
    def handle_copy_selection(self) -> None:
        text_area = self.query_one("#transcript-copy-area", TextArea)
        selected_text = text_area.selected_text or text_area.text
        self.app.copy_to_clipboard(selected_text)
        self._set_status(
            "Copied selection." if text_area.selected_text else "No selection; copied full transcript."
        )

    @on(Button.Pressed, "#transcript-copy-all")
    def handle_copy_all(self) -> None:
        text_area = self.query_one("#transcript-copy-area", TextArea)
        text_area.select_all()
        self.app.copy_to_clipboard(text_area.text)
        self._set_status("Copied full transcript.")

    @on(Button.Pressed, "#transcript-copy-close")
    def handle_close(self) -> None:
        self.dismiss(None)


class ComposerTextArea(TextArea):
    """Chat composer with explicit Enter/Shift+Enter behavior."""

    class SubmitRequested(Message):
        """Posted when the composer should submit the current draft."""

        def __init__(self, composer: ComposerTextArea) -> None:
            super().__init__()
            self._composer = composer

        @property
        def control(self) -> ComposerTextArea:
            return self._composer

    def on_key(self, event: events.Key) -> None:
        if event.key == "shift+enter":
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.post_message(self.SubmitRequested(self))
