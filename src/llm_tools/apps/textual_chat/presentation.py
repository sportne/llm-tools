"""Presentation helpers for the interactive repository chat UI."""

from __future__ import annotations

from rich.console import ConsoleRenderable, Group
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.widgets import Static

from llm_tools.apps.chat_presentation import (
    format_citation,
    format_final_response,
    format_final_response_metadata,
    pretty_json,
)

__all__ = [
    "AssistantMarkdownEntry",
    "TranscriptEntry",
    "format_citation",
    "format_final_response",
    "format_final_response_metadata",
    "pretty_json",
]


class AssistantMarkdownEntry(Static):
    """One completed assistant response rendered with Textual markdown."""

    def __init__(self, *, markdown_text: str, metadata_text: str = "") -> None:
        self.markdown_text = markdown_text
        self.metadata_text = metadata_text
        self.label_text = "Assistant:"
        super().__init__(
            self._build_renderable(),
            classes="transcript-entry assistant transcript-entry-markdown",
        )

    @property
    def transcript_text(self) -> str:
        parts = [f"{self.label_text}\n{self.markdown_text}".rstrip()]
        if self.metadata_text:
            parts.append(self.metadata_text)
        return "\n\n".join(parts).rstrip()

    def _build_renderable(self) -> Group:
        renderables: list[ConsoleRenderable] = [Text(self.label_text, style="bold")]
        if self.markdown_text:
            renderables.append(RichMarkdown(self.markdown_text))
        if self.metadata_text:
            renderables.append(Text(self.metadata_text))
        return Group(*renderables)


class TranscriptEntry(Static):
    """One durable transcript row."""

    def __init__(
        self,
        *,
        role: str,
        text: str,
        assistant_completion_state: str = "complete",
    ) -> None:
        self.role = role
        self._assistant_completion_state = assistant_completion_state
        super().__init__(
            self._format_text(role, text, assistant_completion_state),
            classes=f"transcript-entry {role}",
        )

    def update_text(
        self,
        text: str,
        *,
        assistant_completion_state: str | None = None,
    ) -> None:
        if assistant_completion_state is not None:
            self._assistant_completion_state = assistant_completion_state
        self.update(
            self._format_text(self.role, text, self._assistant_completion_state)
        )

    @property
    def transcript_text(self) -> str:
        renderable = getattr(self, "renderable", None)
        if renderable is None:
            return ""
        return str(renderable).rstrip()

    @staticmethod
    def _format_text(
        role: str,
        text: str,
        assistant_completion_state: str,
    ) -> str:
        if role == "assistant":
            if assistant_completion_state == "interrupted":
                return f"Assistant (interrupted):\n{text}".rstrip()
            return f"Assistant:\n{text}"
        if role == "user":
            return f"You:\n{text}"
        if role == "error":
            return f"Error: {text}"
        return f"System:\n{text}"
