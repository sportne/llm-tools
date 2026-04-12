"""Presentation helpers for the interactive repository chat UI."""

from __future__ import annotations

from rich.console import ConsoleRenderable, Group
from rich.markdown import Markdown as RichMarkdown
from rich.text import Text
from textual.widgets import Static

from llm_tools.workflow_api import ChatCitation, ChatFinalResponse


def format_citation(citation: ChatCitation) -> str:
    """Return one chat citation in transcript-friendly text form."""
    if citation.line_start is None:
        return citation.source_path
    if citation.line_end is None or citation.line_end == citation.line_start:
        return f"{citation.source_path}:{citation.line_start}"
    return f"{citation.source_path}:{citation.line_start}-{citation.line_end}"


def format_final_response(response: ChatFinalResponse) -> str:
    """Return one final chat response in transcript-friendly text form."""
    sections = [response.answer]
    if response.citations:
        sections.append(
            "Citations:\n"
            + "\n".join(f"- {format_citation(citation)}" for citation in response.citations)
        )
    if response.uncertainty:
        sections.append(
            "Uncertainty:\n" + "\n".join(f"- {item}" for item in response.uncertainty)
        )
    if response.missing_information:
        sections.append(
            "Missing Information:\n"
            + "\n".join(f"- {item}" for item in response.missing_information)
        )
    if response.follow_up_suggestions:
        sections.append(
            "Follow-up Suggestions:\n"
            + "\n".join(f"- {item}" for item in response.follow_up_suggestions)
        )
    return "\n\n".join(sections)


def format_final_response_metadata(response: ChatFinalResponse) -> str:
    """Return supplemental response sections as plain transcript text."""
    parts: list[str] = []
    if response.citations:
        parts.append(
            "Citations:\n"
            + "\n".join(f"- {format_citation(citation)}" for citation in response.citations)
        )
    if response.uncertainty:
        parts.append(
            "Uncertainty:\n" + "\n".join(f"- {item}" for item in response.uncertainty)
        )
    if response.missing_information:
        parts.append(
            "Missing Information:\n"
            + "\n".join(f"- {item}" for item in response.missing_information)
        )
    if response.follow_up_suggestions:
        parts.append(
            "Follow-up Suggestions:\n"
            + "\n".join(f"- {item}" for item in response.follow_up_suggestions)
        )
    return "\n\n".join(parts)


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
