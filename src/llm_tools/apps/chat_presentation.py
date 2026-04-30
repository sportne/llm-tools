"""Shared presentation helpers for chat-style application layers."""

from __future__ import annotations

import json
from dataclasses import dataclass

from llm_tools.workflow_api import ChatCitation, ChatFinalResponse


@dataclass(frozen=True)
class CitationDisplay:
    """UI-ready citation label and optional excerpt."""

    label: str
    excerpt: str | None = None


@dataclass(frozen=True)
class FinalResponseDetails:
    """UI-ready supplemental final-response metadata."""

    citations: tuple[CitationDisplay, ...] = ()
    confidence_label: str | None = None
    uncertainty: tuple[str, ...] = ()
    missing_information: tuple[str, ...] = ()
    follow_up_suggestions: tuple[str, ...] = ()

    @property
    def has_content(self) -> bool:
        """Return whether there is any supplemental metadata to render."""
        return bool(
            self.citations
            or self.confidence_label
            or self.uncertainty
            or self.missing_information
            or self.follow_up_suggestions
        )


def pretty_json(value: object) -> str:
    """Return a stable, human-readable JSON representation."""
    if value is None:
        return ""
    return json.dumps(value, indent=2, sort_keys=True, default=str)


def format_citation(citation: ChatCitation) -> str:
    """Return one chat citation in transcript-friendly text form."""
    if citation.line_start is None:
        return citation.source_path
    if citation.line_end is None or citation.line_end == citation.line_start:
        return f"{citation.source_path}:{citation.line_start}"
    return f"{citation.source_path}:{citation.line_start}-{citation.line_end}"


def format_confidence_label(confidence: float | None) -> str | None:
    """Return a compact confidence label for UI display."""
    if confidence is None:
        return None
    return f"Confidence {round(confidence * 100):.0f}%"


def final_response_details(response: ChatFinalResponse) -> FinalResponseDetails:
    """Return supplemental final-response metadata shaped for UI rendering."""
    return FinalResponseDetails(
        citations=tuple(
            CitationDisplay(
                label=format_citation(citation),
                excerpt=citation.excerpt,
            )
            for citation in response.citations
        ),
        confidence_label=format_confidence_label(response.confidence),
        uncertainty=tuple(response.uncertainty),
        missing_information=tuple(response.missing_information),
        follow_up_suggestions=tuple(response.follow_up_suggestions),
    )


def format_final_response(response: ChatFinalResponse) -> str:
    """Return one final chat response in transcript-friendly text form."""
    sections = [response.answer]
    if response.citations:
        sections.append(
            "Citations:\n"
            + "\n".join(
                f"- {format_citation(citation)}" for citation in response.citations
            )
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
            + "\n".join(
                f"- {format_citation(citation)}" for citation in response.citations
            )
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


def format_transcript_text(
    role: str,
    text: str,
    *,
    assistant_completion_state: str = "complete",
) -> str:
    """Return one transcript entry in durable plain-text export form."""
    if role == "assistant":
        if assistant_completion_state == "interrupted":
            return f"Assistant (interrupted):\n{text}".rstrip()
        return f"Assistant:\n{text}".rstrip()
    if role == "user":
        return f"You:\n{text}".rstrip()
    if role == "error":
        return f"Error: {text}".rstrip()
    return f"System:\n{text}".rstrip()
