"""Shared presentation helpers for chat-style application layers."""

from __future__ import annotations

import json

from llm_tools.workflow_api import ChatCitation, ChatFinalResponse


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
