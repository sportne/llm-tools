"""Tests for the prompt-tool fenced protocol adapter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm_tools.llm_adapters import PromptToolAdapter, PromptToolProtocolError
from llm_tools.tool_api import ToolSpec
from llm_tools.workflow_api import ChatFinalResponse


class _ReadFileInput(BaseModel):
    path: str
    limit: int = 10
    options: dict[str, object] = {}


def _tool_specs() -> list[ToolSpec]:
    return [
        ToolSpec(name="read_file", description="Read a file."),
        ToolSpec(name="search_text", description="Search text."),
    ]


def test_parse_decision_tool_and_finalize() -> None:
    adapter = PromptToolAdapter()

    decision = adapter.parse_decision(
        "```decision\nMODE: tool\nTOOL_NAME: read_file\n```",
        tool_specs=_tool_specs(),
    )
    final = adapter.parse_decision(
        "```decision\nMODE: finalize\n```",
        tool_specs=_tool_specs(),
    )

    assert decision.mode == "tool"
    assert decision.tool_name == "read_file"
    assert final.mode == "finalize"
    assert final.tool_name is None


def test_parse_decision_rejects_unknown_tools_and_duplicate_blocks() -> None:
    adapter = PromptToolAdapter()

    with pytest.raises(PromptToolProtocolError, match="unknown tool"):
        adapter.parse_decision(
            "```decision\nMODE: tool\nTOOL_NAME: missing\n```",
            tool_specs=_tool_specs(),
        )
    with pytest.raises(PromptToolProtocolError, match="exactly one"):
        adapter.parse_decision(
            "```decision\nMODE: finalize\n```\n```decision\nMODE: finalize\n```",
            tool_specs=_tool_specs(),
        )
    with pytest.raises(PromptToolProtocolError, match="only one"):
        adapter.parse_decision(
            "```decision\nMODE: finalize\n```\n```tool\nTOOL_NAME: read_file\n```",
            tool_specs=_tool_specs(),
        )


def test_parse_tool_invocation_handles_multiline_and_json_arguments() -> None:
    parsed = PromptToolAdapter().parse_tool_invocation(
        (
            "Brief lead-in.\n"
            "```tool\n"
            "TOOL_NAME: read_file\n"
            "BEGIN_ARG: path\n"
            "docs/notes.md\n"
            "END_ARG\n"
            "BEGIN_ARG: limit\n"
            "25\n"
            "END_ARG\n"
            "BEGIN_ARG: options\n"
            '{"encoding": "utf-8", "lines": [1, 2]}\n'
            "END_ARG\n"
            "```"
        ),
        tool_name="read_file",
        input_model=_ReadFileInput,
    )

    assert len(parsed.invocations) == 1
    invocation = parsed.invocations[0]
    assert invocation.tool_name == "read_file"
    assert invocation.arguments == {
        "path": "docs/notes.md",
        "limit": 25,
        "options": {"encoding": "utf-8", "lines": [1, 2]},
    }


def test_parse_tool_invocation_rejects_extra_missing_and_nonfinal_blocks() -> None:
    adapter = PromptToolAdapter()

    with pytest.raises(PromptToolProtocolError, match="unknown fields"):
        adapter.parse_tool_invocation(
            (
                "```tool\n"
                "TOOL_NAME: read_file\n"
                "BEGIN_ARG: path\nREADME.md\nEND_ARG\n"
                "BEGIN_ARG: extra\nvalue\nEND_ARG\n"
                "```"
            ),
            tool_name="read_file",
            input_model=_ReadFileInput,
        )
    with pytest.raises(PromptToolProtocolError, match="Field required"):
        adapter.parse_tool_invocation(
            "```tool\nTOOL_NAME: read_file\n```",
            tool_name="read_file",
            input_model=_ReadFileInput,
        )
    with pytest.raises(PromptToolProtocolError, match="final substantive"):
        adapter.parse_tool_invocation(
            "```tool\nTOOL_NAME: read_file\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```\ntrailing",
            tool_name="read_file",
            input_model=_ReadFileInput,
        )
    with pytest.raises(PromptToolProtocolError, match="missing END_ARG"):
        adapter.parse_tool_invocation(
            "```tool\nTOOL_NAME: read_file\nBEGIN_ARG: path\nREADME.md\n```",
            tool_name="read_file",
            input_model=_ReadFileInput,
        )


def test_parse_final_response_coerces_plain_text_and_validates_json() -> None:
    adapter = PromptToolAdapter()

    plain = adapter.parse_final_response(
        "Plain markdown answer here.",
        final_response_model=ChatFinalResponse,
    )
    json_response = adapter.parse_final_response(
        '```final\nANSWER:\n{"answer": "Done", "confidence": 0.5}\n```',
        final_response_model=ChatFinalResponse,
    )

    assert plain.final_response["answer"] == "Plain markdown answer here."
    assert json_response.final_response["answer"] == "Done"
    assert json_response.final_response["confidence"] == 0.5


def test_parse_final_response_rejects_empty_answer() -> None:
    adapter = PromptToolAdapter()
    with pytest.raises(PromptToolProtocolError, match="answer"):
        adapter.parse_final_response(
            "```final\nANSWER:\n```",
            final_response_model=ChatFinalResponse,
        )
    with pytest.raises(PromptToolProtocolError, match="only one"):
        PromptToolAdapter().parse_final_response(
            "```tool\nTOOL_NAME: read_file\n```\n```final\nANSWER:\nDone\n```",
            final_response_model=ChatFinalResponse,
        )
