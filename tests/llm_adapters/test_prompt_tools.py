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


class _TypedAnswer(BaseModel):
    answer: str
    confidence: float


class _SummaryOnly(BaseModel):
    summary: str


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
    tool_name_mode = adapter.parse_decision(
        "```decision\nMODE: read_file\nTOOL_NAME: read_file\n```",
        tool_specs=_tool_specs(),
    )
    final = adapter.parse_decision(
        "```decision\nMODE: finalize\n```",
        tool_specs=_tool_specs(),
    )

    assert decision.mode == "tool"
    assert decision.tool_name == "read_file"
    assert tool_name_mode.mode == "tool"
    assert tool_name_mode.tool_name == "read_file"
    assert final.mode == "finalize"
    assert final.tool_name is None


def test_decision_stage_messages_include_tool_use_context() -> None:
    messages = PromptToolAdapter().decision_stage_messages(
        base_messages=[{"role": "user", "content": "inspect"}],
        tool_specs=_tool_specs(),
        decision_context=(
            "Tool calls already made this turn:\n"
            '1. read_file({"path":"README.md"}) -> success'
        ),
    )

    content = messages[-1]["content"]
    assert "Current turn tool-use context:" in content
    assert 'read_file({"path":"README.md"}) -> success' in content


def test_parse_decision_rejects_unknown_tools_and_duplicate_blocks() -> None:
    adapter = PromptToolAdapter()

    with pytest.raises(PromptToolProtocolError, match="finalize must not include"):
        adapter.parse_decision(
            "```decision\nMODE: finalize\nTOOL_NAME: read_file\n```",
            tool_specs=_tool_specs(),
        )
    with pytest.raises(PromptToolProtocolError, match="MODE must"):
        adapter.parse_decision(
            "```decision\nMODE: maybe\n```",
            tool_specs=_tool_specs(),
        )
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


def test_parse_tool_invocation_normalizes_common_tool_block_shapes() -> None:
    adapter = PromptToolAdapter()

    bare_tool = adapter.parse_tool_invocation(
        "```tool\nread_file\nBEGIN_ARG: Path\nREADME.md\nEND_ARG\n```",
        tool_name="read_file",
        input_model=_ReadFileInput,
    )
    colon_tool = adapter.parse_single_action(
        "```tool\nread_file:\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```",
        tool_specs=_tool_specs(),
        input_models={"read_file": _ReadFileInput, "search_text": _ReadFileInput},
        final_response_model=ChatFinalResponse,
    )

    assert bare_tool.invocations[0].tool_name == "read_file"
    assert bare_tool.invocations[0].arguments["path"] == "README.md"
    assert colon_tool.invocations[0].tool_name == "read_file"
    assert colon_tool.invocations[0].arguments["path"] == "README.md"


def test_parse_tool_invocation_rejects_extra_missing_and_nonfinal_blocks() -> None:
    adapter = PromptToolAdapter()

    with pytest.raises(PromptToolProtocolError, match="missing TOOL_NAME") as missing:
        adapter.parse_tool_invocation(
            "```tool\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```",
            tool_name="read_file",
            input_model=_ReadFileInput,
        )
    assert missing.value.invalid_payload
    with pytest.raises(PromptToolProtocolError, match="expected 'read_file'"):
        adapter.parse_tool_invocation(
            "```tool\nTOOL_NAME: search_text\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```",
            tool_name="read_file",
            input_model=_ReadFileInput,
        )
    with pytest.raises(PromptToolProtocolError, match="exactly one fenced tool"):
        adapter.parse_tool_invocation(
            (
                "```tool\nTOOL_NAME: read_file\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```\n"
                "```tool\nTOOL_NAME: read_file\nBEGIN_ARG: path\nREADME.md\nEND_ARG\n```"
            ),
            tool_name="read_file",
            input_model=_ReadFileInput,
        )
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
    with pytest.raises(PromptToolProtocolError, match="missing an argument name"):
        adapter.parse_tool_invocation(
            "```tool\nTOOL_NAME: read_file\nBEGIN_ARG:\nREADME.md\nEND_ARG\n```",
            tool_name="read_file",
            input_model=_ReadFileInput,
        )
    with pytest.raises(PromptToolProtocolError, match="Duplicate argument"):
        adapter.parse_tool_invocation(
            (
                "```tool\nTOOL_NAME: read_file\n"
                "BEGIN_ARG: path\nREADME.md\nEND_ARG\n"
                "BEGIN_ARG: path\nREADME2.md\nEND_ARG\n```"
            ),
            tool_name="read_file",
            input_model=_ReadFileInput,
        )


def test_parse_tool_invocation_handles_scalar_json_literals_and_bad_json() -> None:
    class ScalarInput(BaseModel):
        enabled: bool
        missing_value: object
        label: str

    parsed = PromptToolAdapter().parse_tool_invocation(
        (
            "```tool\n"
            "TOOL_NAME: read_file\n"
            "BEGIN_ARG: enabled\ntrue\nEND_ARG\n"
            "BEGIN_ARG: missing_value\nnull\nEND_ARG\n"
            'BEGIN_ARG: label\n{"unterminated"\nEND_ARG\n'
            "```"
        ),
        tool_name="read_file",
        input_model=ScalarInput,
    )

    assert parsed.invocations[0].arguments == {
        "enabled": True,
        "missing_value": None,
        "label": '{"unterminated"',
    }


def test_parse_single_action_tool_and_final_blocks() -> None:
    adapter = PromptToolAdapter()
    input_models = {
        "read_file": _ReadFileInput,
        "search_text": _ReadFileInput,
    }

    tool = adapter.parse_single_action(
        (
            "```tool\n"
            "TOOL_NAME: read_file\n"
            "BEGIN_ARG: path\nREADME.md\nEND_ARG\n"
            "BEGIN_ARG: limit\n5\nEND_ARG\n"
            "```"
        ),
        tool_specs=_tool_specs(),
        input_models=input_models,
        final_response_model=ChatFinalResponse,
    )
    final = adapter.parse_single_action(
        "```final\nANSWER:\nDone\n```",
        tool_specs=_tool_specs(),
        input_models=input_models,
        final_response_model=ChatFinalResponse,
    )

    assert tool.invocations[0].tool_name == "read_file"
    assert tool.invocations[0].tool_call_id is not None
    assert tool.invocations[0].arguments == {
        "path": "README.md",
        "limit": 5,
        "options": {},
    }
    assert final.final_response["answer"] == "Done"


def test_parse_single_action_accepts_plain_text_final_answer() -> None:
    adapter = PromptToolAdapter()

    parsed = adapter.parse_single_action(
        "The repository stores chat session state in FileChatSessionStore.",
        tool_specs=_tool_specs(),
        input_models={"read_file": _ReadFileInput},
        final_response_model=ChatFinalResponse,
    )

    assert (
        parsed.final_response["answer"]
        == "The repository stores chat session state in FileChatSessionStore."
    )

    tool_mention = adapter.parse_single_action(
        "The read_file tool was used to inspect README.md.",
        tool_specs=_tool_specs(),
        input_models={"read_file": _ReadFileInput},
        final_response_model=ChatFinalResponse,
    )

    assert (
        tool_mention.final_response["answer"]
        == "The read_file tool was used to inspect README.md."
    )


def test_parse_single_action_accepts_prose_with_non_protocol_json_fence() -> None:
    adapter = PromptToolAdapter()

    parsed = adapter.parse_single_action(
        (
            "The NiceGUI assistant persists durable sessions with "
            "FileHarnessStateStore.\n\n"
            "```json\n"
            '{"answer": "duplicate structured answer"}\n'
            "```"
        ),
        tool_specs=_tool_specs(),
        input_models={"read_file": _ReadFileInput},
        final_response_model=ChatFinalResponse,
    )

    assert parsed.final_response["answer"] == (
        "The NiceGUI assistant persists durable sessions with FileHarnessStateStore."
    )


def test_parse_single_action_rejects_wrong_shape_and_category_constraints() -> None:
    adapter = PromptToolAdapter()
    input_models = {"read_file": _ReadFileInput, "search_text": _ReadFileInput}
    with pytest.raises(PromptToolProtocolError, match="unknown tool"):
        adapter.parse_single_action(
            "```tool\nTOOL_NAME: missing\n```",
            tool_specs=_tool_specs(),
            input_models=input_models,
            final_response_model=ChatFinalResponse,
        )
    with pytest.raises(PromptToolProtocolError, match="Expected fenced block"):
        adapter.parse_single_action(
            "```decision\nMODE: finalize\n```",
            tool_specs=_tool_specs(),
            input_models=input_models,
            final_response_model=ChatFinalResponse,
        )
    with pytest.raises(PromptToolProtocolError, match="only substantive"):
        adapter.parse_single_action(
            "lead\n```final\nANSWER: Done\n```",
            tool_specs=_tool_specs(),
            input_models=input_models,
            final_response_model=ChatFinalResponse,
        )
    with pytest.raises(PromptToolProtocolError, match="unknown tool"):
        adapter.parse_single_action(
            (
                "```tool\n"
                "TOOL_NAME: search_text\n"
                "BEGIN_ARG: path\nREADME.md\nEND_ARG\n"
                "```"
            ),
            tool_specs=[_tool_specs()[0]],
            input_models=input_models,
            final_response_model=ChatFinalResponse,
        )
    with pytest.raises(PromptToolProtocolError, match="Expected fenced block"):
        adapter.parse_single_action(
            '```json\n{"answer": "no prose outside this block"}\n```',
            tool_specs=_tool_specs(),
            input_models=input_models,
            final_response_model=ChatFinalResponse,
        )
    with pytest.raises(PromptToolProtocolError, match="Expected exactly one"):
        adapter.parse_single_action(
            "TOOL_NAME: read_file\nBEGIN_ARG: path\nREADME.md\nEND_ARG",
            tool_specs=_tool_specs(),
            input_models=input_models,
            final_response_model=ChatFinalResponse,
        )
    with pytest.raises(PromptToolProtocolError, match="Expected exactly one"):
        adapter.parse_single_action(
            "read_file\nBEGIN_ARG: path\nREADME.md\nEND_ARG",
            tool_specs=_tool_specs(),
            input_models=input_models,
            final_response_model=ChatFinalResponse,
        )


def test_category_grouping_and_decision_parsing() -> None:
    adapter = PromptToolAdapter()
    specs = [
        ToolSpec(
            name="read_file",
            description="Read a file.",
            tags=["filesystem", "read"],
        ),
        ToolSpec(
            name="search_text",
            description="Search text.",
            tags=["text", "search", "read"],
        ),
        ToolSpec(name="custom", description="Custom."),
    ]

    categories = adapter.derive_tool_categories(specs)
    assert [category.name for category in categories] == [
        "filesystem",
        "text",
        "other",
    ]
    assert [spec.name for spec in adapter.category_tool_specs(categories, "text")] == [
        "search_text"
    ]

    parsed = adapter.parse_category_decision(
        "```category\nMODE: category\nCATEGORY: text\n```",
        categories=categories,
    )
    mode_category = adapter.parse_category_decision(
        "```category\nMODE: text\nCATEGORY: text\n```",
        categories=categories,
    )
    category_fence = adapter.parse_category_decision(
        "```text\nMODE: text\nCATEGORY: text\n```",
        categories=categories,
    )
    final = adapter.parse_category_decision(
        "```category\nMODE: finalize\n```",
        categories=categories,
    )
    assert parsed.mode == "category"
    assert parsed.category == "text"
    assert mode_category.mode == "category"
    assert mode_category.category == "text"
    assert category_fence.mode == "category"
    assert category_fence.category == "text"
    assert final.mode == "finalize"
    with pytest.raises(PromptToolProtocolError, match="unknown category"):
        adapter.parse_category_decision(
            "```category\nMODE: category\nCATEGORY: missing\n```",
            categories=categories,
        )


def test_parse_final_response_coerces_plain_text_and_validates_json() -> None:
    adapter = PromptToolAdapter()

    plain_string = adapter.parse_final_response(
        "```final\nANSWER: Done\n```",
        final_response_model=str,
    )
    plain = adapter.parse_final_response(
        "Plain markdown answer here.",
        final_response_model=ChatFinalResponse,
    )
    json_response = adapter.parse_final_response(
        '```final\nANSWER:\n{"answer": "Done", "confidence": 0.5}\n```',
        final_response_model=ChatFinalResponse,
    )

    assert plain_string.final_response == "Done"
    assert plain.final_response["answer"] == "Plain markdown answer here."
    assert json_response.final_response["answer"] == "Done"
    assert json_response.final_response["confidence"] == 0.5


def test_parse_final_response_rejects_empty_answer() -> None:
    adapter = PromptToolAdapter()
    with pytest.raises(PromptToolProtocolError, match="Missing fenced final block"):
        adapter.parse_final_response("", final_response_model=str)
    with pytest.raises(PromptToolProtocolError, match="must not be empty"):
        adapter.parse_final_response(
            "```final\nANSWER:\n```",
            final_response_model=str,
        )
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
    with pytest.raises(PromptToolProtocolError, match="JSON object"):
        adapter.parse_final_response(
            "```final\nANSWER: Done\n```",
            final_response_model=_SummaryOnly,
        )
    with pytest.raises(PromptToolProtocolError, match="Expecting"):
        adapter.parse_final_response(
            '```final\nANSWER:\n{"answer":\n```',
            final_response_model=_TypedAnswer,
        )
    with pytest.raises(PromptToolProtocolError, match="Field required"):
        adapter.parse_final_response(
            '```final\nANSWER:\n{"answer": "Done"}\n```',
            final_response_model=_TypedAnswer,
        )


def test_parse_final_response_rejects_tool_audit_metadata_answers() -> None:
    adapter = PromptToolAdapter()

    with pytest.raises(PromptToolProtocolError, match="audit metadata"):
        adapter.parse_final_response(
            (
                "```final\nANSWER:\n"
                "Tool call audit metadata, not evidence and not an answer: "
                'read_file({"path":"README.md"}).\n```'
            ),
            final_response_model=ChatFinalResponse,
        )
    with pytest.raises(PromptToolProtocolError, match="audit metadata"):
        adapter.parse_final_response(
            (
                "```final\nANSWER:\n"
                '{"answer": "Prior tool call record: name=search_text, '
                'arguments={\\"path\\":\\"src\\"}."}\n```'
            ),
            final_response_model=ChatFinalResponse,
        )


def test_prompt_tool_messages_and_repair_guidance_cover_schema_variants() -> None:
    adapter = PromptToolAdapter()
    base_messages = [{"role": "user", "content": "work"}]
    tool_spec = _tool_specs()[0]

    single_action_messages = adapter.single_action_stage_messages(
        base_messages=base_messages,
        tool_specs=_tool_specs(),
        input_models={"read_file": _ReadFileInput, "search_text": _ReadFileInput},
        final_response_model=ChatFinalResponse,
    )
    category_messages = adapter.category_decision_stage_messages(
        base_messages=base_messages,
        categories=adapter.derive_tool_categories(_tool_specs()),
        final_response_model=ChatFinalResponse,
    )
    decision_messages = adapter.decision_stage_messages(
        base_messages=base_messages,
        tool_specs=_tool_specs(),
    )
    tool_messages = adapter.tool_invocation_stage_messages(
        base_messages=base_messages,
        tool_spec=tool_spec,
        input_model=_ReadFileInput,
    )
    final_messages = adapter.final_response_stage_messages(
        base_messages=base_messages,
        final_response_model=_TypedAnswer,
    )
    repair = adapter.repair_stage_message(
        stage_name="tool:read_file",
        error=RuntimeError("bad block"),
        invalid_payload={object(): "bad"},
        selected_tool=tool_spec,
        input_model=_ReadFileInput,
    )
    unknown_repair = adapter.repair_stage_message(
        stage_name="unknown",
        error=RuntimeError(""),
        invalid_payload=None,
    )
    final_repair = adapter.repair_stage_message(
        stage_name="final_response",
        error=RuntimeError("bad final"),
        invalid_payload="bad",
        final_response_model=_TypedAnswer,
    )
    envelope_repair = adapter.repair_stage_message(
        stage_name="decision",
        error=RuntimeError("bad envelope"),
        invalid_payload='{"actions":[{"tool_name":"read_file"}],"final_response":null}',
        tool_specs=_tool_specs(),
    )

    assert "Prompt-tool output contract:" in single_action_messages[-1]["content"]
    assert "Do not write name:, library_name:" in single_action_messages[-1]["content"]
    assert "MODE is not a category name" in category_messages[-1]["content"]
    assert decision_messages[0] == base_messages[0]
    assert "MODE must be exactly" in decision_messages[-1]["content"]
    assert "BEGIN_ARG: argument_name" in tool_messages[-1]["content"]
    assert "Final response schema:" in final_messages[-1]["content"]
    assert "Tool call audit metadata" in final_messages[-1]["content"]
    assert "must not be repeated as the answer" in final_messages[-1]["content"]
    assert "Exact tool block shape:" in repair
    assert str({object(): "bad"}).startswith("{<object object")
    assert "Return only the fields required" in unknown_repair
    assert "RuntimeError" in unknown_repair
    assert "Final response schema:" in final_repair
    assert "tool result content as evidence" in final_repair
    assert "structured JSON action-envelope shape" in envelope_repair
    assert '"actions"' not in envelope_repair
