"""Tests for the structured-response adapter."""

from __future__ import annotations

import pytest

from llm_tools.llm_adapters import StructuredResponseAdapter


def test_structured_response_adapter_exports_canonical_schema() -> None:
    adapter = StructuredResponseAdapter()

    schema = adapter.export_tool_descriptions(specs=[], input_models={})

    assert "actions" in schema["properties"]
    assert "final_response" in schema["properties"]


def test_structured_response_adapter_restricts_tool_names_in_schema() -> None:
    from llm_tools.tool_api import ToolSpec

    adapter = StructuredResponseAdapter()
    schema = adapter.export_tool_descriptions(
        specs=[ToolSpec(name="read_file", description="Read a file.")],
        input_models={},
    )

    tool_name_schema = schema["properties"]["actions"]["items"]["properties"][
        "tool_name"
    ]
    assert tool_name_schema["enum"] == ["read_file"]


@pytest.mark.parametrize(
    ("payload", "tool_name"),
    [
        (
            '{"actions":[{"tool_name":"read_file","arguments":{"path":"README.md"}}]}',
            "read_file",
        ),
        ({"tool_name": "read_file", "arguments": {"path": "README.md"}}, "read_file"),
        ([{"tool_name": "read_file", "arguments": {"path": "README.md"}}], "read_file"),
    ],
)
def test_structured_response_adapter_parses_action_payloads(
    payload: object, tool_name: str
) -> None:
    adapter = StructuredResponseAdapter()

    response = adapter.parse_model_output(payload)

    assert response.final_response is None
    assert response.invocations[0].tool_name == tool_name


def test_structured_response_adapter_parses_final_response_payload() -> None:
    adapter = StructuredResponseAdapter()

    response = adapter.parse_model_output({"final_response": "Done."})

    assert response.invocations == []
    assert response.final_response == "Done."


@pytest.mark.parametrize(
    "payload",
    [
        '{"actions": [], "final_response": null}',
        '{"actions": [], "final_response": "   "}',
        '{"not_actions": []}',
        "not json",
        123,
    ],
)
def test_structured_response_adapter_rejects_invalid_payloads(payload: object) -> None:
    adapter = StructuredResponseAdapter()

    with pytest.raises((ValueError, Exception)):
        adapter.parse_model_output(payload)
