"""Tests for the prompt-schema adapter."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from llm_tools.llm_adapters import PromptSchemaAdapter
from llm_tools.tool_api import ToolSpec


class ReadFileInput(BaseModel):
    """Test input model."""

    path: str


def test_prompt_schema_adapter_renders_prompt_with_both_response_modes() -> None:
    adapter = PromptSchemaAdapter()

    prompt = adapter.export_tool_descriptions(
        specs=[ToolSpec(name="read_file", description="Read a file.")],
        input_models={"read_file": ReadFileInput},
    )

    assert "actions" in prompt
    assert "final_response" in prompt
    assert "read_file" in prompt
    assert "Return JSON only." in prompt


@pytest.mark.parametrize(
    ("payload", "expected_tool_name", "expected_final_response"),
    [
        (
            '{"actions":[{"tool_name":"read_file","arguments":{"path":"README.md"}}]}',
            "read_file",
            None,
        ),
        ('```json\n{"final_response": "Done."}\n```', None, "Done."),
        (
            [{"tool_name": "read_file", "arguments": {"path": "README.md"}}],
            "read_file",
            None,
        ),
    ],
)
def test_prompt_schema_adapter_parses_supported_payloads(
    payload: object,
    expected_tool_name: str | None,
    expected_final_response: str | None,
) -> None:
    adapter = PromptSchemaAdapter()

    response = adapter.parse_model_output(payload)

    if expected_tool_name is not None:
        assert response.invocations[0].tool_name == expected_tool_name
    else:
        assert response.invocations == []
    assert response.final_response == expected_final_response


def test_prompt_schema_adapter_rejects_invalid_payload_after_repair() -> None:
    adapter = PromptSchemaAdapter()

    with pytest.raises(ValueError):
        adapter.parse_model_output("not json")
