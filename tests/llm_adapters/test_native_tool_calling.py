"""Tests for the native tool-calling adapter."""

from __future__ import annotations

from typing import Any

import pytest
from openai.types.chat import (
    ChatCompletionMessage,
    ChatCompletionMessageFunctionToolCall,
)
from pydantic import BaseModel

from llm_tools.llm_adapters import NativeToolCallingAdapter
from llm_tools.tool_api import ToolSpec


class ReadFileInput(BaseModel):
    """Test input model."""

    path: str


def test_native_adapter_exports_canonical_tool_descriptions() -> None:
    adapter = NativeToolCallingAdapter()

    tools = adapter.export_tool_descriptions(
        specs=[
            ToolSpec(name="read_file", description="Read a file."),
            ToolSpec(name="list_directory", description="List a directory."),
        ],
        input_models={
            "read_file": ReadFileInput,
            "list_directory": ReadFileInput,
        },
    )

    assert [tool["function"]["name"] for tool in tools] == [
        "read_file",
        "list_directory",
    ]
    assert tools[0]["type"] == "function"
    assert tools[0]["function"]["parameters"]["type"] == "object"


def test_native_adapter_parses_sdk_tool_call_message() -> None:
    adapter = NativeToolCallingAdapter()
    payload = ChatCompletionMessage(
        role="assistant",
        content=None,
        tool_calls=[
            ChatCompletionMessageFunctionToolCall(
                id="call_123",
                type="function",
                function={
                    "name": "read_file",
                    "arguments": '{"path": "README.md"}',
                },
            )
        ],
    )

    response = adapter.parse_model_output(payload)

    assert response.final_response is None
    assert response.invocations[0].tool_name == "read_file"
    assert response.invocations[0].arguments == {"path": "README.md"}


@pytest.mark.parametrize(
    "payload",
    [
        {
            "id": "call_123",
            "type": "function",
            "function": {
                "name": "read_file",
                "arguments": {"path": "README.md"},
            },
        },
        [
            {
                "id": "call_123",
                "type": "function",
                "function": {
                    "name": "read_file",
                    "arguments": '{"path": "README.md"}',
                },
            }
        ],
        {
            "tool_calls": [
                {
                    "id": "call_123",
                    "type": "function",
                    "function": {
                        "name": "read_file",
                        "arguments": "",
                    },
                }
            ]
        },
    ],
)
def test_native_adapter_parses_raw_tool_call_payloads(payload: object) -> None:
    adapter = NativeToolCallingAdapter()

    response = adapter.parse_model_output(payload)

    assert len(response.invocations) == 1
    assert response.invocations[0].tool_name == "read_file"


def test_native_adapter_parses_final_response_text_message() -> None:
    adapter = NativeToolCallingAdapter()
    payload = ChatCompletionMessage(
        role="assistant",
        content="Here is the answer.",
        tool_calls=None,
    )

    response = adapter.parse_model_output(payload)

    assert response.invocations == []
    assert response.final_response == "Here is the answer."


def test_native_adapter_parses_final_response_text_parts() -> None:
    adapter = NativeToolCallingAdapter()

    response = adapter.parse_model_output(
        {
            "content": [
                {"type": "text", "text": "Hello"},
                {"type": "text", "text": " world"},
            ]
        }
    )

    assert response.final_response == "Hello world"


@pytest.mark.parametrize(
    "payload",
    [
        {
            "function": {
                "name": "read_file",
                "arguments": '{"path": }',
            }
        },
        {
            "function": {
                "name": "read_file",
                "arguments": '["README.md"]',
            }
        },
        {"content": [{"type": "refusal", "text": "Nope"}]},
    ],
)
def test_native_adapter_rejects_invalid_payloads(payload: dict[str, Any]) -> None:
    adapter = NativeToolCallingAdapter()

    with pytest.raises(ValueError):
        adapter.parse_model_output(payload)


@pytest.mark.parametrize(
    "payload",
    [
        123,
        {"id": "call_123", "type": "function"},
        {"function": {"name": "   ", "arguments": {}}},
        {"content": "   "},
        {"content": [123]},
        {"content": [{"type": "text", "text": {"value": ""}}]},
        {"content": [{"type": "text", "text": 123}]},
        {"content": None},
    ],
)
def test_native_adapter_rejects_additional_invalid_payload_shapes(
    payload: object,
) -> None:
    adapter = NativeToolCallingAdapter()

    with pytest.raises(ValueError):
        adapter.parse_model_output(payload)


def test_native_adapter_accepts_text_value_dict_parts() -> None:
    adapter = NativeToolCallingAdapter()

    response = adapter.parse_model_output(
        {
            "content": [
                {"type": "text", "text": {"value": "Hello"}},
                {"type": "text", "text": {"value": " world"}},
            ]
        }
    )

    assert response.final_response == "Hello world"


def test_native_adapter_accepts_objects_with_model_dump() -> None:
    adapter = NativeToolCallingAdapter()

    class FakeModelDump:
        def model_dump(self, *, mode: str, exclude_none: bool) -> dict[str, object]:
            assert mode == "json"
            assert exclude_none is True
            return {
                "function": {
                    "name": "read_file",
                    "arguments": {"path": "README.md"},
                }
            }

    response = adapter.parse_model_output(FakeModelDump())

    assert response.invocations[0].tool_name == "read_file"
