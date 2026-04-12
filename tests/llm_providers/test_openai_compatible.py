"""Tests for the OpenAI-compatible provider layer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from pydantic import BaseModel

from llm_tools.llm_adapters import (
    NativeToolCallingAdapter,
    PromptSchemaAdapter,
    StructuredOutputAdapter,
)
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import Tool, ToolContext, ToolRegistry, ToolSpec


class EchoInput(BaseModel):
    """Test input."""

    value: str


class EchoOutput(BaseModel):
    """Test output."""

    echoed: str


class EchoTool(Tool[EchoInput, EchoOutput]):
    spec = ToolSpec(
        name="echo",
        description="Echo a value.",
        tags=["example"],
    )
    input_model = EchoInput
    output_model = EchoOutput

    def invoke(self, context: ToolContext, args: EchoInput) -> EchoOutput:
        return EchoOutput(echoed=f"{context.invocation_id}:{args.value}")


@dataclass
class _FakeChoice:
    message: object


@dataclass
class _FakeResponse:
    choices: list[_FakeChoice]


class _FakeCompletions:
    def __init__(self, response: object) -> None:
        self.response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> _FakeResponse:
        self.calls.append(kwargs)
        return _FakeResponse(choices=[_FakeChoice(message=self.response)])


class _FakeChat:
    def __init__(self, response: object) -> None:
        self.completions = _FakeCompletions(response)


class _FakeClient:
    def __init__(self, response: object) -> None:
        self.chat = _FakeChat(response)


class _ModelDumpMessage:
    def __init__(self, content: object) -> None:
        self._content = content

    def model_dump(self, *, mode: str, exclude_none: bool) -> dict[str, object]:
        del mode, exclude_none
        return {"content": self._content}


def _registry() -> ToolRegistry:
    registry = ToolRegistry()
    registry.register(EchoTool())
    return registry


def test_provider_runs_native_tool_calling_requests() -> None:
    fake_client = _FakeClient(
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "echo",
                        "arguments": '{"value": "hello"}',
                    },
                }
            ]
        }
    )
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=fake_client,
        default_request_params={"temperature": 0},
    )

    parsed = provider.run_native_tool_calling(
        adapter=NativeToolCallingAdapter(),
        messages=[{"role": "user", "content": "Say hello"}],
        registry=_registry(),
    )

    call = fake_client.chat.completions.calls[0]
    assert call["model"] == "demo-model"
    assert call["messages"] == [{"role": "user", "content": "Say hello"}]
    assert call["temperature"] == 0
    assert call["tools"][0]["function"]["name"] == "echo"
    assert parsed.invocations[0].tool_name == "echo"
    assert parsed.invocations[0].arguments == {"value": "hello"}


def test_provider_runs_structured_output_requests() -> None:
    fake_client = _FakeClient(
        '{"actions":[{"tool_name":"echo","arguments":{"value":"hi"}}]}'
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=fake_client)

    parsed = provider.run_structured_output(
        adapter=StructuredOutputAdapter(),
        messages=[{"role": "user", "content": "Use echo"}],
        registry=_registry(),
        request_params={"temperature": 0.2},
    )

    call = fake_client.chat.completions.calls[0]
    assert call["response_format"]["type"] == "json_schema"
    assert call["response_format"]["json_schema"]["schema"]["properties"]["actions"][
        "items"
    ]["properties"]["tool_name"]["enum"] == ["echo"]
    assert call["temperature"] == 0.2
    assert parsed.invocations[0].tool_name == "echo"


def test_provider_runs_prompt_schema_requests() -> None:
    fake_client = _FakeClient('{"final_response":"done"}')
    provider = OpenAICompatibleProvider(model="demo-model", client=fake_client)

    parsed = provider.run_prompt_schema(
        adapter=PromptSchemaAdapter(),
        messages=[{"role": "user", "content": "No tool"}],
        registry=_registry(),
    )

    call = fake_client.chat.completions.calls[0]
    assert call["messages"][0]["role"] == "system"
    assert "Choose exactly one" in call["messages"][0]["content"]
    assert call["messages"][1] == {"role": "user", "content": "No tool"}
    assert parsed.final_response == "done"


def test_provider_rejects_missing_choices() -> None:
    class NoChoicesClient:
        class Chat:
            class Completions:
                def create(self, **kwargs: Any) -> object:
                    del kwargs
                    return object()

            completions = Completions()

        chat = Chat()

    provider = OpenAICompatibleProvider(model="demo-model", client=NoChoicesClient())

    try:
        provider.run_native_tool_calling(
            adapter=NativeToolCallingAdapter(),
            messages=[{"role": "user", "content": "hello"}],
            registry=_registry(),
        )
    except ValueError as exc:
        assert "choices" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing choices.")


def test_provider_rejects_missing_message() -> None:
    provider = OpenAICompatibleProvider(model="demo-model", client=_FakeClient(None))

    try:
        provider.run_native_tool_calling(
            adapter=NativeToolCallingAdapter(),
            messages=[{"role": "user", "content": "hello"}],
            registry=_registry(),
        )
    except ValueError as exc:
        assert "message" in str(exc)
    else:
        raise AssertionError("Expected ValueError for missing message.")


def test_provider_extracts_content_from_supported_message_shapes() -> None:
    provider = OpenAICompatibleProvider(model="demo-model", client=_FakeClient({}))

    class ContentModel(BaseModel):
        content: object

    class AttributeMessage:
        def __init__(self, content: object) -> None:
            self.content = content

    assert (
        provider._extract_content(_FakeResponse(choices=[_FakeChoice(message="hello")]))
        == "hello"
    )
    assert (
        provider._extract_content(
            _FakeResponse(
                choices=[
                    _FakeChoice(message=ContentModel(content='{"final_response":"ok"}'))
                ]
            )
        )
        == '{"final_response":"ok"}'
    )
    assert (
        provider._extract_content(
            _FakeResponse(
                choices=[
                    _FakeChoice(
                        message=_ModelDumpMessage(
                            '{"actions":[{"tool_name":"echo","arguments":{"value":"hi"}}]}'
                        )
                    )
                ]
            )
        )
        == '{"actions":[{"tool_name":"echo","arguments":{"value":"hi"}}]}'
    )
    assert (
        provider._extract_content(
            _FakeResponse(
                choices=[_FakeChoice(message={"content": '{"final_response":"done"}'})]
            )
        )
        == '{"final_response":"done"}'
    )
    assert (
        provider._extract_content(
            _FakeResponse(
                choices=[
                    _FakeChoice(message=AttributeMessage('{"final_response":"attr"}'))
                ]
            )
        )
        == '{"final_response":"attr"}'
    )


def test_provider_presets_configure_openai_compatible_targets() -> None:
    ollama = OpenAICompatibleProvider.for_ollama(
        model="gemma4:26b", client=_FakeClient({})
    )
    openai = OpenAICompatibleProvider.for_openai(
        model="gpt-4.1-mini", client=_FakeClient({})
    )

    assert ollama.base_url == "http://localhost:11434/v1"
    assert ollama.model == "gemma4:26b"
    assert openai.base_url is None
    assert openai.model == "gpt-4.1-mini"


def test_provider_prefers_precomputed_tool_descriptions() -> None:
    fake_client = _FakeClient(
        {
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "echo", "arguments": '{"value":"x"}'},
                }
            ]
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=fake_client)
    precomputed_tools = [
        {
            "type": "function",
            "function": {
                "name": "echo",
                "description": "precomputed",
                "parameters": {"type": "object"},
            },
        }
    ]

    provider.run_native_tool_calling(
        adapter=NativeToolCallingAdapter(),
        messages=[{"role": "user", "content": "hello"}],
        tool_descriptions=precomputed_tools,
    )

    call = fake_client.chat.completions.calls[0]
    assert call["tools"] == precomputed_tools


def test_provider_rejects_missing_tool_source_for_export() -> None:
    provider = OpenAICompatibleProvider(model="demo-model", client=_FakeClient({}))

    with pytest.raises(ValueError):
        provider._resolve_tool_descriptions(
            adapter=NativeToolCallingAdapter(),
            registry=None,
            tool_descriptions=None,
        )
