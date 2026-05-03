"""Tests for the native Ollama provider."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import OllamaNativeProvider, ResponseModeStrategy
from llm_tools.llm_providers.ollama_native import StructuredOutputValidationError
from llm_tools.tool_api import ToolSpec


class EchoInput(BaseModel):
    value: str


class SimplePayload(BaseModel):
    status: str


def _response_model() -> tuple[ActionEnvelopeAdapter, type[BaseModel]]:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(
        [ToolSpec(name="echo", description="Echo a value.")],
        {"echo": EchoInput},
    )
    return adapter, response_model


def _chat_response(
    *,
    content: str | None = None,
    tool_calls: list[object] | None = None,
) -> object:
    return SimpleNamespace(
        message=SimpleNamespace(content=content, tool_calls=tool_calls)
    )


def _tool_call(name: str, arguments: dict[str, Any]) -> object:
    return SimpleNamespace(
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _SyncOllamaClient:
    def __init__(self, outcomes: list[object]) -> None:
        self.outcomes = list(outcomes)
        self.calls: list[dict[str, Any]] = []

    def chat(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def list(self) -> object:
        return SimpleNamespace(
            models=[
                SimpleNamespace(model="z-model"),
                SimpleNamespace(model="a-model"),
            ]
        )


class _AsyncOllamaClient:
    def __init__(self, outcomes: list[object]) -> None:
        self._sync = _SyncOllamaClient(outcomes)

    @property
    def calls(self) -> list[dict[str, Any]]:
        return self._sync.calls

    async def chat(self, **kwargs: Any) -> object:
        return self._sync.chat(**kwargs)


def test_native_tools_mode_maps_tool_calls_to_canonical_invocations() -> None:
    adapter, response_model = _response_model()
    client = _SyncOllamaClient(
        [
            _chat_response(
                tool_calls=[
                    _tool_call("echo", {"value": "hello"}),
                    _tool_call("echo", {"value": "again"}),
                ]
            )
        ]
    )
    provider = OllamaNativeProvider(
        model="llama3.2",
        client=client,
        response_mode_strategy=ResponseModeStrategy.TOOLS,
    )

    parsed = provider.run(
        adapter=adapter,
        messages=[{"role": "user", "content": "echo"}],
        response_model=response_model,
    )

    assert [item.tool_name for item in parsed.invocations] == ["echo", "echo"]
    assert parsed.invocations[0].arguments == {"value": "hello"}
    assert provider.last_mode_used is ResponseModeStrategy.TOOLS
    call = client.calls[0]
    assert call["model"] == "llama3.2"
    assert call["tools"][0]["function"]["name"] == "echo"
    assert (
        call["tools"][0]["function"]["parameters"]["properties"]["value"]["type"]
        == "string"
    )


def test_native_tools_mode_rejects_tool_calls_with_final_text() -> None:
    adapter, response_model = _response_model()
    provider = OllamaNativeProvider(
        model="llama3.2",
        client=_SyncOllamaClient(
            [
                _chat_response(
                    content="done",
                    tool_calls=[_tool_call("echo", {"value": "hello"})],
                )
            ]
        ),
        response_mode_strategy=ResponseModeStrategy.TOOLS,
    )

    with pytest.raises(StructuredOutputValidationError, match="both tool calls"):
        provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "echo"}],
            response_model=response_model,
        )


def test_json_mode_uses_ollama_format_schema() -> None:
    client = _SyncOllamaClient([_chat_response(content='{"status": "ok"}')])
    provider = OllamaNativeProvider(
        model="llama3.2",
        client=client,
        response_mode_strategy=ResponseModeStrategy.JSON,
    )

    payload = provider.run_structured(
        messages=[{"role": "user", "content": "probe"}],
        response_model=SimplePayload,
        request_params={"temperature": 0.2},
    )

    assert payload == SimplePayload(status="ok")
    assert provider.last_mode_used is ResponseModeStrategy.JSON
    call = client.calls[0]
    assert call["format"]["title"] == "SimplePayload"
    assert call["options"]["temperature"] == 0.2


def test_prompt_tools_mode_uses_plain_text_chat() -> None:
    client = _SyncOllamaClient([_chat_response(content="plain text")])
    provider = OllamaNativeProvider(
        model="llama3.2",
        client=client,
        response_mode_strategy=ResponseModeStrategy.PROMPT_TOOLS,
    )

    assert (
        provider.run_text(messages=[{"role": "user", "content": "hello"}])
        == "plain text"
    )
    assert provider.last_mode_used is ResponseModeStrategy.PROMPT_TOOLS
    assert "tools" not in client.calls[0]
    assert "format" not in client.calls[0]


def test_auto_mode_falls_back_from_tools_to_json() -> None:
    adapter, response_model = _response_model()
    client = _SyncOllamaClient(
        [
            RuntimeError("tools not supported"),
            _chat_response(
                content=(
                    '{"actions": [{"tool_name": "echo", "arguments": '
                    '{"value": "json"}}], "final_response": null}'
                )
            ),
        ]
    )
    provider = OllamaNativeProvider(model="llama3.2", client=client)

    parsed = provider.run(
        adapter=adapter,
        messages=[{"role": "user", "content": "echo"}],
        response_model=response_model,
    )

    assert parsed.invocations[0].arguments == {"value": "json"}
    assert provider.last_mode_used is ResponseModeStrategy.JSON
    assert "tools" in client.calls[0]
    assert "format" in client.calls[1]


def test_list_available_models_sorts_model_names() -> None:
    provider = OllamaNativeProvider(
        model="llama3.2",
        client=_SyncOllamaClient([]),
    )

    assert provider.list_available_models() == ["a-model", "z-model"]


def test_async_json_mode_uses_ollama_format_schema() -> None:
    client = _AsyncOllamaClient([_chat_response(content='{"status": "ok"}')])
    provider = OllamaNativeProvider(
        model="llama3.2",
        async_client=client,
        response_mode_strategy=ResponseModeStrategy.JSON,
    )

    payload = asyncio.run(
        provider.run_structured_async(
            messages=[{"role": "user", "content": "probe"}],
            response_model=SimplePayload,
        )
    )

    assert payload == SimplePayload(status="ok")
    assert "format" in client.calls[0]


def test_lazy_clients_use_host_and_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    created: list[tuple[str, str | None, float | None]] = []

    class _Client:
        def __init__(self, *, host: str | None, timeout: float | None) -> None:
            created.append(("sync", host, timeout))

    class _AsyncClient:
        def __init__(self, *, host: str | None, timeout: float | None) -> None:
            created.append(("async", host, timeout))

    monkeypatch.setattr("llm_tools.llm_providers.ollama_native.ollama.Client", _Client)
    monkeypatch.setattr(
        "llm_tools.llm_providers.ollama_native.ollama.AsyncClient", _AsyncClient
    )
    provider = OllamaNativeProvider(
        model="llama3.2",
        host="http://127.0.0.1:11434",
        default_request_params={"timeout": 9},
    )

    assert provider._sync_client is provider._sync_client
    assert provider._async_client_instance is provider._async_client_instance
    assert created == [
        ("sync", "http://127.0.0.1:11434", 9),
        ("async", "http://127.0.0.1:11434", 9),
    ]


def test_ollama_preflight_success_and_prompt_tools_mode() -> None:
    json_provider = OllamaNativeProvider(
        model="llama3.2",
        client=_SyncOllamaClient([_chat_response(content='{"status": "ok"}')]),
        response_mode_strategy=ResponseModeStrategy.JSON,
    )

    json_result = json_provider.preflight()

    assert json_result.ok is True
    assert json_result.available_models == ["a-model", "z-model"]
    assert json_result.resolved_mode is ResponseModeStrategy.JSON

    prompt_provider = OllamaNativeProvider(
        model="llama3.2",
        client=_SyncOllamaClient([_chat_response(content="PROMPT_TOOLS_OK")]),
        response_mode_strategy=ResponseModeStrategy.PROMPT_TOOLS,
    )

    prompt_result = prompt_provider.preflight()

    assert prompt_result.ok is True
    assert prompt_result.resolved_mode is ResponseModeStrategy.PROMPT_TOOLS


def test_ollama_preflight_reports_model_errors() -> None:
    class _ListingFailsClient(_SyncOllamaClient):
        def list(self) -> object:
            raise RuntimeError("connection refused")

    provider = OllamaNativeProvider(
        model="missing-model",
        client=_ListingFailsClient([RuntimeError("unknown model missing-model")]),
        response_mode_strategy=ResponseModeStrategy.JSON,
    )

    result = provider.preflight()

    assert result.ok is False
    assert result.connection_succeeded is True
    assert result.model_accepted is False
    assert result.resolved_mode is None
    assert "Check the configured model name" in result.actionable_message


def test_ollama_run_prompt_tools_and_auto_failure_paths() -> None:
    adapter, response_model = _response_model()
    prompt_provider = OllamaNativeProvider(
        model="llama3.2",
        client=_SyncOllamaClient([_chat_response(content="final")]),
        response_mode_strategy=ResponseModeStrategy.PROMPT_TOOLS,
    )
    parsed = prompt_provider.run(
        adapter=adapter,
        messages=[{"role": "user", "content": "hello"}],
        response_model=response_model,
    )
    assert parsed.final_response == "final"

    retry_provider = OllamaNativeProvider(
        model="llama3.2",
        client=_SyncOllamaClient(
            [
                RuntimeError("tools unsupported"),
                _chat_response(content="not-json"),
                _chat_response(content=None),
            ]
        ),
    )
    with pytest.raises(ValueError, match="All Ollama response mode attempts failed"):
        retry_provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )

    non_retry_provider = OllamaNativeProvider(
        model="llama3.2",
        client=_SyncOllamaClient([RuntimeError("quota exceeded")]),
    )
    with pytest.raises(RuntimeError, match="quota exceeded"):
        non_retry_provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )


def test_ollama_run_structured_prompt_tools_paths() -> None:
    sync_provider = OllamaNativeProvider(
        model="llama3.2",
        client=_SyncOllamaClient([_chat_response(content='{"status": "ok"}')]),
        response_mode_strategy=ResponseModeStrategy.PROMPT_TOOLS,
    )
    assert sync_provider.run_structured(
        messages=[{"role": "user", "content": "probe"}],
        response_model=SimplePayload,
    ) == SimplePayload(status="ok")

    async_provider = OllamaNativeProvider(
        model="llama3.2",
        async_client=_AsyncOllamaClient([_chat_response(content='{"status": "ok"}')]),
        response_mode_strategy=ResponseModeStrategy.PROMPT_TOOLS,
    )
    assert asyncio.run(
        async_provider.run_structured_async(
            messages=[{"role": "user", "content": "probe"}],
            response_model=SimplePayload,
        )
    ) == SimplePayload(status="ok")


def test_ollama_async_run_modes() -> None:
    adapter, response_model = _response_model()
    tools_provider = OllamaNativeProvider(
        model="llama3.2",
        async_client=_AsyncOllamaClient(
            [_chat_response(tool_calls=[_tool_call("echo", {"value": "async"})])]
        ),
        response_mode_strategy=ResponseModeStrategy.TOOLS,
    )
    tools_parsed = asyncio.run(
        tools_provider.run_async(
            adapter=adapter,
            messages=[{"role": "user", "content": "echo"}],
            response_model=response_model,
        )
    )
    assert tools_parsed.invocations[0].arguments == {"value": "async"}

    json_provider = OllamaNativeProvider(
        model="llama3.2",
        async_client=_AsyncOllamaClient(
            [
                _chat_response(
                    content=('{"actions": [], "final_response": "async json"}')
                )
            ]
        ),
        response_mode_strategy=ResponseModeStrategy.JSON,
    )
    json_parsed = asyncio.run(
        json_provider.run_async(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )
    )
    assert json_parsed.final_response == "async json"

    prompt_provider = OllamaNativeProvider(
        model="llama3.2",
        async_client=_AsyncOllamaClient([_chat_response(content="async final")]),
        response_mode_strategy=ResponseModeStrategy.PROMPT_TOOLS,
    )
    prompt_parsed = asyncio.run(
        prompt_provider.run_async(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )
    )
    assert prompt_parsed.final_response == "async final"


def test_ollama_response_parsing_accepts_mapping_payloads_and_rejects_bad_shapes() -> (
    None
):
    mapping_response = SimpleNamespace(message={"content": "mapped", "tool_calls": []})
    assert OllamaNativeProvider._response_text(mapping_response) == "mapped"
    assert OllamaNativeProvider._response_tool_calls(mapping_response) == []
    assert OllamaNativeProvider._response_tool_calls(
        SimpleNamespace(
            message={"tool_calls": [{"function": {"name": "echo", "arguments": {}}}]}
        )
    ) == [{"function": {"name": "echo", "arguments": {}}}]

    with pytest.raises(StructuredOutputValidationError, match="assistant text"):
        OllamaNativeProvider._response_text(SimpleNamespace(message={}))
    with pytest.raises(StructuredOutputValidationError, match="not a list"):
        OllamaNativeProvider._response_tool_calls(
            SimpleNamespace(message={"tool_calls": "bad"})
        )
    with pytest.raises(StructuredOutputValidationError, match="neither tool calls"):
        OllamaNativeProvider._tool_response_payload(_chat_response(content=""))
    with pytest.raises(StructuredOutputValidationError, match="function name"):
        OllamaNativeProvider._tool_response_payload(
            _chat_response(tool_calls=[{"function": {"arguments": {}}}])
        )
    with pytest.raises(StructuredOutputValidationError, match="arguments were not"):
        OllamaNativeProvider._tool_response_payload(
            _chat_response(
                tool_calls=[{"function": {"name": "echo", "arguments": "bad"}}]
            )
        )


def test_ollama_json_parsing_and_tool_schema_edges() -> None:
    with pytest.raises(
        StructuredOutputValidationError, match="valid JSON"
    ) as json_error:
        OllamaNativeProvider._parse_json_text("not-json", response_model=SimplePayload)
    assert json_error.value.invalid_payload == "not-json"
    with pytest.raises(StructuredOutputValidationError, match="requested schema"):
        OllamaNativeProvider._parse_json_text(
            '{"other": "value"}', response_model=SimplePayload
        )

    class NoDefs(BaseModel):
        value: str

    class BadDefs(BaseModel):
        value: str

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"$defs": ["not-a-mapping"]}

    class OddDefs(BaseModel):
        value: str

        @classmethod
        def model_json_schema(cls, *args: Any, **kwargs: Any) -> dict[str, Any]:
            return {
                "$defs": {
                    "NoProperties": {},
                    "MissingArguments": {"properties": {"tool_name": {"const": "x"}}},
                    "BlankName": {
                        "properties": {
                            "tool_name": {"const": " "},
                            "arguments": {"type": "object"},
                        }
                    },
                    "MissingRefTarget": {
                        "properties": {
                            "tool_name": {"const": "missing_ref"},
                            "arguments": {"$ref": "#/$defs/Missing"},
                        }
                    },
                    "ExternalRef": {
                        "properties": {
                            "tool_name": {"const": "external_ref"},
                            "arguments": {"$ref": "#/components/Args"},
                        }
                    },
                }
            }

    assert OllamaNativeProvider._ollama_tools_from_response_model(NoDefs) == []
    assert OllamaNativeProvider._ollama_tools_from_response_model(BadDefs) == []
    tools = OllamaNativeProvider._ollama_tools_from_response_model(OddDefs)
    assert [tool["function"]["name"] for tool in tools] == [
        "missing_ref",
        "external_ref",
    ]


def test_ollama_protocol_flags_and_preflight_helpers() -> None:
    auto_provider = OllamaNativeProvider(model="llama3.2", client=_SyncOllamaClient([]))
    json_provider = OllamaNativeProvider(
        model="llama3.2",
        client=_SyncOllamaClient([]),
        response_mode_strategy=ResponseModeStrategy.JSON,
    )
    prompt_provider = OllamaNativeProvider(
        model="llama3.2",
        client=_SyncOllamaClient([]),
        response_mode_strategy=ResponseModeStrategy.PROMPT_TOOLS,
    )

    assert auto_provider.prefers_simplified_json_schema_contract() is False
    assert json_provider.uses_staged_schema_protocol() is True
    assert prompt_provider.uses_prompt_tool_protocol() is True
    assert auto_provider.can_fallback_to_prompt_tools(
        StructuredOutputValidationError("schema")
    )
    assert asyncio.run(
        auto_provider.can_fallback_to_prompt_tools_async(
            StructuredOutputValidationError("schema")
        )
    )
    assert not prompt_provider.can_fallback_to_prompt_tools(
        StructuredOutputValidationError("schema")
    )
    assert auto_provider._should_retry_mode_failure(RuntimeError("schema parse failed"))
    assert not auto_provider._should_retry_mode_failure(RuntimeError("quota exceeded"))
    assert auto_provider._exception_summary(RuntimeError()) == "RuntimeError"
    assert auto_provider._looks_connected(RuntimeError("quota exceeded"))
    assert not auto_provider._looks_connected(RuntimeError("name resolution timeout"))
    assert "Unable to validate" in auto_provider._preflight_error_message(
        RuntimeError("quota exceeded")
    )


def test_ollama_prompt_tools_preflight_rejects_unexpected_text() -> None:
    provider = OllamaNativeProvider(
        model="llama3.2",
        client=_SyncOllamaClient([_chat_response(content="wrong")]),
    )

    with pytest.raises(StructuredOutputValidationError, match="expected text"):
        provider._run_prompt_tools_preflight(request_params=None)
