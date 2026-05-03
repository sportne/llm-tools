"""Tests for the native Ask Sage provider."""

from __future__ import annotations

import asyncio
import json
from typing import Any
from urllib.error import URLError
from urllib.request import Request

import pytest
from pydantic import BaseModel

from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import AskSageNativeProvider, ResponseModeStrategy
from llm_tools.llm_providers import ask_sage_native as ask_sage_module
from llm_tools.llm_providers.ask_sage_native import AskSageNativeProviderError
from llm_tools.tool_api import ToolSpec

ACCESS_VALUE = "-".join(("test", "credential"))


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


class _FakeAskSageTransport:
    def __init__(self, responses: list[object]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any], float | None]] = []

    def __call__(
        self, path: str, payload: dict[str, Any], timeout: float | None
    ) -> object:
        self.calls.append((path, payload, timeout))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def test_ask_sage_run_text_posts_native_query_payload() -> None:
    transport = _FakeAskSageTransport([{"response": "hello"}])
    provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        request_settings={"persona": 7, "dataset": ["alpha"], "limit_references": 3},
        default_request_params={"timeout": 12},
        post_json=transport,
    )

    text = provider.run_text(
        messages=[
            {"role": "system", "content": "Be concise."},
            {"role": "user", "content": "Hi"},
        ],
        request_params={"temperature": 0.3},
    )

    assert text == "hello"
    assert provider.last_mode_used is ResponseModeStrategy.PROMPT_TOOLS
    assert transport.calls == [
        (
            "/query",
            {
                "persona": 7,
                "dataset": ["alpha"],
                "limit_references": 3,
                "message": "system: Be concise.\n\nuser: Hi",
                "model": "gpt-4.1-mini",
                "temperature": 0.3,
            },
            12.0,
        )
    ]


def test_ask_sage_rejects_non_https_base_urls() -> None:
    with pytest.raises(AskSageNativeProviderError, match="HTTPS base URL"):
        AskSageNativeProvider(
            model="gpt-4.1-mini",
            access_token=ACCESS_VALUE,
            base_url="http://api.asksage.ai/server",
        )
    with pytest.raises(AskSageNativeProviderError, match="HTTPS base URL"):
        AskSageNativeProvider(
            model="gpt-4.1-mini",
            access_token=ACCESS_VALUE,
            base_url="not-a-url",
        )


def test_ask_sage_json_mode_parses_structured_payload() -> None:
    transport = _FakeAskSageTransport([{"answer": '{"status": "ok"}'}])
    provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        response_mode_strategy=ResponseModeStrategy.JSON,
        post_json=transport,
    )

    payload = provider.run_structured(
        messages=[{"role": "user", "content": "probe"}],
        response_model=SimplePayload,
    )

    assert payload == SimplePayload(status="ok")
    assert provider.last_mode_used is ResponseModeStrategy.JSON
    assert "Return a JSON object" in transport.calls[0][1]["message"]


def test_ask_sage_run_uses_json_action_envelope() -> None:
    adapter, response_model = _response_model()
    transport = _FakeAskSageTransport(
        [
            {
                "response": (
                    '{"actions": [{"tool_name": "echo", "arguments": '
                    '{"value": "hi"}}], "final_response": null}'
                )
            }
        ]
    )
    provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        response_mode_strategy=ResponseModeStrategy.JSON,
        post_json=transport,
    )

    parsed = provider.run(
        adapter=adapter,
        messages=[{"role": "user", "content": "echo"}],
        response_model=response_model,
    )

    assert parsed.invocations[0].tool_name == "echo"
    assert parsed.invocations[0].arguments == {"value": "hi"}


def test_ask_sage_tools_mode_is_explicitly_unsupported() -> None:
    provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        response_mode_strategy=ResponseModeStrategy.TOOLS,
        post_json=_FakeAskSageTransport([]),
    )

    with pytest.raises(AskSageNativeProviderError, match="native tools mode"):
        provider.run(
            adapter=ActionEnvelopeAdapter(),
            messages=[{"role": "user", "content": "echo"}],
            response_model=SimplePayload,
        )


def test_ask_sage_list_available_models_extracts_common_payload_shapes() -> None:
    provider = AskSageNativeProvider(
        model="discovery",
        access_token=ACCESS_VALUE,
        post_json=_FakeAskSageTransport(
            [{"models": [{"model": "z-model"}, {"name": "a-model"}]}]
        ),
    )

    assert provider.list_available_models() == ["a-model", "z-model"]


def test_ask_sage_async_text_uses_sync_transport_in_thread() -> None:
    provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        post_json=_FakeAskSageTransport([{"message": "async hello"}]),
    )

    assert (
        asyncio.run(
            provider.run_text_async(messages=[{"role": "user", "content": "Hi"}])
        )
        == "async hello"
    )


def test_ask_sage_preflight_success_reports_resolved_json_mode() -> None:
    provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        post_json=_FakeAskSageTransport(
            [
                {"models": [{"id": "gpt-4.1-mini"}]},
                {"response": '{"status": "ok"}'},
            ]
        ),
    )

    result = provider.preflight()

    assert result.ok is True
    assert result.connection_succeeded is True
    assert result.model_listing_supported is True
    assert result.available_models == ["gpt-4.1-mini"]
    assert result.resolved_mode is ResponseModeStrategy.JSON


def test_ask_sage_preflight_reports_prompt_tools_and_mode_errors() -> None:
    prompt_provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        response_mode_strategy=ResponseModeStrategy.PROMPT_TOOLS,
        post_json=_FakeAskSageTransport(
            [{"models": ["gpt-4.1-mini"]}, {"response": "PROMPT_TOOLS_OK"}]
        ),
    )
    prompt_result = prompt_provider.preflight()
    assert prompt_result.ok is True
    assert prompt_result.resolved_mode is ResponseModeStrategy.PROMPT_TOOLS

    tools_provider = AskSageNativeProvider(
        model="missing-model",
        access_token=ACCESS_VALUE,
        response_mode_strategy=ResponseModeStrategy.TOOLS,
        post_json=_FakeAskSageTransport([RuntimeError("connection refused")]),
    )
    tools_result = tools_provider.preflight()
    assert tools_result.ok is False
    assert tools_result.connection_succeeded is True
    assert tools_result.model_accepted is True
    assert "native tools mode" in tools_result.actionable_message

    model_error_provider = AskSageNativeProvider(
        model="missing-model",
        access_token=ACCESS_VALUE,
        response_mode_strategy=ResponseModeStrategy.JSON,
        post_json=_FakeAskSageTransport(
            [{"models": []}, RuntimeError("unknown model missing-model")]
        ),
    )
    model_error_result = model_error_provider.preflight()
    assert model_error_result.model_accepted is False
    assert "Check the configured model name" in model_error_result.actionable_message


def test_ask_sage_auto_falls_back_from_json_to_prompt_text() -> None:
    adapter, response_model = _response_model()
    provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        post_json=_FakeAskSageTransport(
            [{"response": "not-json"}, {"response": "plain answer"}]
        ),
    )

    parsed = provider.run(
        adapter=adapter,
        messages=[{"role": "user", "content": "hello"}],
        response_model=response_model,
    )

    assert parsed.final_response == "plain answer"
    assert provider.last_mode_used is ResponseModeStrategy.PROMPT_TOOLS


def test_ask_sage_auto_reports_all_retryable_mode_failures() -> None:
    adapter, response_model = _response_model()
    provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        post_json=_FakeAskSageTransport(
            [
                {"response": "not-json"},
                {"unexpected": "no assistant text"},
            ]
        ),
    )

    with pytest.raises(ValueError, match="All Ask Sage response mode attempts failed"):
        provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )


def test_ask_sage_non_retryable_auto_error_is_raised() -> None:
    adapter, response_model = _response_model()
    provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        post_json=_FakeAskSageTransport([RuntimeError("quota exceeded")]),
    )

    with pytest.raises(RuntimeError, match="quota exceeded"):
        provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )


def test_ask_sage_structured_validation_errors_include_payload() -> None:
    invalid_json_provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        post_json=_FakeAskSageTransport([{"response": "not-json"}]),
    )
    with pytest.raises(AskSageNativeProviderError, match="valid JSON") as json_error:
        invalid_json_provider.run_structured(
            messages=[{"role": "user", "content": "probe"}],
            response_model=SimplePayload,
        )
    assert json_error.value.invalid_payload == "not-json"

    invalid_schema_provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        post_json=_FakeAskSageTransport([{"response": '{"other": "value"}'}]),
    )
    with pytest.raises(
        AskSageNativeProviderError, match="requested schema"
    ) as schema_error:
        invalid_schema_provider.run_structured(
            messages=[{"role": "user", "content": "probe"}],
            response_model=SimplePayload,
        )
    assert schema_error.value.invalid_payload == {"other": "value"}


def test_ask_sage_async_structured_and_run_paths() -> None:
    adapter, response_model = _response_model()
    run_provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        response_mode_strategy=ResponseModeStrategy.PROMPT_TOOLS,
        post_json=_FakeAskSageTransport([{"text": "async final"}]),
    )
    parsed = asyncio.run(
        run_provider.run_async(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )
    )
    assert parsed.final_response == "async final"

    structured_provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        response_mode_strategy=ResponseModeStrategy.JSON,
        post_json=_FakeAskSageTransport([{"output": '{"status": "ok"}'}]),
    )
    payload = asyncio.run(
        structured_provider.run_structured_async(
            messages=[{"role": "user", "content": "probe"}],
            response_model=SimplePayload,
        )
    )
    assert payload == SimplePayload(status="ok")


def test_ask_sage_query_payload_defaults_and_filters_request_params() -> None:
    transport = _FakeAskSageTransport([{"result": "hello"}])
    provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        default_request_params={"live": True, "unused": "ignored"},
        post_json=transport,
    )

    provider.run_text(
        messages=[
            {"role": "", "content": ""},
            {"role": "user", "content": "Hi"},
        ],
        request_params={"limit_references": 2},
    )

    assert transport.calls[0][1] == {
        "persona": 1,
        "dataset": "none",
        "message": "user: Hi",
        "model": "gpt-4.1-mini",
        "limit_references": 2,
        "live": True,
    }


def test_ask_sage_extractors_accept_common_shapes_and_reject_empty_text() -> None:
    assert AskSageNativeProvider._extract_model_names(["a", " ", {"id": "b"}]) == [
        "a",
        "b",
    ]
    assert AskSageNativeProvider._extract_model_names({"data": ["c"]}) == ["c"]
    assert AskSageNativeProvider._extract_model_names({"result": {"name": "d"}}) == [
        "d"
    ]
    assert AskSageNativeProvider._extract_model_names("invalid") == []
    assert AskSageNativeProvider._extract_response_text(" final ") == " final "
    with pytest.raises(AskSageNativeProviderError, match="assistant text"):
        AskSageNativeProvider._extract_response_text({"response": ""})


def test_ask_sage_protocol_flags_and_retry_checks() -> None:
    auto_provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        post_json=_FakeAskSageTransport([]),
    )
    json_provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        response_mode_strategy=ResponseModeStrategy.JSON,
        post_json=_FakeAskSageTransport([]),
    )
    prompt_provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        response_mode_strategy=ResponseModeStrategy.PROMPT_TOOLS,
        post_json=_FakeAskSageTransport([]),
    )

    assert auto_provider.prefers_simplified_json_schema_contract() is False
    assert json_provider.uses_staged_schema_protocol() is True
    assert prompt_provider.uses_prompt_tool_protocol() is True
    assert auto_provider.can_fallback_to_prompt_tools(
        AskSageNativeProviderError("schema failed")
    )
    assert asyncio.run(
        auto_provider.can_fallback_to_prompt_tools_async(
            AskSageNativeProviderError("schema failed")
        )
    )
    assert not prompt_provider.can_fallback_to_prompt_tools(
        AskSageNativeProviderError("schema failed")
    )
    assert auto_provider._should_retry_mode_failure(RuntimeError("json parse failed"))
    assert not auto_provider._should_retry_mode_failure(RuntimeError("quota exceeded"))
    assert auto_provider._exception_summary(RuntimeError()) == "RuntimeError"
    assert auto_provider._looks_connected(RuntimeError("quota exceeded"))
    assert not auto_provider._looks_connected(RuntimeError("DNS timeout"))


def test_ask_sage_post_json_uses_native_headers_and_wraps_transport_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps({"response": "ok"}).encode("utf-8")

    def _fake_urlopen(request: Request, timeout: float) -> _Response:
        captured["url"] = request.full_url
        captured["data"] = request.data
        captured["headers"] = {
            key.lower(): value for key, value in request.header_items()
        }
        captured["timeout"] = timeout
        return _Response()

    monkeypatch.setattr(ask_sage_module, "urlopen", _fake_urlopen)
    provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        base_url="https://example.invalid/server/",
    )

    assert provider._post_json("/query", {"message": "hi"}, None) == {"response": "ok"}
    assert captured["url"] == "https://example.invalid/server/query"
    assert captured["data"] == b'{"message": "hi"}'
    assert captured["headers"]["X-access-tokens".lower()] == ACCESS_VALUE
    assert captured["timeout"] == 60.0

    def _failing_urlopen(request: Request, timeout: float) -> object:
        raise URLError("name resolution failed")

    monkeypatch.setattr(ask_sage_module, "urlopen", _failing_urlopen)
    with pytest.raises(AskSageNativeProviderError, match="request failed"):
        provider._post_json("/query", {}, 5.0)


def test_ask_sage_prompt_tools_preflight_rejects_unexpected_text() -> None:
    provider = AskSageNativeProvider(
        model="gpt-4.1-mini",
        access_token=ACCESS_VALUE,
        post_json=_FakeAskSageTransport([{"response": "wrong"}]),
    )

    with pytest.raises(AskSageNativeProviderError, match="expected text"):
        provider._run_prompt_tools_preflight(request_params=None)
