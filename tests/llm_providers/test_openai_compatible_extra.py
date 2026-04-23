"""Additional coverage tests for the OpenAI-compatible provider."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any

from pydantic import BaseModel

import llm_tools.llm_providers.openai_compatible as provider_module
from llm_tools.llm_providers import OpenAICompatibleProvider, ProviderModeStrategy


class _FakeMode(str, Enum):  # noqa: UP042
    TOOLS = "TOOLS"
    JSON = "JSON"
    MD_JSON = "MD_JSON"


@dataclass
class _BareClient:
    chat: Any
    models: Any | None = None


class _RealNamedInstructor:
    __name__ = "instructor"
    Mode = _FakeMode

    @staticmethod
    def from_openai(client: Any, *, mode: _FakeMode) -> Any:
        del client, mode
        raise AssertionError("real instructor should be bypassed for fake clients")


class _WrappedInstructor:
    __name__ = "wrapped"
    Mode = _FakeMode

    @staticmethod
    def from_openai(client: Any, *, mode: _FakeMode) -> Any:
        del mode
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))


class InstructorSchemaValidationError(RuntimeError):
    pass


InstructorSchemaValidationError.__module__ = "instructor.testing"


def _wrapped_schema_failure(message: str) -> RuntimeError:
    wrapped = RuntimeError(message)
    wrapped.__cause__ = InstructorSchemaValidationError(message)
    return wrapped


class InstructorMessageCarrierError(RuntimeError):
    pass


InstructorMessageCarrierError.__module__ = "instructor.testing"


class OpenAITransportError(RuntimeError):
    pass


OpenAITransportError.__module__ = "openai.testing"


class PydanticTransportError(RuntimeError):
    pass


PydanticTransportError.__module__ = "pydantic.testing"


class _NativeSyncCompletions:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self._response


class _NativeAsyncCompletions:
    def __init__(self, response: Any) -> None:
        self._response = response
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        return self._response


def _native_response(json_payload: str) -> Any:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=json_payload),
            )
        ]
    )


class _ProbeModel(BaseModel):
    status: str
    count: int
    items: list[str]


def test_provider_helper_methods_cover_listing_and_parameter_merging() -> None:
    data_page = SimpleNamespace(
        data=[
            SimpleNamespace(id=" one "),
            SimpleNamespace(id=""),
            SimpleNamespace(id=None),
            SimpleNamespace(id="two"),
            SimpleNamespace(id="one"),
        ]
    )
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=_BareClient(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: kwargs)
            ),
            models=SimpleNamespace(list=lambda: data_page),
        ),
        default_request_params={"temperature": 0.2, "top_p": 1.0},
    )
    assert provider.list_available_models() == ["one", "two"]

    iterable_page = [
        SimpleNamespace(id=" zed "),
        SimpleNamespace(id=""),
        SimpleNamespace(id="alpha"),
    ]
    provider._client = _BareClient(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: kwargs)
        ),
        models=SimpleNamespace(list=lambda: iterable_page),
    )
    assert provider.list_available_models() == ["alpha", "zed"]
    assert provider._candidate_modes() == [
        ProviderModeStrategy.TOOLS,
        ProviderModeStrategy.JSON,
        ProviderModeStrategy.MD_JSON,
    ]
    assert OpenAICompatibleProvider.for_ollama(
        model="demo-model",
        mode_strategy=ProviderModeStrategy.AUTO,
    )._candidate_modes() == [
        ProviderModeStrategy.JSON,
        ProviderModeStrategy.MD_JSON,
        ProviderModeStrategy.TOOLS,
    ]
    assert OpenAICompatibleProvider.for_ollama(
        model="demo-model",
        mode_strategy=ProviderModeStrategy.AUTO,
    ).uses_staged_schema_protocol()
    assert not OpenAICompatibleProvider.for_openai(
        model="demo-model",
        mode_strategy=ProviderModeStrategy.AUTO,
    ).uses_staged_schema_protocol()
    assert OpenAICompatibleProvider(
        model="demo-model",
        mode_strategy=ProviderModeStrategy.JSON,
    )._candidate_modes() == [ProviderModeStrategy.JSON]
    assert provider._merged_request_params(None) == {
        "temperature": 0.2,
        "top_p": 1.0,
    }
    assert provider._merged_request_params({"temperature": 0.7}) == {
        "temperature": 0.7,
        "top_p": 1.0,
    }
    message = provider._fallback_error_message(
        [(ProviderModeStrategy.TOOLS, RuntimeError("boom"))]
    )
    assert "Overall failure type: transport-related" in message
    assert "tools: transport-related (RuntimeError: boom)" in message
    assert OpenAICompatibleProvider._failure_category(RuntimeError("boom")) == (
        "transport-related"
    )
    assert OpenAICompatibleProvider._is_real_instructor_module(_RealNamedInstructor())
    assert not OpenAICompatibleProvider._is_real_instructor_module(
        SimpleNamespace(__name__="other")
    )
    assert not OpenAICompatibleProvider._is_compatible_chat_client(SimpleNamespace())
    assert not OpenAICompatibleProvider._is_compatible_chat_client(
        SimpleNamespace(chat=SimpleNamespace())
    )
    assert OpenAICompatibleProvider._is_compatible_chat_client(
        SimpleNamespace(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: kwargs)
            )
        )
    )
    assert OpenAICompatibleProvider._mode_attempts_are_distinct(None, object())
    previous_client = object()
    assert not OpenAICompatibleProvider._mode_attempts_are_distinct(
        previous_client, previous_client
    )
    assert OpenAICompatibleProvider._should_retry_mode_failure(
        InstructorSchemaValidationError("schema validation failed")
    )
    assert OpenAICompatibleProvider._should_retry_mode_failure(
        _wrapped_schema_failure("schema validation failed")
    )
    wrapped_runtime = RuntimeError("boom")
    assert not OpenAICompatibleProvider._should_retry_mode_failure(wrapped_runtime)
    chain = OpenAICompatibleProvider._iter_exception_chain(
        _wrapped_schema_failure("schema validation failed")
    )
    assert [type(exc).__name__ for exc in chain] == [
        "RuntimeError",
        "InstructorSchemaValidationError",
    ]


def test_ollama_json_mode_uses_native_json_schema_payload(monkeypatch: Any) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _RealNamedInstructor())
    completions = _NativeSyncCompletions(
        _native_response('{"status":"ok","count":3,"items":["alpha","beta","gamma"]}')
    )
    provider = OpenAICompatibleProvider.for_ollama(
        model="gemma4:e4b",
        client=_BareClient(
            chat=SimpleNamespace(completions=completions),
            models=SimpleNamespace(list=lambda: [SimpleNamespace(id="gemma4:e4b")]),
        ),
        mode_strategy=ProviderModeStrategy.JSON,
    )

    payload = provider.run_structured(
        messages=[{"role": "user", "content": "Return structured data."}],
        response_model=_ProbeModel,
        request_params={"temperature": 0},
    )

    assert payload.model_dump() == {
        "status": "ok",
        "count": 3,
        "items": ["alpha", "beta", "gamma"],
    }
    assert len(completions.calls) == 1
    call = completions.calls[0]
    assert call["model"] == "gemma4:e4b"
    assert call["temperature"] == 0
    assert call["response_format"] == {
        "type": "json_schema",
        "json_schema": {
            "name": "_ProbeModel",
            "strict": True,
            "schema": _ProbeModel.model_json_schema(),
        },
    }
    assert "response_model" not in call


def test_ollama_json_mode_uses_native_json_schema_payload_async(monkeypatch: Any) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _RealNamedInstructor())
    completions = _NativeAsyncCompletions(
        _native_response('{"status":"ok","count":3,"items":["alpha","beta","gamma"]}')
    )
    provider = OpenAICompatibleProvider.for_ollama(
        model="gemma4:e4b",
        async_client=_BareClient(chat=SimpleNamespace(completions=completions)),
        mode_strategy=ProviderModeStrategy.JSON,
    )

    payload = asyncio.run(
        provider.run_structured_async(
            messages=[{"role": "user", "content": "Return structured data."}],
            response_model=_ProbeModel,
            request_params={"temperature": 0},
        )
    )

    assert payload.model_dump() == {
        "status": "ok",
        "count": 3,
        "items": ["alpha", "beta", "gamma"],
    }
    assert len(completions.calls) == 1
    assert completions.calls[0]["response_format"]["type"] == "json_schema"


def test_md_json_mode_extracts_json_from_markdown_content() -> None:
    completions = _NativeSyncCompletions(
        _native_response(
            "I will use the strict schema.\n\n```json\n"
            '{"status":"ok","count":3,"items":["alpha","beta","gamma"]}\n'
            "```"
        )
    )
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=_BareClient(chat=SimpleNamespace(completions=completions)),
        mode_strategy=ProviderModeStrategy.MD_JSON,
    )

    payload = provider.run_structured(
        messages=[{"role": "user", "content": "Return structured data."}],
        response_model=_ProbeModel,
        request_params={"temperature": 0},
    )

    assert payload.model_dump() == {
        "status": "ok",
        "count": 3,
        "items": ["alpha", "beta", "gamma"],
    }
    assert len(completions.calls) == 1
    assert "response_model" not in completions.calls[0]


def test_md_json_mode_extracts_json_from_markdown_content_async() -> None:
    completions = _NativeAsyncCompletions(
        _native_response(
            "Response follows.\n\n"
            '{"status":"ok","count":3,"items":["alpha","beta","gamma"]}'
        )
    )
    provider = OpenAICompatibleProvider(
        model="demo-model",
        async_client=_BareClient(chat=SimpleNamespace(completions=completions)),
        mode_strategy=ProviderModeStrategy.MD_JSON,
    )

    payload = asyncio.run(
        provider.run_structured_async(
            messages=[{"role": "user", "content": "Return structured data."}],
            response_model=_ProbeModel,
            request_params={"temperature": 0},
        )
    )

    assert payload.model_dump() == {
        "status": "ok",
        "count": 3,
        "items": ["alpha", "beta", "gamma"],
    }
    assert len(completions.calls) == 1


def test_provider_helper_methods_cover_retry_classification_edges() -> None:
    assert OpenAICompatibleProvider._should_retry_mode_failure(
        json.JSONDecodeError("bad json", "{}", 0)
    )
    assert OpenAICompatibleProvider._should_retry_mode_failure(
        PydanticTransportError("pydantic validation failed")
    )
    assert not OpenAICompatibleProvider._should_retry_mode_failure(
        OpenAITransportError("schema validation failed")
    )

    assert OpenAICompatibleProvider._should_retry_mode_failure(
        InstructorMessageCarrierError("schema validation failed")
    )
    assert not OpenAICompatibleProvider._should_retry_mode_failure(
        InstructorMessageCarrierError("plain failure")
    )


def test_provider_helper_methods_cover_exception_chain_context_and_cycles() -> None:
    root = InstructorSchemaValidationError("schema validation failed")
    wrapped = RuntimeError("outer")
    wrapped.__context__ = root
    assert OpenAICompatibleProvider._should_retry_mode_failure(wrapped)

    cycle = RuntimeError("cycle")
    cycle.__cause__ = cycle
    chain = OpenAICompatibleProvider._iter_exception_chain(cycle)
    assert [type(exc).__name__ for exc in chain] == ["RuntimeError"]


def test_provider_preflight_reports_success_for_selected_mode() -> None:
    create_calls: list[dict[str, object]] = []
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=_BareClient(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (
                        create_calls.append(kwargs) or {"status": "ok"}
                    )
                )
            ),
            models=SimpleNamespace(list=lambda: [SimpleNamespace(id="demo-model")]),
        ),
        mode_strategy=ProviderModeStrategy.JSON,
    )

    report = provider.preflight(request_params={"temperature": 0.0})

    assert report.ok is True
    assert report.connection_succeeded is True
    assert report.model_accepted is True
    assert report.selected_mode_supported is True
    assert report.model_listing_supported is True
    assert report.available_models == ["demo-model"]
    assert report.resolved_mode is ProviderModeStrategy.JSON
    assert create_calls[0]["model"] == "demo-model"
    assert create_calls[0]["temperature"] == 0.0


def test_provider_preflight_reports_mode_failure_for_selected_mode() -> None:
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=_BareClient(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(
                        InstructorMessageCarrierError("schema validation failed")
                    )
                )
            ),
            models=SimpleNamespace(list=lambda: [SimpleNamespace(id="demo-model")]),
        ),
        mode_strategy=ProviderModeStrategy.JSON,
    )

    report = provider.preflight()

    assert report.ok is False
    assert report.connection_succeeded is True
    assert report.model_accepted is True
    assert report.selected_mode_supported is False
    assert report.model_listing_supported is True
    assert "provider mode 'json'" in report.actionable_message


def test_provider_preflight_reports_transport_failures_without_blaming_mode() -> None:
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=_BareClient(
            chat=SimpleNamespace(
                completions=SimpleNamespace(
                    create=lambda **kwargs: (_ for _ in ()).throw(
                        RuntimeError("401 unauthorized")
                    )
                )
            ),
            models=SimpleNamespace(list=lambda: [SimpleNamespace(id="demo-model")]),
        ),
        mode_strategy=ProviderModeStrategy.JSON,
    )

    report = provider.preflight()

    assert report.ok is False
    assert report.selected_mode_supported is False
    assert report.actionable_message == (
        "Unable to validate this provider configuration. RuntimeError: 401 unauthorized"
    )


def test_provider_preflight_handles_missing_model_listing_support() -> None:
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=_BareClient(
            chat=SimpleNamespace(
                completions=SimpleNamespace(create=lambda **kwargs: {"status": "ok"})
            ),
            models=SimpleNamespace(
                list=lambda: (_ for _ in ()).throw(RuntimeError("listing disabled"))
            ),
        ),
    )

    report = provider.preflight()

    assert report.ok is True
    assert report.connection_succeeded is True
    assert report.model_listing_supported is False
    assert report.available_models == []
    assert report.error_message == "RuntimeError: listing disabled"


def test_provider_client_factory_and_instructor_wrapping_helpers(
    monkeypatch: Any,
) -> None:
    fake_sync_client = _BareClient(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: kwargs)
        )
    )
    fake_async_client = _BareClient(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=lambda **kwargs: kwargs)
        )
    )

    monkeypatch.setattr(provider_module, "_instructor", None)
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=fake_sync_client,
        async_client=fake_async_client,
    )
    assert provider._require_instructor() is provider_module._INSTRUCTOR_FALLBACK

    monkeypatch.setattr(provider_module, "_instructor", _RealNamedInstructor())
    assert (
        provider._instructor_sync_client(ProviderModeStrategy.TOOLS) is fake_sync_client
    )
    assert (
        provider._instructor_async_client(ProviderModeStrategy.TOOLS)
        is fake_async_client
    )

    monkeypatch.setattr(provider_module, "_instructor", _WrappedInstructor())
    wrapped_provider = OpenAICompatibleProvider(
        model="demo-model",
        client=fake_sync_client,
        async_client=fake_async_client,
    )
    assert (
        wrapped_provider._instructor_sync_client(ProviderModeStrategy.JSON)
        is fake_sync_client
    )
    assert (
        wrapped_provider._instructor_async_client(ProviderModeStrategy.JSON)
        is fake_async_client
    )

    sync_calls: list[dict[str, object]] = []
    async_calls: list[dict[str, object]] = []

    class _FakeOpenAI:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            sync_calls.append({"api_key": api_key, "base_url": base_url})

    class _FakeAsyncOpenAI:
        def __init__(self, *, api_key: str | None, base_url: str | None) -> None:
            async_calls.append({"api_key": api_key, "base_url": base_url})

    monkeypatch.setattr(provider_module, "OpenAI", _FakeOpenAI)
    monkeypatch.setattr(provider_module, "AsyncOpenAI", _FakeAsyncOpenAI)
    constructed = OpenAICompatibleProvider(
        model="demo-model",
        api_key="secret",
        base_url="http://endpoint.test/v1",
    )
    assert isinstance(constructed._sync_client, _FakeOpenAI)
    assert isinstance(constructed._async_client_instance, _FakeAsyncOpenAI)
    assert sync_calls == [{"api_key": "secret", "base_url": "http://endpoint.test/v1"}]
    assert async_calls == [{"api_key": "secret", "base_url": "http://endpoint.test/v1"}]
