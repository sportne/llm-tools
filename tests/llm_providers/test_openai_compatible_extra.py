"""Additional coverage tests for the OpenAI-compatible provider."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any

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
    assert "tools: RuntimeError" in message
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
