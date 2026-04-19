"""Tests for the Instructor-backed OpenAI-compatible provider."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from typing import Any

import pytest
from pydantic import BaseModel

import llm_tools.llm_providers.openai_compatible as provider_module
from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider, ProviderModeStrategy
from llm_tools.tool_api import ToolSpec


class EchoInput(BaseModel):
    value: str


def _response_model() -> tuple[ActionEnvelopeAdapter, type[BaseModel]]:
    adapter = ActionEnvelopeAdapter()
    response_model = adapter.build_response_model(
        [ToolSpec(name="echo", description="Echo a value.")],
        {"echo": EchoInput},
    )
    return adapter, response_model


class _SyncCompletions:
    def __init__(self, outcomes: dict[str, object]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        mode = kwargs.get("__mode", "TOOLS")
        outcome = self._outcomes[mode]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _AsyncCompletions:
    def __init__(self, outcomes: dict[str, object]) -> None:
        self._outcomes = outcomes
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        mode = kwargs.get("__mode", "TOOLS")
        outcome = self._outcomes[mode]
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@dataclass
class _SyncClient:
    outcomes: dict[str, object]

    def __post_init__(self) -> None:
        self.chat = type(
            "_SyncChat",
            (),
            {"completions": _SyncCompletions(self.outcomes)},
        )()


@dataclass
class _AsyncClient:
    outcomes: dict[str, object]

    def __post_init__(self) -> None:
        self.chat = type(
            "_AsyncChat",
            (),
            {"completions": _AsyncCompletions(self.outcomes)},
        )()


class _FakeMode(str, Enum):  # noqa: UP042
    TOOLS = "TOOLS"
    JSON = "JSON"
    MD_JSON = "MD_JSON"


class _PatchedCompletions:
    def __init__(self, completions: _SyncCompletions, *, mode: _FakeMode) -> None:
        self._completions = completions
        self._mode = mode

    def create(self, **kwargs: Any) -> object:
        return self._completions.create(__mode=self._mode.value, **kwargs)


class _PatchedAsyncCompletions:
    def __init__(self, completions: _AsyncCompletions, *, mode: _FakeMode) -> None:
        self._completions = completions
        self._mode = mode

    async def create(self, **kwargs: Any) -> object:
        return await self._completions.create(__mode=self._mode.value, **kwargs)


class _FakeInstructor:
    Mode = _FakeMode

    @staticmethod
    def from_openai(client: Any, *, mode: _FakeMode) -> Any:
        completions = client.chat.completions
        if isinstance(completions, _AsyncCompletions):
            return type(
                "_PatchedAsyncClient",
                (),
                {
                    "chat": type(
                        "_PatchedAsyncChat",
                        (),
                        {
                            "completions": _PatchedAsyncCompletions(
                                completions, mode=mode
                            )
                        },
                    )()
                },
            )()
        return type(
            "_PatchedSyncClient",
            (),
            {
                "chat": type(
                    "_PatchedSyncChat",
                    (),
                    {"completions": _PatchedCompletions(completions, mode=mode)},
                )()
            },
        )()


class _StrictInstructorModule:
    __name__ = "instructor"
    Mode = _FakeMode

    @staticmethod
    def from_openai(client: Any, *, mode: _FakeMode) -> Any:
        del client, mode
        raise AssertionError("from_openai should be bypassed for fake clients")


class InstructorSchemaValidationError(RuntimeError):
    pass


InstructorSchemaValidationError.__module__ = "instructor.testing"


def _wrapped_schema_failure(message: str) -> RuntimeError:
    wrapped = RuntimeError(message)
    wrapped.__cause__ = InstructorSchemaValidationError(message)
    return wrapped


def test_provider_run_prefers_tools_mode_when_available(monkeypatch: Any) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    adapter, response_model = _response_model()
    client = _SyncClient(
        outcomes={
            "TOOLS": {
                "actions": [{"tool_name": "echo", "arguments": {"value": "hello"}}],
                "final_response": None,
            },
            "JSON": RuntimeError("should-not-run"),
            "MD_JSON": RuntimeError("should-not-run"),
        }
    )
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=client,
        default_request_params={"temperature": 0},
    )

    parsed = provider.run(
        adapter=adapter,
        messages=[{"role": "user", "content": "hello"}],
        response_model=response_model,
    )

    assert parsed.invocations[0].tool_name == "echo"
    assert parsed.invocations[0].arguments == {"value": "hello"}
    assert provider.last_mode_used is ProviderModeStrategy.TOOLS
    call = client.chat.completions.calls[0]
    assert call["model"] == "demo-model"
    assert call["temperature"] == 0


def test_provider_run_auto_falls_back_to_json_for_retryable_schema_errors(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    adapter, response_model = _response_model()
    client = _SyncClient(
        outcomes={
            "TOOLS": _wrapped_schema_failure("schema validation failed"),
            "JSON": {"actions": [], "final_response": "done"},
            "MD_JSON": RuntimeError("should-not-run"),
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=client)

    parsed = provider.run(
        adapter=adapter,
        messages=[{"role": "user", "content": "hello"}],
        response_model=response_model,
    )

    assert parsed.final_response == "done"
    assert provider.last_mode_used is ProviderModeStrategy.JSON
    assert [call["__mode"] for call in client.chat.completions.calls] == [
        "TOOLS",
        "JSON",
    ]


def test_provider_run_auto_retries_wrapped_validation_failure(monkeypatch: Any) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    adapter, response_model = _response_model()
    client = _SyncClient(
        outcomes={
            "TOOLS": _wrapped_schema_failure("schema validation failed"),
            "JSON": {"actions": [], "final_response": "wrapped"},
            "MD_JSON": RuntimeError("should-not-run"),
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=client)

    parsed = provider.run(
        adapter=adapter,
        messages=[{"role": "user", "content": "hello"}],
        response_model=response_model,
    )

    assert parsed.final_response == "wrapped"
    assert [call["__mode"] for call in client.chat.completions.calls] == [
        "TOOLS",
        "JSON",
    ]


def test_provider_run_auto_does_not_retry_generic_failure(monkeypatch: Any) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    adapter, response_model = _response_model()
    client = _SyncClient(
        outcomes={
            "TOOLS": RuntimeError("transport failed"),
            "JSON": {"actions": [], "final_response": "should-not-run"},
            "MD_JSON": RuntimeError("should-not-run"),
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=client)

    with pytest.raises(RuntimeError, match="transport failed"):
        provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )

    assert [call["__mode"] for call in client.chat.completions.calls] == ["TOOLS"]


def test_provider_run_honors_explicit_mode_strategy(monkeypatch: Any) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    adapter, response_model = _response_model()
    client = _SyncClient(
        outcomes={
            "TOOLS": RuntimeError("unused"),
            "JSON": RuntimeError("unused"),
            "MD_JSON": {"actions": [], "final_response": "md-json"},
        }
    )
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=client,
        mode_strategy=ProviderModeStrategy.MD_JSON,
    )

    parsed = provider.run(
        adapter=adapter,
        messages=[{"role": "user", "content": "hello"}],
        response_model=response_model,
    )

    assert parsed.final_response == "md-json"
    assert provider.last_mode_used is ProviderModeStrategy.MD_JSON
    assert [call["__mode"] for call in client.chat.completions.calls] == ["MD_JSON"]


def test_provider_run_explicit_mode_raises_original_exception(monkeypatch: Any) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    adapter, response_model = _response_model()
    client = _SyncClient(
        outcomes={
            "TOOLS": RuntimeError("unused"),
            "JSON": RuntimeError("unused"),
            "MD_JSON": RuntimeError("md-json failed"),
        }
    )
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=client,
        mode_strategy=ProviderModeStrategy.MD_JSON,
    )

    with pytest.raises(RuntimeError, match="md-json failed"):
        provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )

    assert [call["__mode"] for call in client.chat.completions.calls] == ["MD_JSON"]


def test_provider_run_async_uses_fallback_order_for_retryable_schema_errors(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    adapter, response_model = _response_model()
    client = _AsyncClient(
        outcomes={
            "TOOLS": _wrapped_schema_failure("schema validation failed"),
            "JSON": _wrapped_schema_failure("json validation failed"),
            "MD_JSON": {
                "actions": [{"tool_name": "echo", "arguments": {"value": "async"}}],
                "final_response": None,
            },
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", async_client=client)

    parsed = asyncio.run(
        provider.run_async(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )
    )

    assert parsed.invocations[0].arguments == {"value": "async"}
    assert provider.last_mode_used is ProviderModeStrategy.MD_JSON
    assert [call["__mode"] for call in client.chat.completions.calls] == [
        "TOOLS",
        "JSON",
        "MD_JSON",
    ]


def test_provider_presets_keep_openai_compatible_defaults() -> None:
    ollama = OpenAICompatibleProvider.for_ollama(
        model="gemma4:26b",
        client=_SyncClient({"TOOLS": {"actions": [], "final_response": "ok"}}),
    )
    openai = OpenAICompatibleProvider.for_openai(
        model="gpt-4.1-mini",
        client=_SyncClient({"TOOLS": {"actions": [], "final_response": "ok"}}),
    )

    assert ollama.base_url == "http://localhost:11434/v1"
    assert ollama.model == "gemma4:26b"
    assert openai.base_url is None
    assert openai.model == "gpt-4.1-mini"


def test_provider_uses_local_fallback_when_instructor_missing(monkeypatch: Any) -> None:
    monkeypatch.setattr(provider_module, "_instructor", None)
    adapter, response_model = _response_model()
    client = _SyncClient(
        outcomes={
            "TOOLS": {"actions": [], "final_response": "fallback"},
            "JSON": {"actions": [], "final_response": "fallback"},
            "MD_JSON": {"actions": [], "final_response": "fallback"},
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=client)

    parsed = provider.run(
        adapter=adapter,
        messages=[{"role": "user", "content": "hello"}],
        response_model=response_model,
    )

    assert parsed.final_response == "fallback"


def test_provider_run_async_auto_does_not_retry_generic_failure(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    adapter, response_model = _response_model()
    client = _AsyncClient(
        outcomes={
            "TOOLS": RuntimeError("async transport failed"),
            "JSON": {"actions": [], "final_response": "should-not-run"},
            "MD_JSON": RuntimeError("should-not-run"),
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", async_client=client)

    with pytest.raises(RuntimeError, match="async transport failed"):
        asyncio.run(
            provider.run_async(
                adapter=adapter,
                messages=[{"role": "user", "content": "hello"}],
                response_model=response_model,
            )
        )

    assert [call["__mode"] for call in client.chat.completions.calls] == ["TOOLS"]


def test_provider_run_async_explicit_mode_raises_original_exception(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    adapter, response_model = _response_model()
    client = _AsyncClient(
        outcomes={
            "TOOLS": RuntimeError("unused"),
            "JSON": RuntimeError("unused"),
            "MD_JSON": RuntimeError("async md-json failed"),
        }
    )
    provider = OpenAICompatibleProvider(
        model="demo-model",
        async_client=client,
        mode_strategy=ProviderModeStrategy.MD_JSON,
    )

    with pytest.raises(RuntimeError, match="async md-json failed"):
        asyncio.run(
            provider.run_async(
                adapter=adapter,
                messages=[{"role": "user", "content": "hello"}],
                response_model=response_model,
            )
        )

    assert [call["__mode"] for call in client.chat.completions.calls] == ["MD_JSON"]


def test_provider_missing_instructor_async_does_not_resend_identical_requests(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(provider_module, "_instructor", None)
    adapter, response_model = _response_model()
    client = _AsyncClient(
        outcomes={
            "TOOLS": InstructorSchemaValidationError("schema validation failed"),
            "JSON": {"actions": [], "final_response": "should-not-run"},
            "MD_JSON": {"actions": [], "final_response": "should-not-run"},
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", async_client=client)

    with pytest.raises(ValueError, match="tools: InstructorSchemaValidationError"):
        asyncio.run(
            provider.run_async(
                adapter=adapter,
                messages=[{"role": "user", "content": "hello"}],
                response_model=response_model,
            )
        )

    assert len(client.chat.completions.calls) == 1


def test_provider_caches_wrapped_clients_per_mode(monkeypatch: Any) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=_SyncClient({"TOOLS": {"actions": [], "final_response": "ok"}}),
        async_client=_AsyncClient({"TOOLS": {"actions": [], "final_response": "ok"}}),
    )

    sync_client = provider._instructor_sync_client(ProviderModeStrategy.TOOLS)
    async_client = provider._instructor_async_client(ProviderModeStrategy.TOOLS)

    assert provider._instructor_sync_client(ProviderModeStrategy.TOOLS) is sync_client
    assert provider._instructor_async_client(ProviderModeStrategy.TOOLS) is async_client


def test_provider_missing_instructor_does_not_resend_identical_requests(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(provider_module, "_instructor", None)
    adapter, response_model = _response_model()
    client = _SyncClient(
        outcomes={
            "TOOLS": InstructorSchemaValidationError("schema validation failed"),
            "JSON": {"actions": [], "final_response": "should-not-run"},
            "MD_JSON": {"actions": [], "final_response": "should-not-run"},
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=client)

    with pytest.raises(ValueError, match="tools: InstructorSchemaValidationError"):
        provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )

    assert len(client.chat.completions.calls) == 1


def test_provider_bypasses_real_instructor_wrapper_for_non_openai_clients(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _StrictInstructorModule())
    adapter, response_model = _response_model()
    client = _SyncClient(
        outcomes={
            "TOOLS": {"actions": [], "final_response": "ok"},
            "JSON": {"actions": [], "final_response": "ok"},
            "MD_JSON": {"actions": [], "final_response": "ok"},
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=client)

    parsed = provider.run(
        adapter=adapter,
        messages=[{"role": "user", "content": "hello"}],
        response_model=response_model,
    )

    assert parsed.final_response == "ok"


def test_provider_bypassed_instructor_does_not_resend_identical_requests(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _StrictInstructorModule())
    adapter, response_model = _response_model()
    client = _SyncClient(
        outcomes={
            "TOOLS": InstructorSchemaValidationError("schema validation failed"),
            "JSON": {"actions": [], "final_response": "should-not-run"},
            "MD_JSON": {"actions": [], "final_response": "should-not-run"},
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=client)

    with pytest.raises(ValueError, match="tools: InstructorSchemaValidationError"):
        provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )

    assert len(client.chat.completions.calls) == 1
