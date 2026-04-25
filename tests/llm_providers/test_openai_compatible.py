"""Tests for the Instructor-backed OpenAI-compatible provider."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace
from typing import Any

import pytest
from pydantic import BaseModel

import llm_tools.llm_providers.openai_compatible as provider_module
from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider, ProviderModeStrategy
from llm_tools.tool_api import ToolSpec


class EchoInput(BaseModel):
    value: str


class SimplePayload(BaseModel):
    answer: str


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
        mode = kwargs.get("__mode", "RAW")
        if mode not in self._outcomes:
            mode = "TOOLS"
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
        mode = kwargs.get("__mode", "RAW")
        if mode not in self._outcomes:
            mode = "TOOLS"
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


def _text_response(content: str) -> object:
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


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


def test_provider_run_text_sends_plain_chat_completion() -> None:
    client = _SyncClient(outcomes={"TOOLS": _text_response("plain text")})
    provider = OpenAICompatibleProvider(model="demo-model", client=client)

    text = provider.run_text(
        messages=[{"role": "user", "content": "hello"}],
        request_params={"temperature": 0.2},
    )

    assert text == "plain text"
    assert provider.last_mode_used is ProviderModeStrategy.PROMPT_TOOLS
    call = client.chat.completions.calls[0]
    assert call["model"] == "demo-model"
    assert call["temperature"] == 0.2
    assert "tools" not in call
    assert "response_model" not in call
    assert "response_format" not in call


def test_provider_prompt_tools_structured_mode_uses_raw_text_transport() -> None:
    client = _SyncClient(outcomes={"TOOLS": _text_response('{"answer": "ok"}')})
    provider = OpenAICompatibleProvider(
        model="demo-model",
        client=client,
        mode_strategy=ProviderModeStrategy.PROMPT_TOOLS,
    )

    payload = provider.run_structured(
        messages=[{"role": "user", "content": "hello"}],
        response_model=SimplePayload,
    )

    assert payload == SimplePayload(answer="ok")
    assert provider.last_mode_used is ProviderModeStrategy.PROMPT_TOOLS
    call = client.chat.completions.calls[0]
    assert "tools" not in call
    assert "response_model" not in call
    assert "response_format" not in call
    assert call["messages"][-1]["role"] == "system"
    assert "Return a JSON object" in call["messages"][-1]["content"]


def test_provider_run_text_async_sends_plain_chat_completion() -> None:
    async_client = _AsyncClient(outcomes={"TOOLS": _text_response("plain text")})
    provider = OpenAICompatibleProvider(
        model="demo-model",
        async_client=async_client,
    )

    text = asyncio.run(
        provider.run_text_async(
            messages=[{"role": "user", "content": "hello"}],
            request_params={"temperature": 0.2},
        )
    )

    assert text == "plain text"
    assert provider.last_mode_used is ProviderModeStrategy.PROMPT_TOOLS
    call = async_client.chat.completions.calls[0]
    assert "tools" not in call
    assert "response_model" not in call
    assert "response_format" not in call


def test_provider_run_auto_falls_back_to_json_for_retryable_schema_errors(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    adapter, response_model = _response_model()
    client = _SyncClient(
        outcomes={
            "TOOLS": _wrapped_schema_failure("schema validation failed"),
            "JSON": {"actions": [], "final_response": "done"},
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
    assert [call.get("__mode") for call in client.chat.completions.calls] == [
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
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=client)

    parsed = provider.run(
        adapter=adapter,
        messages=[{"role": "user", "content": "hello"}],
        response_model=response_model,
    )

    assert parsed.final_response == "wrapped"
    assert [call.get("__mode") for call in client.chat.completions.calls] == [
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


def test_provider_run_async_uses_fallback_order_for_retryable_schema_errors(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    adapter, response_model = _response_model()
    client = _AsyncClient(
        outcomes={
            "TOOLS": _wrapped_schema_failure("schema validation failed"),
            "JSON": _wrapped_schema_failure("json validation failed"),
            "RAW": {
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
    assert provider.last_mode_used is ProviderModeStrategy.PROMPT_TOOLS
    assert [call.get("__mode") for call in client.chat.completions.calls] == [
        "TOOLS",
        "JSON",
        None,
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


def test_provider_missing_instructor_async_does_not_resend_identical_requests(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(provider_module, "_instructor", None)
    adapter, response_model = _response_model()
    client = _AsyncClient(
        outcomes={
            "TOOLS": InstructorSchemaValidationError("schema validation failed"),
            "JSON": {"actions": [], "final_response": "should-not-run"},
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", async_client=client)

    with pytest.raises(
        ValueError,
        match=r"tools: schema/parse-related \(InstructorSchemaValidationError: schema validation failed\)",
    ):
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
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=client)

    with pytest.raises(
        ValueError,
        match=r"tools: schema/parse-related \(InstructorSchemaValidationError: schema validation failed\)",
    ):
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
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=client)

    with pytest.raises(
        ValueError,
        match=r"tools: schema/parse-related \(InstructorSchemaValidationError: schema validation failed\)",
    ):
        provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )

    assert len(client.chat.completions.calls) == 1


def test_provider_run_auto_reports_all_retryable_mode_failures(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(provider_module, "_instructor", _FakeInstructor())
    adapter, response_model = _response_model()
    client = _SyncClient(
        outcomes={
            "TOOLS": _wrapped_schema_failure("schema validation failed"),
            "JSON": _wrapped_schema_failure("json validation failed"),
            "RAW": _wrapped_schema_failure("prompt tools validation failed"),
        }
    )
    provider = OpenAICompatibleProvider(model="demo-model", client=client)

    with pytest.raises(
        ValueError, match="All provider mode attempts failed"
    ) as exc_info:
        provider.run(
            adapter=adapter,
            messages=[{"role": "user", "content": "hello"}],
            response_model=response_model,
        )

    message = str(exc_info.value)
    assert "Overall failure type: schema/parse-related" in message
    assert (
        "tools: schema/parse-related (RuntimeError: schema validation failed)"
        in message
    )
    assert (
        "json: schema/parse-related (RuntimeError: json validation failed)" in message
    )
    assert (
        "prompt_tools: schema/parse-related (RuntimeError: prompt tools validation failed)"
        in message
    )
    assert [call.get("__mode") for call in client.chat.completions.calls] == [
        "TOOLS",
        "JSON",
        None,
    ]
