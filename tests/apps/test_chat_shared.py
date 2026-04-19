"""Focused tests for the shared app helpers that remain public."""

from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

from llm_tools.apps.chat_config import ChatLLMConfig, ChatPolicyConfig, ProviderPreset
from llm_tools.apps.chat_presentation import (
    format_citation,
    format_final_response,
    format_final_response_metadata,
    format_transcript_text,
    pretty_json,
)
from llm_tools.apps.streamlit_models import StreamlitPreferences, StreamlitRuntimeConfig
from llm_tools.tool_api.redaction import RedactionConfig
from llm_tools.workflow_api import ChatCitation, ChatFinalResponse

_CHAT_RUNTIME_MODULE = importlib.import_module("llm_tools.apps.chat_runtime")


def test_chat_config_validation_and_metadata() -> None:
    metadata = ChatLLMConfig(
        provider=ProviderPreset.OPENAI
    ).credential_prompt_metadata()
    assert metadata.api_key_env_var == "OPENAI_API_KEY"
    assert metadata.expects_api_key is True

    custom_metadata = ChatLLMConfig(
        provider=ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
        prompt_for_api_key_if_missing=False,
    ).credential_prompt_metadata()
    assert custom_metadata.api_key_env_var == "OPENAI_API_KEY"
    assert custom_metadata.prompt_for_api_key_if_missing is False

    ollama_metadata = ChatLLMConfig(api_key_env_var=None).credential_prompt_metadata()
    assert ollama_metadata.api_key_env_var == "API key"
    assert ollama_metadata.expects_api_key is False

    with pytest.raises(ValidationError):
        ChatLLMConfig(model_name="   ")
    with pytest.raises(ValidationError):
        ChatLLMConfig(api_base_url="   ")
    with pytest.raises(ValidationError):
        ChatLLMConfig(temperature=1.5)
    with pytest.raises(ValidationError):
        ChatLLMConfig(timeout_seconds=0)
    with pytest.raises(ValidationError):
        ChatPolicyConfig(enabled_tools=["read_file", " "])


def test_streamlit_models_validate_and_normalize() -> None:
    runtime = StreamlitRuntimeConfig(
        model_name="  demo  ",
        api_base_url="  ",
        root_path="  ",
        enabled_tools=[" read_file ", "search_text"],
        provider_mode_strategy="json",
    )
    assert runtime.model_name == "demo"
    assert runtime.api_base_url is None
    assert runtime.root_path is None
    assert runtime.enabled_tools == ["read_file", "search_text"]
    assert runtime.provider_mode_strategy.value == "json"

    with pytest.raises(ValueError):
        StreamlitRuntimeConfig(model_name="   ")
    with pytest.raises(ValueError):
        StreamlitRuntimeConfig(enabled_tools=["read_file", " "])

    prefs = StreamlitPreferences(
        recent_roots=["  /a  ", ""],
        recent_models={" ollama ": [" gemma ", " "]},
        recent_base_urls={" ": ["skip"], "openai": [" https://api.example.com ", " "]},
    )
    assert prefs.recent_roots == ["/a"]
    assert prefs.recent_models == {"ollama": ["gemma"]}
    assert prefs.recent_base_urls == {"openai": ["https://api.example.com"]}


def test_chat_presentation_formatters() -> None:
    citation = ChatCitation(source_path="README.md", line_start=4, line_end=7)
    response = ChatFinalResponse(
        answer="Done",
        citations=[citation],
        uncertainty=["Need confirmation"],
        missing_information=["No workspace root"],
        follow_up_suggestions=["Select a root"],
    )

    assert pretty_json({"b": 1, "a": 2}).splitlines()[0] == "{"
    assert format_citation(citation) == "README.md:4-7"
    assert "Citations:" in format_final_response(response)
    assert "Missing Information:" in format_final_response_metadata(response)
    assert format_transcript_text("assistant", "Answer") == "Assistant:\nAnswer"
    assert (
        format_transcript_text(
            "assistant",
            "Partial",
            assistant_completion_state="interrupted",
        )
        == "Assistant (interrupted):\nPartial"
    )


def test_chat_runtime_provider_factory_and_executor_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = ChatLLMConfig()
    openai_calls: list[dict[str, object]] = []
    ollama_calls: list[dict[str, object]] = []
    custom_calls: list[dict[str, object]] = []

    class _FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            custom_calls.append(dict(kwargs))
            self.kind = "custom-provider"

        @classmethod
        def for_openai(cls, **kwargs: object) -> str:
            openai_calls.append(dict(kwargs))
            return "openai-provider"

        @classmethod
        def for_ollama(cls, **kwargs: object) -> str:
            ollama_calls.append(dict(kwargs))
            return "ollama-provider"

    monkeypatch.setattr(_CHAT_RUNTIME_MODULE, "OpenAICompatibleProvider", _FakeProvider)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    openai_config = config.model_copy(
        update={
            "provider": ProviderPreset.OPENAI,
            "api_key_env_var": "OPENAI_API_KEY",
            "api_base_url": None,
        }
    )
    assert (
        _CHAT_RUNTIME_MODULE.create_provider(
            openai_config,
            api_key=None,
            model_name="gpt-4.1-mini",
            mode_strategy="json",
        )
        == "openai-provider"
    )
    assert openai_calls[0]["api_key"] == "env-key"
    assert openai_calls[0]["mode_strategy"] == "json"

    ollama_config = config.model_copy(
        update={
            "provider": ProviderPreset.OLLAMA,
            "api_base_url": None,
        }
    )
    assert (
        _CHAT_RUNTIME_MODULE.create_provider(
            ollama_config,
            api_key=None,
            model_name="gemma4:26b",
            mode_strategy="md_json",
        )
        == "ollama-provider"
    )
    assert ollama_calls[0]["base_url"] == "http://127.0.0.1:11434/v1"
    assert ollama_calls[0]["api_key"] == "ollama"
    assert ollama_calls[0]["mode_strategy"] == "md_json"

    custom_config = config.model_copy(
        update={
            "provider": ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
            "api_base_url": "https://example.invalid/v1",
            "api_key_env_var": "OPENAI_API_KEY",
        }
    )
    custom_provider = _CHAT_RUNTIME_MODULE.create_provider(
        custom_config,
        api_key="provided",
        model_name="custom-model",
        mode_strategy="tools",
    )
    assert custom_provider.kind == "custom-provider"
    assert custom_calls[0]["api_key"] == "provided"
    assert custom_calls[0]["mode_strategy"] == "tools"
    with pytest.raises(ValueError, match="require api_base_url"):
        _CHAT_RUNTIME_MODULE.create_provider(
            custom_config.model_copy(update={"api_base_url": None}),
            api_key=None,
            model_name="custom-model",
        )

    captured: dict[str, object] = {}
    monkeypatch.setattr(_CHAT_RUNTIME_MODULE, "build_chat_registry", lambda: "registry")

    def _fake_workflow_executor(**kwargs):
        captured["executor_kwargs"] = kwargs
        return "executor"

    monkeypatch.setattr(
        _CHAT_RUNTIME_MODULE, "WorkflowExecutor", _fake_workflow_executor
    )
    registry, executor = _CHAT_RUNTIME_MODULE.build_chat_executor(
        redaction_config=RedactionConfig(redact_logs=True)
    )
    assert registry == "registry"
    assert executor == "executor"
    policy = captured["executor_kwargs"]["policy"]
    assert policy.allow_network is False
    assert policy.allow_filesystem is True
    assert policy.allow_subprocess is False
    assert policy.redaction.redact_logs is True


def test_chat_runtime_registry_contains_expected_read_tools() -> None:
    registry = _CHAT_RUNTIME_MODULE.build_chat_registry()
    tool_names = {spec.name for spec in registry.list_tools()}
    assert {"list_directory", "find_files", "read_file", "search_text"}.issubset(
        tool_names
    )


def test_chat_presentation_empty_and_single_line_branches() -> None:
    bare_response = ChatFinalResponse(answer="Only answer")
    assert format_final_response(bare_response) == "Only answer"
    assert format_final_response_metadata(bare_response) == ""
    assert pretty_json(None) == ""

    path_only = ChatCitation(source_path="README.md")
    single_line = ChatCitation(source_path="README.md", line_start=9)
    same_line = ChatCitation(source_path="README.md", line_start=11, line_end=11)
    assert format_citation(path_only) == "README.md"
    assert format_citation(single_line) == "README.md:9"
    assert format_citation(same_line) == "README.md:11"

    assert format_transcript_text("user", "Question") == "You:\nQuestion"
    assert format_transcript_text("error", "Problem") == "Error: Problem"
    assert format_transcript_text("system", "Notice") == "System:\nNotice"
