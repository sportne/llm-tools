"""Focused tests for the shared app helpers that remain public."""

from __future__ import annotations

import importlib

import pytest
from pydantic import ValidationError

from llm_tools.apps.assistant_app.models import NiceGUIPreferences, NiceGUIRuntimeConfig
from llm_tools.apps.chat_config import (
    ChatLLMConfig,
    ChatPolicyConfig,
    ProviderAuthScheme,
    ProviderConnectionConfig,
    ProviderProtocol,
)
from llm_tools.apps.chat_presentation import (
    final_response_details,
    format_citation,
    format_confidence_label,
    format_final_response,
    format_final_response_metadata,
    format_transcript_text,
    pretty_json,
)
from llm_tools.llm_providers import ResponseModeStrategy
from llm_tools.tool_api.redaction import RedactionConfig
from llm_tools.workflow_api import ChatCitation, ChatFinalResponse

_CHAT_RUNTIME_MODULE = importlib.import_module("llm_tools.apps.chat_runtime")


def test_chat_config_validation_and_metadata() -> None:
    metadata = ChatLLMConfig().credential_prompt_metadata()
    assert metadata.expects_api_key is True

    custom_config = ChatLLMConfig(
        provider_connection=ProviderConnectionConfig(
            auth_scheme=ProviderAuthScheme.NONE
        ),
        prompt_for_api_key_if_missing=False,
    )
    custom_metadata = custom_config.credential_prompt_metadata()
    assert custom_metadata.prompt_for_api_key_if_missing is False
    assert custom_metadata.expects_api_key is False
    assert (
        ChatLLMConfig(response_mode_strategy="prompt_tools").response_mode_strategy
        is ResponseModeStrategy.PROMPT_TOOLS
    )

    assert ChatLLMConfig(selected_model="   ").selected_model is None
    assert (
        ChatLLMConfig(
            provider_connection=ProviderConnectionConfig(api_base_url="  ")
        ).provider_connection.api_base_url
        is None
    )
    with pytest.raises(ValidationError):
        ChatLLMConfig(temperature=1.5)
    with pytest.raises(ValidationError):
        ChatLLMConfig(timeout_seconds=0)
    with pytest.raises(ValidationError, match="requires_bearer_token"):
        ProviderConnectionConfig(requires_bearer_token=False)
    with pytest.raises(ValidationError):
        ChatPolicyConfig(enabled_tools=["read_file", " "])


def test_nicegui_runtime_models_validate_and_normalize() -> None:
    runtime = NiceGUIRuntimeConfig(
        selected_model="  demo  ",
        root_path="  ",
        enabled_tools=[" read_file ", "search_text"],
        disabled_skill_names=[" demo ", ""],
        disabled_skill_paths=[" /opt/demo/SKILL.md ", ""],
        response_mode_strategy="prompt_tools",
    )
    assert runtime.selected_model == "demo"
    assert runtime.provider_connection.api_base_url is None
    assert runtime.root_path is None
    assert runtime.enabled_tools == ["read_file", "search_text"]
    assert runtime.disabled_skill_names == ["demo"]
    assert runtime.disabled_skill_paths == ["/opt/demo/SKILL.md"]
    assert runtime.response_mode_strategy.value == "prompt_tools"

    assert NiceGUIRuntimeConfig().protection.enabled is False

    assert NiceGUIRuntimeConfig(selected_model="   ").selected_model is None
    with pytest.raises(ValueError):
        NiceGUIRuntimeConfig(enabled_tools=["read_file", " "])

    prefs = NiceGUIPreferences(
        recent_roots=["  /a  ", ""],
        recent_models={" ollama ": [" gemma ", " "]},
        recent_base_urls={" ": ["skip"], "openai": [" https://api.example.com ", " "]},
    )
    assert prefs.recent_roots == ["/a"]
    assert prefs.recent_models == {"ollama": ["gemma"]}
    assert prefs.recent_base_urls == {"openai": ["https://api.example.com"]}


def test_chat_presentation_formatters() -> None:
    citation = ChatCitation(
        source_path="README.md",
        line_start=4,
        line_end=7,
        excerpt="Relevant excerpt",
    )
    response = ChatFinalResponse(
        answer="Done",
        citations=[citation],
        confidence=0.7,
        uncertainty=["Need confirmation"],
        missing_information=["No workspace root"],
        follow_up_suggestions=["Select a root"],
    )

    assert pretty_json({"b": 1, "a": 2}).splitlines()[0] == "{"
    assert format_citation(citation) == "README.md:4-7"
    assert format_confidence_label(None) is None
    assert format_confidence_label(0.7) == "Confidence 70%"
    details = final_response_details(response)
    assert details.has_content is True
    assert details.citations[0].label == "README.md:4-7"
    assert details.citations[0].excerpt == "Relevant excerpt"
    assert details.confidence_label == "Confidence 70%"
    assert details.follow_up_suggestions == ("Select a root",)
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
    custom_calls: list[dict[str, object]] = []

    class _FakeProvider:
        def __init__(self, **kwargs: object) -> None:
            custom_calls.append(dict(kwargs))
            self.kind = "custom-provider"

    monkeypatch.setattr(_CHAT_RUNTIME_MODULE, "OpenAICompatibleProvider", _FakeProvider)
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    custom_provider = _CHAT_RUNTIME_MODULE.create_provider(
        provider_protocol=ProviderProtocol.OPENAI_API,
        provider_connection=ProviderConnectionConfig(
            api_base_url="https://example.invalid/v1"
        ),
        api_key="provided",
        selected_model="custom-model",
        response_mode_strategy="tools",
        timeout_seconds=config.timeout_seconds,
    )
    assert custom_provider.kind == "custom-provider"
    assert custom_calls[0]["api_key"] == "provided"
    assert custom_calls[0]["model"] == "custom-model"
    assert custom_calls[0]["base_url"] == "https://example.invalid/v1"
    assert custom_calls[0]["response_mode_strategy"] == "tools"
    env_provider = _CHAT_RUNTIME_MODULE.create_provider(
        provider_protocol=ProviderProtocol.OPENAI_API,
        provider_connection=ProviderConnectionConfig(
            api_base_url="https://example.invalid/v1"
        ),
        api_key=None,
        selected_model="env-model",
        response_mode_strategy="auto",
        timeout_seconds=config.timeout_seconds,
    )
    assert env_provider.kind == "custom-provider"
    assert custom_calls[1]["api_key"] == "env-key"
    assert custom_calls[1]["model"] == "env-model"
    no_bearer_provider = _CHAT_RUNTIME_MODULE.create_provider(
        provider_protocol=ProviderProtocol.OPENAI_API,
        provider_connection=ProviderConnectionConfig(
            api_base_url="http://127.0.0.1:11434/v1",
            auth_scheme=ProviderAuthScheme.NONE,
        ),
        api_key=None,
        selected_model="local-model",
        response_mode_strategy="auto",
        timeout_seconds=config.timeout_seconds,
        allow_env_api_key=False,
    )
    assert no_bearer_provider.kind == "custom-provider"
    assert custom_calls[2]["api_key"] == "unused"
    ollama_calls: list[dict[str, object]] = []

    class _FakeOllamaProvider:
        def __init__(self, **kwargs: object) -> None:
            ollama_calls.append(dict(kwargs))
            self.kind = "ollama-native"

    monkeypatch.setattr(
        _CHAT_RUNTIME_MODULE, "OllamaNativeProvider", _FakeOllamaProvider
    )
    ollama_provider = _CHAT_RUNTIME_MODULE.create_provider(
        provider_protocol=ProviderProtocol.OLLAMA_NATIVE,
        provider_connection=ProviderConnectionConfig(
            api_base_url="http://127.0.0.1:11434",
            auth_scheme=ProviderAuthScheme.NONE,
        ),
        api_key=None,
        selected_model="llama3.2",
        response_mode_strategy="auto",
        timeout_seconds=30,
    )
    assert ollama_provider.kind == "ollama-native"
    assert ollama_calls == [
        {
            "model": "llama3.2",
            "host": "http://127.0.0.1:11434",
            "response_mode_strategy": "auto",
            "default_request_params": {"timeout": 30},
        }
    ]
    ask_sage_calls: list[dict[str, object]] = []

    class _FakeAskSageProvider:
        def __init__(self, **kwargs: object) -> None:
            ask_sage_calls.append(dict(kwargs))
            self.kind = "ask-sage-native"

    monkeypatch.setattr(
        _CHAT_RUNTIME_MODULE, "AskSageNativeProvider", _FakeAskSageProvider
    )
    ask_sage_provider = _CHAT_RUNTIME_MODULE.create_provider(
        provider_protocol=ProviderProtocol.ASK_SAGE_NATIVE,
        provider_connection=ProviderConnectionConfig(
            api_base_url="https://api.asksage.ai/server",
            auth_scheme=ProviderAuthScheme.X_ACCESS_TOKENS,
        ),
        provider_request_settings={"persona": 7},
        api_key="token",
        selected_model="gpt-4.1-mini",
        response_mode_strategy="json",
        timeout_seconds=45,
    )
    assert ask_sage_provider.kind == "ask-sage-native"
    assert ask_sage_calls == [
        {
            "model": "gpt-4.1-mini",
            "access_token": "token",
            "base_url": "https://api.asksage.ai/server",
            "response_mode_strategy": "json",
            "request_settings": {"persona": 7},
            "default_request_params": {"timeout": 45},
        }
    ]
    monkeypatch.undo()
    with pytest.raises(ValueError, match="HTTPS base URL"):
        _CHAT_RUNTIME_MODULE.create_provider(
            provider_protocol=ProviderProtocol.ASK_SAGE_NATIVE,
            provider_connection=ProviderConnectionConfig(
                api_base_url="http://api.asksage.ai/server",
                auth_scheme=ProviderAuthScheme.X_ACCESS_TOKENS,
            ),
            api_key="token",
            selected_model="gpt-4.1-mini",
            response_mode_strategy="json",
            timeout_seconds=45,
        )
    with pytest.raises(ValueError, match="Ollama provider protocol uses auth"):
        _CHAT_RUNTIME_MODULE.create_provider(
            provider_protocol=ProviderProtocol.OLLAMA_NATIVE,
            provider_connection=ProviderConnectionConfig(
                api_base_url="http://127.0.0.1:11434",
                auth_scheme=ProviderAuthScheme.BEARER,
            ),
            api_key=None,
            selected_model="llama3.2",
            response_mode_strategy="auto",
            timeout_seconds=30,
        )
    with pytest.raises(ValueError, match="Ask Sage provider protocol uses auth"):
        _CHAT_RUNTIME_MODULE.create_provider(
            provider_protocol=ProviderProtocol.ASK_SAGE_NATIVE,
            provider_connection=ProviderConnectionConfig(
                api_base_url="https://api.asksage.ai/server",
                auth_scheme=ProviderAuthScheme.BEARER,
            ),
            api_key="token",
            selected_model="gpt-4.1-mini",
            response_mode_strategy="json",
            timeout_seconds=45,
        )
    with pytest.raises(ValueError, match="provider credentials"):
        _CHAT_RUNTIME_MODULE.create_provider(
            provider_protocol=ProviderProtocol.ASK_SAGE_NATIVE,
            provider_connection=ProviderConnectionConfig(
                api_base_url="https://api.asksage.ai/server",
                auth_scheme=ProviderAuthScheme.X_ACCESS_TOKENS,
            ),
            api_key=None,
            selected_model="gpt-4.1-mini",
            response_mode_strategy="json",
            timeout_seconds=45,
        )
    with pytest.raises(ValueError, match="does not support x_access_tokens"):
        _CHAT_RUNTIME_MODULE.create_provider(
            provider_protocol=ProviderProtocol.OPENAI_API,
            provider_connection=ProviderConnectionConfig(
                api_base_url="https://example.invalid/v1",
                auth_scheme=ProviderAuthScheme.X_ACCESS_TOKENS,
            ),
            api_key="token",
            selected_model="custom-model",
            response_mode_strategy="auto",
            timeout_seconds=config.timeout_seconds,
        )
    with pytest.raises(ValueError, match="Choose a model"):
        _CHAT_RUNTIME_MODULE.create_provider(
            provider_protocol=ProviderProtocol.OPENAI_API,
            provider_connection=ProviderConnectionConfig(
                api_base_url="https://example.invalid/v1"
            ),
            api_key=None,
            selected_model="  ",
            response_mode_strategy="auto",
            timeout_seconds=config.timeout_seconds,
        )
    with pytest.raises(ValueError, match="API base URL"):
        _CHAT_RUNTIME_MODULE.create_provider(
            provider_protocol=ProviderProtocol.OPENAI_API,
            provider_connection=ProviderConnectionConfig(api_base_url=None),
            api_key=None,
            selected_model="custom-model",
            response_mode_strategy="auto",
            timeout_seconds=config.timeout_seconds,
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
    assert final_response_details(bare_response).has_content is False
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
