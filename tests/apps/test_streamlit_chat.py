"""Tests for the Streamlit repository chat app layer."""

from __future__ import annotations

import importlib
import runpy
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.apps._imports import import_streamlit_chat_modules

from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import SideEffectClass
from llm_tools.workflow_api import (
    ChatFinalResponse,
    ChatSessionState,
    ChatTokenUsage,
    ChatWorkflowApprovalEvent,
    ChatWorkflowApprovalResolvedEvent,
    ChatWorkflowApprovalState,
    ChatWorkflowResultEvent,
    ChatWorkflowTurnResult,
)
from llm_tools.workflow_api.models import ApprovalRequest

_STREAMLIT_MODULES = import_streamlit_chat_modules()
_CHAT_RUNTIME_MODULE = importlib.import_module("llm_tools.apps.chat_runtime")
build_parser = _STREAMLIT_MODULES.app.build_parser
process_streamlit_chat_turn = _STREAMLIT_MODULES.app.process_streamlit_chat_turn
resolve_enabled_tool_names = _STREAMLIT_MODULES.app.resolve_enabled_tool_names
StreamlitTurnOutcome = _STREAMLIT_MODULES.app.StreamlitTurnOutcome
_resolve_chat_config = _STREAMLIT_MODULES.app._resolve_chat_config


class _FakeProvider:
    def __init__(self, responses: list[ParsedModelResponse]) -> None:
        self._responses = list(responses)

    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        return self._responses.pop(0)


class _FakeRunner:
    def __init__(self, events: list[object]) -> None:
        self._events = events
        self.resolutions: list[bool] = []

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._events)

    def resolve_pending_approval(self, approved: bool) -> bool:
        self.resolutions.append(approved)
        return True


class _RerunRequestError(RuntimeError):
    pass


class _FakeSidebar:
    def __init__(self, streamlit: _FakeStreamlit) -> None:
        self._streamlit = streamlit

    def __enter__(self) -> _FakeSidebar:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def subheader(self, text: str) -> None:
        self._streamlit.sidebar_messages.append(("subheader", text))

    def text_input(self, label: str, *, value: str = "", type: str = "default") -> str:
        del value, type
        self._streamlit.sidebar_messages.append(("text_input", label))
        return self._streamlit.sidebar_text_input


class _NullContext:
    def __enter__(self) -> _NullContext:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False


class _FakeStreamlit:
    def __init__(
        self,
        *,
        chat_input: str | None = None,
        button_value: bool = False,
        sidebar_text_input: str = "",
    ) -> None:
        self.session_state: dict[str, object] = {}
        self.chat_input_value = chat_input
        self.button_value = button_value
        self.sidebar_text_input = sidebar_text_input
        self.sidebar = _FakeSidebar(self)
        self.info_messages: list[str] = []
        self.error_messages: list[str] = []
        self.markdown_messages: list[str] = []
        self.caption_messages: list[str] = []
        self.sidebar_messages: list[tuple[str, str]] = []
        self.chat_message_roles: list[str] = []
        self.chat_input_calls = 0
        self.rerun_called = False
        self.page_config_calls = 0
        self.title_messages: list[str] = []
        self.button_calls = 0

    def set_page_config(self, **kwargs: object) -> None:
        del kwargs
        self.page_config_calls += 1

    def title(self, text: str) -> None:
        self.title_messages.append(text)
        self.markdown_messages.append(text)

    def subheader(self, text: str) -> None:
        self.markdown_messages.append(text)

    def markdown(self, text: str) -> None:
        self.markdown_messages.append(text)

    def caption(self, text: str) -> None:
        self.caption_messages.append(text)

    def button(self, label: str, *, use_container_width: bool = False) -> bool:
        del label, use_container_width
        self.button_calls += 1
        return self.button_value

    def info(self, text: str) -> None:
        self.info_messages.append(text)

    def error(self, text: str) -> None:
        self.error_messages.append(text)

    def code(self, text: str) -> None:
        self.markdown_messages.append(text)

    def chat_message(self, role: str) -> _NullContext:
        self.chat_message_roles.append(role)
        return _NullContext()

    def chat_input(self, prompt: str) -> str | None:
        del prompt
        self.chat_input_calls += 1
        return self.chat_input_value

    def spinner(self, text: str) -> _NullContext:
        self.caption_messages.append(text)
        return _NullContext()

    def rerun(self) -> None:
        self.rerun_called = True
        raise _RerunRequestError()


def test_streamlit_chat_package_imports_without_loading_streamlit() -> None:
    module = importlib.import_module("llm_tools.apps.streamlit_chat")
    main_module = importlib.import_module("llm_tools.apps.streamlit_chat.__main__")

    assert hasattr(module, "main")
    assert hasattr(module, "run_streamlit_chat_app")
    assert hasattr(main_module, "main")


def test_streamlit_chat_config_loading_and_cli_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "chat.yaml"
    config_path.write_text(
        """
llm:
  provider: ollama
  model_name: base-model
session:
  max_context_tokens: 99
""".strip(),
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        [
            str(tmp_path),
            "--config",
            str(config_path),
            "--model",
            "override-model",
            "--max-context-tokens",
            "123",
        ]
    )

    resolved = _resolve_chat_config(args)
    assert resolved.llm.model_name == "override-model"
    assert resolved.session.max_context_tokens == 123


def test_chat_runtime_provider_factory_and_context_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    openai_calls: list[dict[str, object]] = []
    ollama_calls: list[dict[str, object]] = []
    custom_calls: list[dict[str, object]] = []

    monkeypatch.setattr(
        "llm_tools.apps.chat_runtime.OpenAICompatibleProvider.for_openai",
        lambda **kwargs: openai_calls.append(kwargs) or SimpleNamespace(kind="openai"),
    )
    monkeypatch.setattr(
        "llm_tools.apps.chat_runtime.OpenAICompatibleProvider.for_ollama",
        lambda **kwargs: ollama_calls.append(kwargs) or SimpleNamespace(kind="ollama"),
    )

    _CHAT_RUNTIME_MODULE.create_provider(
        _STREAMLIT_MODULES.package.TextualChatConfig().llm.model_copy(
            update={"provider": _STREAMLIT_MODULES.package.ProviderPreset.OPENAI}
        ),
        api_key="secret",
        model_name="gpt-demo",
    )
    _CHAT_RUNTIME_MODULE.create_provider(
        _STREAMLIT_MODULES.package.TextualChatConfig().llm.model_copy(
            update={"provider": _STREAMLIT_MODULES.package.ProviderPreset.OLLAMA}
        ),
        api_key=None,
        model_name="llama-demo",
    )

    class _CustomProvider:
        @classmethod
        def for_openai(cls, **kwargs: object) -> object:
            return SimpleNamespace(kind="openai-from-custom", kwargs=kwargs)

        @classmethod
        def for_ollama(cls, **kwargs: object) -> object:
            return SimpleNamespace(kind="ollama-from-custom", kwargs=kwargs)

        def __init__(self, **kwargs: object) -> None:
            custom_calls.append(kwargs)

    monkeypatch.setattr(
        "llm_tools.apps.chat_runtime.OpenAICompatibleProvider",
        _CustomProvider,
    )
    _CHAT_RUNTIME_MODULE.create_provider(
        _STREAMLIT_MODULES.package.TextualChatConfig().llm.model_copy(
            update={
                "provider": _STREAMLIT_MODULES.package.ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
                "api_base_url": "http://custom/v1",
            }
        ),
        api_key="secret",
        model_name="custom-demo",
    )
    with pytest.raises(ValueError):
        _CHAT_RUNTIME_MODULE.create_provider(
            _STREAMLIT_MODULES.package.TextualChatConfig().llm.model_copy(
                update={
                    "provider": _STREAMLIT_MODULES.package.ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
                    "api_base_url": None,
                }
            ),
            api_key=None,
            model_name="custom-demo",
        )

    config = _STREAMLIT_MODULES.package.TextualChatConfig().model_copy(
        update={
            "session": _STREAMLIT_MODULES.package.TextualChatConfig().session.model_copy(
                update={"max_context_tokens": 10}
            )
        }
    )
    context = _CHAT_RUNTIME_MODULE.build_chat_context(
        root_path=tmp_path,
        config=config,
        app_name="streamlit-chat",
    )
    assert context.workspace == str(tmp_path)
    assert context.metadata["tool_limits"]["max_read_file_chars"] == 40
    assert openai_calls and ollama_calls and custom_calls


def test_chat_runtime_executor_and_registry_helpers() -> None:
    registry, executor = _CHAT_RUNTIME_MODULE.build_chat_executor()
    assert registry.list_registered_tools()
    assert executor._policy.allow_filesystem is True
    assert executor._policy.allow_network is False
    assert "read_file" in _CHAT_RUNTIME_MODULE.build_available_tool_names()
    assert "read_file" in _CHAT_RUNTIME_MODULE.build_available_tool_specs()


def test_streamlit_chat_resolve_enabled_tools_filters_unknown_entries() -> None:
    config = _STREAMLIT_MODULES.package.TextualChatConfig().model_copy(
        update={
            "policy": _STREAMLIT_MODULES.package.TextualChatConfig().policy.model_copy(
                update={"enabled_tools": ["search_text", "unknown"]}
            )
        }
    )

    assert resolve_enabled_tool_names(config) == {"search_text"}


def test_streamlit_chat_turn_processes_completed_result(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    config = _STREAMLIT_MODULES.package.TextualChatConfig()

    outcome = process_streamlit_chat_turn(
        root_path=tmp_path,
        config=config,
        provider=_FakeProvider(
            [
                ParsedModelResponse(
                    invocations=[
                        {
                            "tool_name": "search_text",
                            "arguments": {"path": ".", "query": "needle"},
                        }
                    ]
                ),
                ParsedModelResponse(
                    final_response=ChatFinalResponse(
                        answer="It is defined in src/app.py.",
                        citations=[{"source_path": "src/app.py", "line_start": 1}],
                        confidence=0.8,
                        uncertainty=["Only one match was checked."],
                        follow_up_suggestions=["Inspect surrounding code."],
                    ).model_dump(mode="json")
                ),
            ]
        ),
        session_state=ChatSessionState(),
        user_message="Where is needle defined?",
    )

    assert isinstance(outcome, StreamlitTurnOutcome)
    assert len(outcome.transcript_entries) == 1
    assert outcome.transcript_entries[0].role == "assistant"
    assert outcome.transcript_entries[0].final_response is not None
    assert outcome.transcript_entries[0].final_response.answer.startswith(
        "It is defined"
    )
    assert len(outcome.session_state.turns) == 1


def test_streamlit_chat_turn_denies_interactive_approvals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approval_state = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-1",
            invocation_index=1,
            request={
                "tool_name": "search_text",
                "arguments": {"path": ".", "query": "needle"},
            },
            tool_name="search_text",
            tool_version="0.1.0",
            policy_reason="approval required",
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:01:00Z",
        ),
        tool_name="search_text",
        redacted_arguments={"path": ".", "query": "needle"},
        policy_reason="approval required",
    )
    runner = _FakeRunner(
        [
            ChatWorkflowApprovalEvent(approval=approval_state),
            ChatWorkflowApprovalResolvedEvent(
                approval=approval_state,
                resolution="denied",
            ),
            ChatWorkflowResultEvent(
                result=ChatWorkflowTurnResult(
                    status="needs_continuation",
                    continuation_reason="Approval was denied.",
                )
            ),
        ]
    )
    monkeypatch.setattr(
        "llm_tools.apps.streamlit_chat.app.run_interactive_chat_session_turn",
        lambda **kwargs: runner,
    )

    outcome = process_streamlit_chat_turn(
        root_path=tmp_path,
        config=_STREAMLIT_MODULES.package.TextualChatConfig().model_copy(
            update={
                "policy": _STREAMLIT_MODULES.package.TextualChatConfig().policy.model_copy(
                    update={"require_approval_for": {SideEffectClass.LOCAL_READ}}
                )
            }
        ),
        provider=object(),
        session_state=ChatSessionState(),
        user_message="Need approval",
    )

    assert runner.resolutions == [False]
    assert [entry.role for entry in outcome.transcript_entries] == [
        "system",
        "system",
        "system",
    ]
    assert "denies interactive approvals" in outcome.transcript_entries[0].text
    assert outcome.transcript_entries[-1].text == "Approval was denied."


def test_streamlit_chat_turn_handles_timed_out_approval_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approval_state = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-1",
            invocation_index=1,
            request={
                "tool_name": "search_text",
                "arguments": {"path": ".", "query": "needle"},
            },
            tool_name="search_text",
            tool_version="0.1.0",
            policy_reason="approval required",
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:01:00Z",
        ),
        tool_name="search_text",
        redacted_arguments={"path": ".", "query": "needle"},
        policy_reason="approval required",
    )
    runner = _FakeRunner(
        [
            ChatWorkflowApprovalEvent(approval=approval_state),
            ChatWorkflowApprovalResolvedEvent(
                approval=approval_state,
                resolution="timed_out",
            ),
            ChatWorkflowResultEvent(
                result=ChatWorkflowTurnResult(
                    status="needs_continuation",
                    continuation_reason="Approval timed out.",
                )
            ),
        ]
    )
    monkeypatch.setattr(
        "llm_tools.apps.streamlit_chat.app.run_interactive_chat_session_turn",
        lambda **kwargs: runner,
    )

    outcome = process_streamlit_chat_turn(
        root_path=tmp_path,
        config=_STREAMLIT_MODULES.package.TextualChatConfig().model_copy(
            update={
                "policy": _STREAMLIT_MODULES.package.TextualChatConfig().policy.model_copy(
                    update={"require_approval_for": {SideEffectClass.LOCAL_READ}}
                )
            }
        ),
        provider=object(),
        session_state=ChatSessionState(),
        user_message="Need approval",
    )

    assert outcome.transcript_entries[1].text == "Pending approval request timed out."
    assert outcome.transcript_entries[-1].text == "Approval timed out."


def test_streamlit_chat_turn_handles_interrupted_result_with_context_warning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        "llm_tools.apps.streamlit_chat.app.run_interactive_chat_session_turn",
        lambda **kwargs: _FakeRunner(
            [
                ChatWorkflowResultEvent(
                    result=ChatWorkflowTurnResult(
                        status="interrupted",
                        interruption_reason="Stopped by user.",
                        context_warning="Older turns were removed.",
                    )
                )
            ]
        ),
    )

    outcome = process_streamlit_chat_turn(
        root_path=tmp_path,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        provider=object(),
        session_state=ChatSessionState(),
        user_message="Stop",
    )

    assert [entry.text for entry in outcome.transcript_entries] == [
        "Older turns were removed.",
        "Stopped by user.",
    ]


def test_streamlit_chat_turn_handles_approved_resolution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approval_state = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-1",
            invocation_index=1,
            request={
                "tool_name": "search_text",
                "arguments": {"path": ".", "query": "needle"},
            },
            tool_name="search_text",
            tool_version="0.1.0",
            policy_reason="approval required",
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:01:00Z",
        ),
        tool_name="search_text",
        redacted_arguments={"path": ".", "query": "needle"},
        policy_reason="approval required",
    )
    monkeypatch.setattr(
        "llm_tools.apps.streamlit_chat.app.run_interactive_chat_session_turn",
        lambda **kwargs: _FakeRunner(
            [
                ChatWorkflowApprovalResolvedEvent(
                    approval=approval_state,
                    resolution="approved",
                ),
                ChatWorkflowResultEvent(
                    result=ChatWorkflowTurnResult(
                        status="needs_continuation",
                        continuation_reason="Continue after approval.",
                    )
                ),
            ]
        ),
    )

    outcome = process_streamlit_chat_turn(
        root_path=tmp_path,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        provider=_FakeProvider([]),
        session_state=ChatSessionState(),
        user_message="Need approval",
    )

    assert outcome.transcript_entries[0].text == "Approved pending approval request."
    assert outcome.transcript_entries[1].text == "Continue after approval."


def test_streamlit_chat_app_requires_api_key_before_chat(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_STREAMLIT_MODULES.app.os, "getenv", lambda name: None)

    config = _STREAMLIT_MODULES.package.TextualChatConfig().model_copy(
        update={
            "llm": _STREAMLIT_MODULES.package.TextualChatConfig().llm.model_copy(
                update={"provider": _STREAMLIT_MODULES.package.ProviderPreset.OPENAI}
            )
        }
    )
    _STREAMLIT_MODULES.app.run_streamlit_chat_app(root_path=tmp_path, config=config)

    assert fake_st.page_config_calls == 1
    assert fake_st.title_messages == ["llm-tools Streamlit Chat"]
    assert fake_st.info_messages == [
        "Set OPENAI_API_KEY or enter it in the sidebar to start chatting."
    ]
    assert fake_st.chat_input_calls == 0
    assert fake_st.session_state[_STREAMLIT_MODULES.app._API_KEY_STATE_SLOT] == ""
    transcript = fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]
    assert len(transcript) == 1
    assert transcript[0].role == "system"
    assert (
        fake_st.session_state[_STREAMLIT_MODULES.app._SESSION_STATE_SLOT]
        == ChatSessionState()
    )
    assert fake_st.session_state[_STREAMLIT_MODULES.app._TOKEN_USAGE_STATE_SLOT] is None


def test_streamlit_chat_app_uses_env_api_key_without_prompt(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_STREAMLIT_MODULES.app.os, "getenv", lambda name: "env-secret")

    config = _STREAMLIT_MODULES.package.TextualChatConfig().model_copy(
        update={
            "llm": _STREAMLIT_MODULES.package.TextualChatConfig().llm.model_copy(
                update={"provider": _STREAMLIT_MODULES.package.ProviderPreset.OPENAI}
            )
        }
    )
    _STREAMLIT_MODULES.app.run_streamlit_chat_app(root_path=tmp_path, config=config)

    assert fake_st.chat_input_calls == 1
    assert not fake_st.info_messages
    assert fake_st.sidebar_messages == []


def test_streamlit_chat_app_clear_chat_resets_transcript_and_api_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(button_value=True)
    fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT] = [
        _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(role="user", text="old")
    ]
    fake_st.session_state[_STREAMLIT_MODULES.app._SESSION_STATE_SLOT] = (
        ChatSessionState(turns=[])
    )
    fake_st.session_state[_STREAMLIT_MODULES.app._TOKEN_USAGE_STATE_SLOT] = "tokens"
    fake_st.session_state[_STREAMLIT_MODULES.app._API_KEY_STATE_SLOT] = "secret"
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)

    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app.run_streamlit_chat_app(
            root_path=tmp_path,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )

    assert fake_st.rerun_called is True
    assert (
        len(fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]) == 1
    )
    assert fake_st.session_state[_STREAMLIT_MODULES.app._API_KEY_STATE_SLOT] == ""
    assert (
        fake_st.session_state[_STREAMLIT_MODULES.app._SESSION_STATE_SLOT]
        == ChatSessionState()
    )
    assert fake_st.session_state[_STREAMLIT_MODULES.app._TOKEN_USAGE_STATE_SLOT] is None


def test_streamlit_chat_app_processes_prompt_and_reruns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(chat_input="question")
    created_provider: dict[str, object] = {}
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "create_provider",
        lambda config, api_key, model_name: (
            created_provider.update(
                {
                    "provider": SimpleNamespace(model=model_name),
                    "provider_name": config.provider.value,
                    "api_key": api_key,
                    "model_name": model_name,
                }
            )
            or created_provider["provider"]
        ),
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "process_streamlit_chat_turn",
        lambda **kwargs: _STREAMLIT_MODULES.app.StreamlitTurnOutcome(
            session_state=ChatSessionState(),
            transcript_entries=[
                _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
                    role="assistant",
                    text="Answer",
                    final_response=ChatFinalResponse(answer="Answer"),
                )
            ],
            token_usage=ChatTokenUsage(session_tokens=11, active_context_tokens=22),
        ),
    )

    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app.run_streamlit_chat_app(
            root_path=tmp_path,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )

    transcript = fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]
    assert [entry.role for entry in transcript] == ["system", "user", "assistant"]
    assert transcript[1].text == "question"
    assert created_provider == {
        "provider": created_provider["provider"],
        "provider_name": "ollama",
        "api_key": None,
        "model_name": "gemma4:26b",
    }
    assert fake_st.session_state[
        _STREAMLIT_MODULES.app._TOKEN_USAGE_STATE_SLOT
    ] == ChatTokenUsage(session_tokens=11, active_context_tokens=22)
    assert fake_st.rerun_called is True


def test_streamlit_chat_app_records_turn_exception_and_reruns(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(chat_input="question")
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "create_provider",
        lambda config, api_key, model_name: SimpleNamespace(model=model_name),
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "process_streamlit_chat_turn",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app.run_streamlit_chat_app(
            root_path=tmp_path,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )

    transcript = fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]
    assert [entry.role for entry in transcript] == ["system", "user", "error"]
    assert transcript[-1].role == "error"
    assert transcript[-1].text == "boom"
    assert fake_st.rerun_called is True


def test_streamlit_chat_render_helpers_cover_remaining_branches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)

    _STREAMLIT_MODULES.app._render_final_response(
        ChatFinalResponse(
            answer="Answer",
            citations=[
                {
                    "source_path": "src/app.py",
                    "line_start": 1,
                    "excerpt": "needle = 1",
                }
            ],
            confidence=0.5,
            uncertainty=["unclear"],
            missing_information=["missing"],
            follow_up_suggestions=["next"],
        )
    )
    _STREAMLIT_MODULES.app._render_transcript_entry(
        _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(role="user", text="hello")
    )
    _STREAMLIT_MODULES.app._render_transcript_entry(
        _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(role="system", text="notice")
    )
    _STREAMLIT_MODULES.app._render_transcript_entry(
        _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(role="error", text="boom")
    )
    _STREAMLIT_MODULES.app._render_transcript_entry(
        _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
            role="assistant",
            text="plain assistant text",
        )
    )

    assert "needle = 1" in fake_st.markdown_messages
    assert fake_st.error_messages == ["boom"]


def test_streamlit_chat_app_propagates_provider_creation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(chat_input="question")
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "create_provider",
        lambda config, api_key, model_name: (_ for _ in ()).throw(
            ValueError("provider failed")
        ),
    )

    with pytest.raises(ValueError, match="provider failed"):
        _STREAMLIT_MODULES.app.run_streamlit_chat_app(
            root_path=tmp_path,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )

    transcript = fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]
    assert [entry.role for entry in transcript] == ["system"]
    assert fake_st.rerun_called is False


def test_streamlit_chat_main_and_runner_dispatch_to_app_layer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.streamlit_chat")
    called: list[str] = []

    monkeypatch.setattr(
        "llm_tools.apps.streamlit_chat.app.run_streamlit_chat_app",
        lambda **kwargs: called.append("run"),
    )
    monkeypatch.setattr(
        "llm_tools.apps.streamlit_chat.app.main",
        lambda argv=None: called.append("main") or (0 if argv is None else len(argv)),
    )

    package.run_streamlit_chat_app(
        root_path=Path("."),
        config=package.TextualChatConfig(),
    )
    assert package.main() == 0
    assert called == ["run", "main"]


def test_streamlit_chat_module_entrypoint_dispatches_to_package_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.streamlit_chat")
    main_module = importlib.import_module("llm_tools.apps.streamlit_chat.__main__")

    monkeypatch.setattr(package, "main", lambda: 7)

    assert main_module._main() == 7
    assert main_module.main() == 7


def test_streamlit_chat_module_entrypoint_raises_system_exit_with_main_return_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = importlib.import_module("llm_tools.apps.streamlit_chat")
    called: list[str] = []
    monkeypatch.setattr(package, "main", lambda: called.append("main") or 0)

    sys.modules.pop("llm_tools.apps.streamlit_chat.__main__", None)
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("llm_tools.apps.streamlit_chat.__main__", run_name="__main__")

    assert exc.value.code == 0
    assert called == ["main"]


def test_streamlit_chat_console_script_and_optional_dependency_are_declared() -> None:
    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    assert (
        pyproject["project"]["scripts"]["llm-tools-streamlit-chat"]
        == "llm_tools.apps.streamlit_chat:main"
    )
    assert (
        "streamlit>=1.41,<2"
        in pyproject["project"]["optional-dependencies"]["streamlit"]
    )


def test_streamlit_chat_launch_and_script_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: list[str] = []
    fake_cli = SimpleNamespace(main=lambda: captured.extend(sys.argv) or 0)
    monkeypatch.setitem(sys.modules, "streamlit.web.cli", fake_cli)
    monkeypatch.setitem(sys.modules, "streamlit.web", SimpleNamespace(cli=fake_cli))
    monkeypatch.setitem(
        sys.modules,
        "streamlit",
        SimpleNamespace(web=SimpleNamespace(cli=fake_cli)),
    )

    assert (
        _STREAMLIT_MODULES.app._launch_streamlit_app(["repo", "--config", "chat.yaml"])
        == 0
    )
    assert captured[:3] == [
        "streamlit",
        "run",
        str(Path(_STREAMLIT_MODULES.app.__file__).resolve()),
    ]

    config_path = tmp_path / "chat.yaml"
    config_path.write_text("llm:\n  provider: ollama\n", encoding="utf-8")
    called: list[tuple[Path, object]] = []
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "run_streamlit_chat_app",
        lambda *, root_path, config: called.append((root_path, config)),
    )
    _STREAMLIT_MODULES.app._run_streamlit_script(
        [str(tmp_path), "--config", str(config_path)]
    )
    assert called and called[0][0] == tmp_path.resolve()

    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_launch_streamlit_app",
        lambda script_args: len(script_args),
    )
    assert _STREAMLIT_MODULES.app.main(["a", "b"]) == 2
