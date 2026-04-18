"""Tests for the Streamlit repository chat app layer."""

from __future__ import annotations

import importlib
import queue
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
    ChatWorkflowInspectorEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
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
        self.cancelled = False

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(self._events)

    def resolve_pending_approval(self, approved: bool) -> bool:
        self.resolutions.append(approved)
        return True

    def cancel(self) -> None:
        self.cancelled = True


class _DoneThread:
    def is_alive(self) -> bool:
        return False


class _ThreadSpy:
    def __init__(
        self, target: object | None = None, args: tuple[object, ...] = ()
    ) -> None:
        self.target = target
        self.args = args
        self.started = False

    def start(self) -> None:
        self.started = True

    def is_alive(self) -> bool:
        return self.started


class _ExplodingRunner:
    def __iter__(self):  # type: ignore[no-untyped-def]
        raise RuntimeError("worker boom")


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
        button_values: dict[str, bool] | None = None,
        text_input_values: dict[str, str] | None = None,
        checkbox_values: dict[str, bool] | None = None,
    ) -> None:
        self.session_state: dict[str, object] = {}
        self.chat_input_value = chat_input
        self.button_value = button_value
        self.button_values = button_values or {}
        self.text_input_values = text_input_values or {}
        self.checkbox_values = checkbox_values or {}
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
        self.button_labels: list[str] = []
        self.text_area_values: dict[str, str] = {}
        self.download_calls: list[tuple[str, str, str | None]] = []
        self.checkbox_calls: list[str] = []

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

    def button(
        self,
        label: str,
        *,
        use_container_width: bool = False,
        disabled: bool = False,
    ) -> bool:
        del use_container_width
        self.button_calls += 1
        self.button_labels.append(label)
        if disabled:
            return False
        return self.button_values.get(label, self.button_value)

    def text_input(
        self,
        label: str,
        *,
        value: str = "",
        type: str = "default",
    ) -> str:
        del type
        return self.text_input_values.get(label, value)

    def checkbox(
        self,
        label: str,
        *,
        value: bool = False,
        key: str | None = None,
    ) -> bool:
        token = key or label
        self.checkbox_calls.append(token)
        return self.checkbox_values.get(token, value)

    def text_area(
        self,
        label: str,
        value: str = "",
        *,
        height: int | None = None,
    ) -> str:
        del height
        self.text_area_values[label] = value
        return value

    def download_button(
        self,
        label: str,
        *,
        data: str,
        file_name: str | None = None,
        use_container_width: bool = False,
    ) -> bool:
        del use_container_width
        self.download_calls.append((label, data, file_name))
        return False

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
        value = self.chat_input_value
        self.chat_input_value = None
        return value

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
    assert outcome.transcript_entries[0].text == (
        "Approval requested for search_text: approval required"
    )
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
        "_start_streamlit_turn",
        lambda *, root_path, config, provider, user_message: (
            fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT].append(
                _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
                    role="user",
                    text=user_message,
                )
            ),
            fake_st.session_state.__setitem__(
                _STREAMLIT_MODULES.app._TURN_STATE_SLOT,
                _STREAMLIT_MODULES.app.StreamlitTurnState(
                    busy=True,
                    status_text="thinking",
                    active_turn_number=1,
                ),
            ),
            fake_st.session_state.__setitem__(
                _STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT,
                SimpleNamespace(turn_number=1),
            ),
        ),
    )

    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app.run_streamlit_chat_app(
            root_path=tmp_path,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )

    transcript = fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]
    assert [entry.role for entry in transcript] == ["system", "user"]
    assert transcript[1].text == "question"
    assert created_provider == {
        "provider": created_provider["provider"],
        "provider_name": "ollama",
        "api_key": None,
        "model_name": "gemma4:26b",
    }
    assert fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT].busy is True
    assert fake_st.rerun_called is True


def test_streamlit_chat_apply_error_event_records_turn_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT] = []
    fake_st.session_state[_STREAMLIT_MODULES.app._INSPECTOR_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitInspectorState()
    )
    fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitTurnState(
            busy=True,
            status_text="thinking",
            pending_interrupt_draft="carry",
        )
    )

    _STREAMLIT_MODULES.app._apply_queued_event(
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="error",
            payload="boom",
            turn_number=1,
        )
    )

    transcript = fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]
    assert [entry.role for entry in transcript] == ["error"]
    assert transcript[-1].role == "error"
    assert transcript[-1].text == "boom"
    turn_state = fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT]
    assert turn_state.busy is False
    assert turn_state.pending_interrupt_draft is None


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


def test_streamlit_chat_command_controls_update_session_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    _STREAMLIT_MODULES.app._ensure_session_state(
        tmp_path,
        _STREAMLIT_MODULES.package.TextualChatConfig(),
    )

    for command in ("/tools disable read_file", "/approvals on", "/inspect", "/copy"):
        outcome = _STREAMLIT_MODULES.app._run_streamlit_command(
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
            raw_command=command,
        )
        _STREAMLIT_MODULES.app._apply_streamlit_command(outcome)

    exit_outcome = _STREAMLIT_MODULES.app._run_streamlit_command(
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        raw_command="quit",
    )
    _STREAMLIT_MODULES.app._apply_streamlit_command(exit_outcome)

    control_state = fake_st.session_state[_STREAMLIT_MODULES.app._CONTROL_STATE_SLOT]
    assert "read_file" not in control_state.enabled_tools
    assert SideEffectClass.LOCAL_READ in control_state.require_approval_for
    assert control_state.inspector_open is True
    assert (
        fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_EXPORT_STATE_SLOT]
        is True
    )
    transcript = fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]
    assert transcript[-1].text.startswith("Streamlit chat keeps running.")


def test_streamlit_chat_state_and_api_key_helpers_cover_session_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(sidebar_text_input=" typed-secret ")
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_STREAMLIT_MODULES.app.os, "getenv", lambda name: None)

    config = _STREAMLIT_MODULES.package.TextualChatConfig().model_copy(
        update={
            "llm": _STREAMLIT_MODULES.package.TextualChatConfig().llm.model_copy(
                update={"provider": _STREAMLIT_MODULES.package.ProviderPreset.OPENAI}
            )
        }
    )
    _STREAMLIT_MODULES.app._ensure_session_state(tmp_path, config)
    assert (
        fake_st.session_state[
            _STREAMLIT_MODULES.app._TURN_STATE_SLOT
        ].active_turn_number
        == 0
    )
    assert (
        fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_EXPORT_STATE_SLOT]
        is False
    )

    transcript_entry = _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
        role="assistant",
        text="Answer",
        final_response=ChatFinalResponse(answer="Answer"),
    )
    assert transcript_entry.transcript_text.startswith("Assistant:")

    runner = _FakeRunner([])
    fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
            runner=runner,
            event_queue=queue.Queue(),
            thread=_DoneThread(),  # type: ignore[arg-type]
            turn_number=1,
        )
    )
    _STREAMLIT_MODULES.app._reset_session_state(tmp_path, config)
    assert runner.cancelled is True

    api_key = _STREAMLIT_MODULES.app._resolve_api_key(config)
    assert api_key == "typed-secret"
    assert (
        fake_st.session_state[_STREAMLIT_MODULES.app._API_SECRET_STATE_SLOT]
        == "typed-secret"
    )
    assert _STREAMLIT_MODULES.app._current_api_key(config) == "typed-secret"
    assert (
        _STREAMLIT_MODULES.app._current_api_key(
            _STREAMLIT_MODULES.package.TextualChatConfig()
        )
        is None
    )


def test_streamlit_chat_drains_background_events_and_updates_inspector(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    _STREAMLIT_MODULES.app._ensure_session_state(
        tmp_path,
        _STREAMLIT_MODULES.package.TextualChatConfig(),
    )

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
    event_queue: queue.Queue[object] = queue.Queue()
    for queued_event in (
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="status",
            payload=ChatWorkflowStatusEvent(status="thinking").model_dump(mode="json"),
            turn_number=1,
        ),
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="approval_requested",
            payload=ChatWorkflowApprovalEvent(approval=approval_state).model_dump(
                mode="json"
            ),
            turn_number=1,
        ),
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="inspector",
            payload=ChatWorkflowInspectorEvent(
                round_index=1,
                kind="parsed_response",
                payload={"ok": True},
            ).model_dump(mode="json"),
            turn_number=1,
        ),
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="approval_resolved",
            payload=ChatWorkflowApprovalResolvedEvent(
                approval=approval_state,
                resolution="approved",
            ).model_dump(mode="json"),
            turn_number=1,
        ),
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="result",
            payload=ChatWorkflowResultEvent(
                result=ChatWorkflowTurnResult(
                    status="completed",
                    final_response=ChatFinalResponse(answer="Answer"),
                    session_state=ChatSessionState(),
                    token_usage=ChatTokenUsage(
                        session_tokens=7,
                        active_context_tokens=9,
                    ),
                )
            ).model_dump(mode="json"),
            turn_number=1,
        ),
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="complete",
            payload=None,
            turn_number=1,
        ),
    ):
        event_queue.put(queued_event)

    fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitTurnState(busy=True, active_turn_number=1)
    )
    fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
            runner=_FakeRunner([]),
            event_queue=event_queue,
            thread=_DoneThread(),  # type: ignore[arg-type]
            turn_number=1,
        )
    )

    assert _STREAMLIT_MODULES.app._drain_active_turn_events() is None
    transcript = fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]
    assert transcript[-1].final_response is not None
    inspector_state = fake_st.session_state[
        _STREAMLIT_MODULES.app._INSPECTOR_STATE_SLOT
    ]
    assert inspector_state.parsed_responses[0].label == "Turn 1 Round 1 parsed response"
    assert fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] is None


def test_streamlit_chat_low_level_worker_and_queue_helpers_cover_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    _STREAMLIT_MODULES.app._ensure_session_state(
        tmp_path,
        _STREAMLIT_MODULES.package.TextualChatConfig(),
    )

    result_event = ChatWorkflowResultEvent(
        result=ChatWorkflowTurnResult(
            status="needs_continuation",
            continuation_reason="Need more budget.",
            context_warning="Older turns were removed.",
            session_state=ChatSessionState(),
            token_usage=ChatTokenUsage(session_tokens=3, active_context_tokens=4),
        )
    )
    fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitTurnState(
            busy=True,
            pending_interrupt_draft="follow-up",
        )
    )
    pending_prompt = _STREAMLIT_MODULES.app._apply_turn_result(result_event)
    assert pending_prompt == "follow-up"
    transcript = fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]
    assert transcript[-2].text == "Older turns were removed."
    assert transcript[-1].text == "Need more budget."

    interrupted_event = ChatWorkflowResultEvent(
        result=ChatWorkflowTurnResult(
            status="interrupted",
            interruption_reason="Stopped by user.",
            new_messages=[],
        )
    )
    fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitTurnState(busy=True)
    )
    _STREAMLIT_MODULES.app._apply_turn_result(interrupted_event)
    assert transcript[-1].text == "Stopped by user."

    handle = _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
        runner=_FakeRunner([]),
        event_queue=queue.Queue(),
        thread=_DoneThread(),  # type: ignore[arg-type]
        turn_number=7,
    )
    fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] = handle
    fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitTurnState(busy=False)
    )
    _STREAMLIT_MODULES.app._apply_queued_event(
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="complete",
            payload=None,
            turn_number=7,
        )
    )
    assert fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] is None

    queued_result = ChatWorkflowResultEvent(
        result=ChatWorkflowTurnResult(
            status="completed",
            final_response=ChatFinalResponse(answer="Queued answer"),
            session_state=ChatSessionState(),
        )
    )
    event_queue: queue.Queue[object] = queue.Queue()
    event_queue.put(
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="result",
            payload=queued_result.model_dump(mode="json"),
            turn_number=9,
        )
    )
    fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitTurnState(
            busy=True,
            pending_interrupt_draft="queued prompt",
        )
    )
    fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
            runner=_FakeRunner([]),
            event_queue=event_queue,
            thread=_DoneThread(),  # type: ignore[arg-type]
            turn_number=9,
        )
    )
    assert _STREAMLIT_MODULES.app._drain_active_turn_events() == "queued prompt"

    serialized = _STREAMLIT_MODULES.app._serialize_workflow_event(
        ChatWorkflowStatusEvent(status="thinking"),
        turn_number=3,
    )
    assert serialized.kind == "status"
    with pytest.raises(TypeError):
        _STREAMLIT_MODULES.app._serialize_workflow_event(object(), turn_number=3)

    success_queue: queue.Queue[object] = queue.Queue()
    success_handle = _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
        runner=_FakeRunner([ChatWorkflowStatusEvent(status="thinking")]),
        event_queue=success_queue,
        thread=_DoneThread(),  # type: ignore[arg-type]
        turn_number=4,
    )
    _STREAMLIT_MODULES.app._worker_run_turn(success_handle)
    assert success_queue.get_nowait().kind == "status"
    assert success_queue.get_nowait().kind == "complete"

    error_queue: queue.Queue[object] = queue.Queue()
    error_handle = _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
        runner=_ExplodingRunner(),  # type: ignore[arg-type]
        event_queue=error_queue,
        thread=_DoneThread(),  # type: ignore[arg-type]
        turn_number=5,
    )
    _STREAMLIT_MODULES.app._worker_run_turn(error_handle)
    assert error_queue.get_nowait().kind == "error"
    assert error_queue.get_nowait().kind == "complete"


def test_streamlit_chat_busy_prompt_interrupts_active_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(chat_input="replacement")
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_STREAMLIT_MODULES.app.os, "getenv", lambda name: "env-secret")
    _STREAMLIT_MODULES.app._ensure_session_state(
        tmp_path,
        _STREAMLIT_MODULES.package.TextualChatConfig(),
    )
    runner = _FakeRunner([])
    fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitTurnState(busy=True, active_turn_number=1)
    )
    fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
            runner=runner,
            event_queue=queue.Queue(),
            thread=_DoneThread(),  # type: ignore[arg-type]
            turn_number=1,
        )
    )

    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app.run_streamlit_chat_app(
            root_path=tmp_path,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )

    turn_state = fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT]
    assert turn_state.pending_interrupt_draft == "replacement"
    assert turn_state.status_text == "stopping"
    assert runner.cancelled is True


def test_streamlit_chat_turn_lifecycle_and_command_wrappers_cover_remaining_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(
        text_input_values={"Switch model": "model-2"},
        checkbox_values={
            "tool:read_file": False,
            "transcript_export": True,
            "show_inspector": True,
            "approvals:local_read": True,
        },
    )
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_STREAMLIT_MODULES.app.os, "getenv", lambda name: "env-secret")
    _STREAMLIT_MODULES.app._ensure_session_state(
        tmp_path,
        _STREAMLIT_MODULES.package.TextualChatConfig(),
    )

    created_threads: list[_ThreadSpy] = []

    def _thread_factory(*args: object, **kwargs: object) -> _ThreadSpy:
        thread = _ThreadSpy(
            target=kwargs.get("target"),
            args=kwargs.get("args", ()),
        )
        created_threads.append(thread)
        return thread

    monkeypatch.setattr(_STREAMLIT_MODULES.app.threading, "Thread", _thread_factory)
    fake_runner = _FakeRunner([])
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_build_chat_runner", lambda **kwargs: fake_runner
    )
    _STREAMLIT_MODULES.app._start_streamlit_turn(
        root_path=tmp_path,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        provider=object(),
        user_message="hello",
    )
    assert created_threads[-1].started is True
    turn_state = fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT]
    assert turn_state.busy is True
    assert (
        fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT][-1].text
        == "hello"
    )

    approval_state = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-2",
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {"path": "x"}},
            tool_name="read_file",
            tool_version="0.1.0",
            policy_reason="approval required",
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:01:00Z",
        ),
        tool_name="read_file",
        redacted_arguments={"path": "x"},
        policy_reason="approval required",
    )
    turn_state.pending_approval = approval_state
    _STREAMLIT_MODULES.app._resolve_active_approval(approved=True)
    assert turn_state.approval_decision_in_flight is True
    _STREAMLIT_MODULES.app._cancel_active_turn()
    assert fake_runner.cancelled is True
    turn_state.busy = False
    turn_state.pending_approval = None

    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "create_provider",
        lambda config, api_key, model_name: SimpleNamespace(
            model=model_name,
            list_available_models=lambda: ["model-1", "model-2"],
        ),
    )
    outcome = _STREAMLIT_MODULES.app._run_streamlit_command(
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        raw_command="/model",
    )
    assert outcome.handled is True
    assert "Available models" in outcome.notices[0].text

    switch_outcome = _STREAMLIT_MODULES.app._run_streamlit_command(
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        raw_command="/model switched-model",
    )
    assert switch_outcome.provider is not None

    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "create_provider",
        lambda config, api_key, model_name: (_ for _ in ()).throw(
            RuntimeError("bad provider")
        ),
    )
    error_outcome = _STREAMLIT_MODULES.app._run_streamlit_command(
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        raw_command="/model broken-model",
    )
    assert error_outcome.notices[0].role == "error"

    fake_st.session_state[_STREAMLIT_MODULES.app._TOKEN_USAGE_STATE_SLOT] = (
        ChatTokenUsage(session_tokens=10, active_context_tokens=20)
    )
    fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT].confidence = 0.8
    with pytest.raises(_RerunRequestError):
        fake_st.button_values["Reset tools to defaults"] = True
        _STREAMLIT_MODULES.app._render_sidebar_tool_controls(
            fake_st.session_state[_STREAMLIT_MODULES.app._CONTROL_STATE_SLOT]
        )
    fake_st.button_values.clear()

    with pytest.raises(_RerunRequestError):
        fake_st.button_values["List available models"] = True
        monkeypatch.setattr(
            _STREAMLIT_MODULES.app,
            "_run_streamlit_command",
            lambda **kwargs: _STREAMLIT_MODULES.app.ChatCommandOutcome(handled=True),
        )
        _STREAMLIT_MODULES.app._render_sidebar_model_controls(
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
            control_state=fake_st.session_state[
                _STREAMLIT_MODULES.app._CONTROL_STATE_SLOT
            ],
        )
    fake_st.button_values.clear()

    fake_st.session_state[
        _STREAMLIT_MODULES.app._TURN_STATE_SLOT
    ].pending_approval = approval_state
    fake_st.session_state[
        _STREAMLIT_MODULES.app._TURN_STATE_SLOT
    ].approval_decision_in_flight = False
    with pytest.raises(_RerunRequestError):
        fake_st.button_values["Approve pending request"] = True
        _STREAMLIT_MODULES.app._render_sidebar_approval_controls(
            control_state=fake_st.session_state[
                _STREAMLIT_MODULES.app._CONTROL_STATE_SLOT
            ],
            turn_state=fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT],
        )
    fake_st.button_values.clear()

    _STREAMLIT_MODULES.app._render_sidebar_transcript_export()
    assert "Transcript export" in fake_st.text_area_values
    assert fake_st.download_calls[-1][0] == "Download transcript"

    _STREAMLIT_MODULES.app._render_sidebar_inspector(
        control_state=fake_st.session_state[_STREAMLIT_MODULES.app._CONTROL_STATE_SLOT],
        turn_state=fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT],
    )
    assert any("Tool Execution Records" in text for text in fake_st.caption_messages)

    session_state = fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT]
    session_state.busy = True
    with pytest.raises(_RerunRequestError):
        fake_st.button_values["Refresh active turn"] = True
        _STREAMLIT_MODULES.app._render_sidebar_session(
            root_path=tmp_path,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
            control_state=fake_st.session_state[
                _STREAMLIT_MODULES.app._CONTROL_STATE_SLOT
            ],
            turn_state=session_state,
        )


def test_streamlit_chat_additional_helper_and_command_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(
        text_input_values={"Switch model": "model-3"},
        checkbox_values={
            "show_inspector": False,
            "transcript_export": False,
            "approvals:local_read": False,
        },
    )
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_STREAMLIT_MODULES.app.os, "getenv", lambda name: None)
    _STREAMLIT_MODULES.app._ensure_session_state(
        tmp_path,
        _STREAMLIT_MODULES.package.TextualChatConfig(),
    )

    config_path = tmp_path / "streamlit-chat.yaml"
    config_path.write_text("llm:\n  model_name: base-model\n", encoding="utf-8")
    args = build_parser().parse_args(
        [
            str(tmp_path),
            "--config",
            str(config_path),
            "--provider",
            "openai",
            "--model",
            "override-model",
            "--temperature",
            "0.25",
            "--api-base-url",
            "http://example.invalid/v1",
            "--max-read-lines",
            "50",
        ]
    )
    resolved = _resolve_chat_config(args)
    assert resolved.llm.provider.value == "openai"
    assert resolved.llm.temperature == 0.25
    assert resolved.llm.api_base_url == "http://example.invalid/v1"
    assert resolved.tool_limits.max_read_lines == 50

    monkeypatch.setitem(sys.modules, "streamlit", fake_st)
    assert _STREAMLIT_MODULES.app._streamlit_module() is fake_st

    fake_st.session_state[_STREAMLIT_MODULES.app._API_SECRET_STATE_SLOT] = (
        "cached-secret"
    )
    openai_config = _STREAMLIT_MODULES.package.TextualChatConfig().model_copy(
        update={
            "llm": _STREAMLIT_MODULES.package.TextualChatConfig().llm.model_copy(
                update={"provider": _STREAMLIT_MODULES.package.ProviderPreset.OPENAI}
            )
        }
    )
    assert _STREAMLIT_MODULES.app._current_api_key(openai_config) == "cached-secret"
    assert _STREAMLIT_MODULES.app._resolve_api_key(openai_config) == "cached-secret"

    approval_state = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-4",
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {"path": "x"}},
            tool_name="read_file",
            tool_version="0.1.0",
            policy_reason="approval required",
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:01:00Z",
        ),
        tool_name="read_file",
        redacted_arguments={"path": "x"},
        policy_reason="approval required",
    )
    inspector_event = ChatWorkflowInspectorEvent(
        round_index=2,
        kind="provider_messages",
        payload={"messages": []},
    )
    assert (
        _STREAMLIT_MODULES.app._serialize_workflow_event(
            ChatWorkflowApprovalEvent(approval=approval_state),
            turn_number=2,
        ).kind
        == "approval_requested"
    )
    assert (
        _STREAMLIT_MODULES.app._serialize_workflow_event(
            ChatWorkflowApprovalResolvedEvent(
                approval=approval_state,
                resolution="timed_out",
            ),
            turn_number=2,
        ).kind
        == "approval_resolved"
    )
    assert (
        _STREAMLIT_MODULES.app._serialize_workflow_event(
            inspector_event,
            turn_number=2,
        ).kind
        == "inspector"
    )
    assert (
        _STREAMLIT_MODULES.app._serialize_workflow_event(
            ChatWorkflowResultEvent(
                result=ChatWorkflowTurnResult(
                    status="completed",
                    final_response=ChatFinalResponse(answer="ok"),
                )
            ),
            turn_number=2,
        ).kind
        == "result"
    )

    fake_st.session_state[_STREAMLIT_MODULES.app._API_SECRET_STATE_SLOT] = ""
    missing_key_list = _STREAMLIT_MODULES.app._run_streamlit_command(
        config=openai_config,
        raw_command="/model",
    )
    missing_key_switch = _STREAMLIT_MODULES.app._run_streamlit_command(
        config=openai_config,
        raw_command="/model model-4",
    )
    assert missing_key_list.notices[0].text.startswith("Set OPENAI_API_KEY")
    assert missing_key_switch.notices[0].text.startswith("Set OPENAI_API_KEY")

    fake_st.session_state[_STREAMLIT_MODULES.app._API_SECRET_STATE_SLOT] = "env-secret"
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "create_provider",
        lambda config, api_key, model_name: SimpleNamespace(
            list_available_models=lambda: (_ for _ in ()).throw(
                RuntimeError("cannot list")
            )
        ),
    )
    model_notice = _STREAMLIT_MODULES.app._run_streamlit_command(
        config=openai_config,
        raw_command="/model",
    )
    assert "Unable to list available models" in model_notice.notices[0].text

    rejecting_runner = SimpleNamespace(
        resolve_pending_approval=lambda approved: False,
        cancel=lambda: None,
    )
    fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
            runner=rejecting_runner,  # type: ignore[arg-type]
            event_queue=queue.Queue(),
            thread=_DoneThread(),  # type: ignore[arg-type]
            turn_number=3,
        )
    )
    turn_state = fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT]
    _STREAMLIT_MODULES.app._resolve_active_approval(approved=True)
    assert turn_state.approval_decision_in_flight is False

    cancelling_runner = _FakeRunner([])
    fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
            runner=cancelling_runner,
            event_queue=queue.Queue(),
            thread=_DoneThread(),  # type: ignore[arg-type]
            turn_number=4,
        )
    )
    turn_state.pending_interrupt_draft = "draft"
    _STREAMLIT_MODULES.app._cancel_active_turn()
    assert turn_state.pending_interrupt_draft is None
    assert cancelling_runner.cancelled is True

    transcript_before = list(
        fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]
    )
    _STREAMLIT_MODULES.app._submit_streamlit_prompt(
        root_path=tmp_path,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        prompt="   ",
    )
    assert (
        fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]
        == transcript_before
    )

    turn_state.busy = True
    with pytest.raises(_RerunRequestError):
        fake_st.button_values["Stop active turn"] = True
        _STREAMLIT_MODULES.app._render_sidebar_session(
            root_path=tmp_path,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
            control_state=fake_st.session_state[
                _STREAMLIT_MODULES.app._CONTROL_STATE_SLOT
            ],
            turn_state=turn_state,
        )
    fake_st.button_values.clear()

    with pytest.raises(_RerunRequestError):
        fake_st.button_values["Switch model"] = True
        monkeypatch.setattr(
            _STREAMLIT_MODULES.app,
            "_run_streamlit_command",
            lambda **kwargs: _STREAMLIT_MODULES.app.ChatCommandOutcome(handled=True),
        )
        _STREAMLIT_MODULES.app._render_sidebar_model_controls(
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
            control_state=fake_st.session_state[
                _STREAMLIT_MODULES.app._CONTROL_STATE_SLOT
            ],
        )
    fake_st.button_values.clear()

    turn_state.busy = False
    turn_state.pending_approval = approval_state
    turn_state.approval_decision_in_flight = False
    fake_st.session_state[
        _STREAMLIT_MODULES.app._CONTROL_STATE_SLOT
    ].require_approval_for.add(SideEffectClass.LOCAL_READ)
    with pytest.raises(_RerunRequestError):
        fake_st.button_values["Deny pending request"] = True
        _STREAMLIT_MODULES.app._render_sidebar_approval_controls(
            control_state=fake_st.session_state[
                _STREAMLIT_MODULES.app._CONTROL_STATE_SLOT
            ],
            turn_state=turn_state,
        )
    assert (
        SideEffectClass.LOCAL_READ
        not in fake_st.session_state[
            _STREAMLIT_MODULES.app._CONTROL_STATE_SLOT
        ].require_approval_for
    )
    fake_st.button_values.clear()

    _STREAMLIT_MODULES.app._render_sidebar_transcript_export()
    assert fake_st.download_calls == []

    _STREAMLIT_MODULES.app._render_sidebar_inspector(
        control_state=fake_st.session_state[_STREAMLIT_MODULES.app._CONTROL_STATE_SLOT],
        turn_state=turn_state,
    )

    fake_st.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitTurnState()
    )
    with pytest.raises(ValueError):
        _STREAMLIT_MODULES.app._apply_queued_event(
            _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
                kind="bogus", payload=None, turn_number=1
            )
        )


def test_streamlit_chat_run_app_and_reducer_extra_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(chat_input="/help")
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(_STREAMLIT_MODULES.app.os, "getenv", lambda name: "env-secret")

    submitted_prompts: list[str] = []
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_drain_active_turn_events",
        lambda: "resumed prompt",
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_submit_streamlit_prompt",
        lambda *, root_path, config, prompt: submitted_prompts.append(prompt),
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_run_streamlit_command",
        lambda **kwargs: _STREAMLIT_MODULES.app.ChatCommandOutcome(
            handled=True,
            notices=[
                _STREAMLIT_MODULES.app.ChatControlNotice(role="system", text="help")
            ],
        ),
    )

    config = _STREAMLIT_MODULES.package.TextualChatConfig().model_copy(
        update={
            "ui": _STREAMLIT_MODULES.package.TextualChatConfig().ui.model_copy(
                update={"show_footer_help": False}
            )
        }
    )
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app.run_streamlit_chat_app(root_path=tmp_path, config=config)

    assert submitted_prompts == ["resumed prompt"]
    assert not any(
        "Use /help for controls" in text for text in fake_st.caption_messages
    )
    transcript = fake_st.session_state[_STREAMLIT_MODULES.app._TRANSCRIPT_STATE_SLOT]
    assert transcript[-1].text == "help"

    busy_streamlit = _FakeStreamlit()
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: busy_streamlit
    )
    monkeypatch.setattr(_STREAMLIT_MODULES.app.os, "getenv", lambda name: "env-secret")
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_drain_active_turn_events", lambda: None
    )
    monkeypatch.setattr(_STREAMLIT_MODULES.app.time, "sleep", lambda seconds: None)
    _STREAMLIT_MODULES.app._ensure_session_state(
        tmp_path,
        _STREAMLIT_MODULES.package.TextualChatConfig(),
    )
    busy_streamlit.session_state[_STREAMLIT_MODULES.app._TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitTurnState(busy=True)
    )
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app.run_streamlit_chat_app(
            root_path=tmp_path,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )

    monkeypatch.setattr(
        "llm_tools.apps.streamlit_chat.app.run_interactive_chat_session_turn",
        lambda **kwargs: _FakeRunner(
            [
                object(),
                ChatWorkflowResultEvent(
                    result=ChatWorkflowTurnResult(
                        status="interrupted",
                        interruption_reason="Stopped by user.",
                        new_messages=[
                            {
                                "role": "assistant",
                                "content": "partial answer",
                                "completion_state": "interrupted",
                            }
                        ],
                    )
                ),
            ]
        ),
    )
    outcome = process_streamlit_chat_turn(
        root_path=tmp_path,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        provider=object(),
        session_state=ChatSessionState(),
        user_message="interrupt",
    )
    assert outcome.transcript_entries[-1].assistant_completion_state == "interrupted"
    assert outcome.transcript_entries[-1].text == "partial answer"

    render_streamlit = _FakeStreamlit()
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: render_streamlit
    )
    _STREAMLIT_MODULES.app._render_transcript_entry(
        _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
            role="assistant",
            text="partial answer",
            assistant_completion_state="interrupted",
        )
    )
    _STREAMLIT_MODULES.app._render_transcript_entry(
        _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
            role="assistant",
            text="final",
            final_response=ChatFinalResponse(answer="final"),
        )
    )
    assert "Assistant (interrupted)" in render_streamlit.caption_messages


def test_streamlit_chat_process_turn_covers_status_inspector_and_interruption_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    approval_state = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-3",
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {"path": "x"}},
            tool_name="read_file",
            tool_version="0.1.0",
            policy_reason="approval required",
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:01:00Z",
        ),
        tool_name="read_file",
        redacted_arguments={"path": "x"},
        policy_reason="approval required",
    )
    monkeypatch.setattr(
        "llm_tools.apps.streamlit_chat.app.run_interactive_chat_session_turn",
        lambda **kwargs: _FakeRunner(
            [
                ChatWorkflowStatusEvent(status="thinking"),
                ChatWorkflowInspectorEvent(
                    round_index=1,
                    kind="tool_execution",
                    payload={"tool_name": "read_file"},
                ),
                ChatWorkflowApprovalEvent(approval=approval_state),
                ChatWorkflowApprovalResolvedEvent(
                    approval=approval_state,
                    resolution="cancelled",
                ),
                ChatWorkflowResultEvent(
                    result=ChatWorkflowTurnResult(
                        status="interrupted",
                        interruption_reason="Approval cancelled.",
                        new_messages=[],
                    )
                ),
            ]
        ),
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
        approval_resolver=lambda approval: True,
    )

    assert (
        outcome.transcript_entries[1].text == "Pending approval request was cancelled."
    )
    assert outcome.transcript_entries[-1].text == "Approval cancelled."


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
