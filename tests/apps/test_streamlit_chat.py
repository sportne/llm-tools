"""Tests for the redesigned Streamlit chat app layer."""

from __future__ import annotations

import importlib
import runpy
import sys
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest
from tests.apps._imports import import_streamlit_chat_modules

from llm_tools.apps.chat_controls import ChatCommandOutcome, ChatControlNotice
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import SideEffectClass, ToolContext
from llm_tools.tools._path_utils import get_workspace_root
from llm_tools.workflow_api import (
    ApprovalRequest,
    ChatFinalResponse,
    ChatMessage,
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

_STREAMLIT_MODULES = import_streamlit_chat_modules()
_CHAT_RUNTIME_MODULE = importlib.import_module("llm_tools.apps.chat_runtime")
_CHAT_PROMPTS_MODULE = importlib.import_module("llm_tools.apps.chat_prompts")
build_parser = _STREAMLIT_MODULES.app.build_parser
process_streamlit_chat_turn = _STREAMLIT_MODULES.app.process_streamlit_chat_turn
resolve_enabled_tool_names = _STREAMLIT_MODULES.app.resolve_enabled_tool_names
_resolve_chat_config = _STREAMLIT_MODULES.app._resolve_chat_config


class _FakeProvider:
    def __init__(self, responses: list[ParsedModelResponse]) -> None:
        self._responses = list(responses)

    def run(self, **kwargs: object) -> ParsedModelResponse:
        del kwargs
        return self._responses.pop(0)


class _RerunRequestError(RuntimeError):
    pass


class _FakeBlock:
    def __init__(self, streamlit: _FakeStreamlit) -> None:
        self._streamlit = streamlit

    def __enter__(self) -> _FakeBlock:
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> bool:
        return False

    def button(self, label: str, **kwargs: object) -> bool:
        return self._streamlit.button(label, **kwargs)

    def text_input(self, label: str, **kwargs: object) -> str:
        return self._streamlit.text_input(label, **kwargs)

    def text_area(self, label: str, *args: object, **kwargs: object) -> str:
        return self._streamlit.text_area(label, *args, **kwargs)

    def checkbox(self, label: str, **kwargs: object) -> bool:
        return self._streamlit.checkbox(label, **kwargs)

    def toggle(self, label: str, **kwargs: object) -> bool:
        return self._streamlit.toggle(label, **kwargs)

    def selectbox(self, label: str, **kwargs: object) -> object:
        return self._streamlit.selectbox(label, **kwargs)

    def columns(self, spec: int | list[int]) -> list[_FakeBlock]:
        return self._streamlit.columns(spec)

    def markdown(self, text: str, **kwargs: object) -> None:
        self._streamlit.markdown(text, **kwargs)

    def caption(self, text: str) -> None:
        self._streamlit.caption(text)

    def code(self, text: str) -> None:
        self._streamlit.code(text)

    def warning(self, text: str) -> None:
        self._streamlit.warning(text)

    def error(self, text: str) -> None:
        self._streamlit.error(text)

    def form_submit_button(self, label: str, **kwargs: object) -> bool:
        return self._streamlit.form_submit_button(label, **kwargs)

    def download_button(self, label: str, **kwargs: object) -> bool:
        return self._streamlit.download_button(label, **kwargs)

    def tabs(self, names: list[str]) -> list[_FakeBlock]:
        return self._streamlit.tabs(names)


class _FakeStreamlit:
    def __init__(
        self,
        *,
        button_values: dict[str, bool] | None = None,
        text_input_values: dict[str, str] | None = None,
        checkbox_values: dict[str, bool] | None = None,
        selectbox_values: dict[str, object] | None = None,
    ) -> None:
        self.session_state: dict[str, object] = {}
        self.button_values = button_values or {}
        self.text_input_values = text_input_values or {}
        self.checkbox_values = checkbox_values or {}
        self.selectbox_values = selectbox_values or {}
        self.sidebar = _FakeBlock(self)
        self.page_config_kwargs: list[dict[str, object]] = []
        self.markdown_messages: list[str] = []
        self.caption_messages: list[str] = []
        self.warning_messages: list[str] = []
        self.error_messages: list[str] = []
        self.button_labels: list[str] = []
        self.download_calls: list[tuple[str, str, str | None]] = []
        self.text_area_values: dict[str, str] = {}
        self.chat_roles: list[str] = []
        self.rerun_called = False

    def set_page_config(self, **kwargs: object) -> None:
        self.page_config_kwargs.append(kwargs)

    def markdown(self, text: str, unsafe_allow_html: bool = False) -> None:
        del unsafe_allow_html
        self.markdown_messages.append(text)

    def caption(self, text: str) -> None:
        self.caption_messages.append(text)

    def warning(self, text: str) -> None:
        self.warning_messages.append(text)

    def info(self, text: str) -> None:
        self.caption_messages.append(text)

    def error(self, text: str) -> None:
        self.error_messages.append(text)

    def code(self, text: str) -> None:
        self.markdown_messages.append(text)

    def button(
        self,
        label: str,
        *,
        key: str | None = None,
        use_container_width: bool = False,
        disabled: bool = False,
    ) -> bool:
        del use_container_width
        self.button_labels.append(label)
        if disabled:
            return False
        token = key or label
        return self.button_values.get(token, self.button_values.get(label, False))

    def text_input(
        self,
        label: str,
        *,
        value: str = "",
        key: str | None = None,
        disabled: bool = False,
        placeholder: str | None = None,
        type: str = "default",
    ) -> str:
        del disabled, placeholder, type
        token = key or label
        return self.text_input_values.get(
            token, self.text_input_values.get(label, value)
        )

    def text_area(
        self,
        label: str,
        value: str = "",
        *,
        key: str | None = None,
        height: int | None = None,
        placeholder: str | None = None,
        label_visibility: str | None = None,
    ) -> str:
        del height, placeholder, label_visibility
        token = key or label
        resolved = self.text_input_values.get(
            token, self.text_input_values.get(label, value)
        )
        self.text_area_values[token] = str(resolved)
        self.session_state[token] = str(resolved)
        return str(resolved)

    def checkbox(
        self,
        label: str,
        *,
        value: bool = False,
        key: str | None = None,
        disabled: bool = False,
    ) -> bool:
        if disabled:
            return value
        token = key or label
        return self.checkbox_values.get(token, self.checkbox_values.get(label, value))

    def toggle(
        self,
        label: str,
        *,
        value: bool = False,
        key: str | None = None,
        disabled: bool = False,
    ) -> bool:
        if disabled:
            return value
        token = key or label
        return self.checkbox_values.get(token, self.checkbox_values.get(label, value))

    def selectbox(
        self,
        label: str,
        *,
        options: list[object],
        index: int = 0,
        key: str | None = None,
        disabled: bool = False,
        format_func=None,
    ) -> object:
        del format_func
        if disabled:
            return options[index]
        token = key or label
        return self.selectbox_values.get(
            token, self.selectbox_values.get(label, options[index])
        )

    def columns(self, spec: int | list[int]) -> list[_FakeBlock]:
        count = spec if isinstance(spec, int) else len(spec)
        return [_FakeBlock(self) for _ in range(count)]

    def tabs(self, names: list[str]) -> list[_FakeBlock]:
        return [_FakeBlock(self) for _ in names]

    def expander(self, label: str, expanded: bool = False) -> _FakeBlock:
        del expanded
        self.button_labels.append(label)
        return _FakeBlock(self)

    def popover(self, label: str) -> _FakeBlock:
        del label
        return _FakeBlock(self)

    def container(self) -> _FakeBlock:
        return _FakeBlock(self)

    def chat_message(self, role: str) -> _FakeBlock:
        self.chat_roles.append(role)
        return _FakeBlock(self)

    def form(self, key: str, **kwargs: object) -> _FakeBlock:
        del key, kwargs
        return _FakeBlock(self)

    def form_submit_button(
        self,
        label: str,
        *,
        key: str | None = None,
        use_container_width: bool = False,
        shortcut: str | None = None,
        disabled: bool = False,
        **kwargs: object,
    ) -> bool:
        del use_container_width, shortcut, kwargs
        return self.button(label, key=key, disabled=disabled)

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

    def rerun(self) -> None:
        self.rerun_called = True
        raise _RerunRequestError()


def _make_runtime(*, root_path: str | None = None):
    return _STREAMLIT_MODULES.app.StreamlitRuntimeConfig(
        provider=_STREAMLIT_MODULES.package.ProviderPreset.OLLAMA,
        model_name="gemma4:26b",
        api_base_url="http://127.0.0.1:11434/v1",
        root_path=root_path,
        enabled_tools=["read_file", "search_text"] if root_path is not None else [],
    )


def _make_record(*, session_id: str = "session-1", root_path: str | None = None):
    runtime = _make_runtime(root_path=root_path)
    return _STREAMLIT_MODULES.app._new_session_record(session_id, runtime)


def _make_app_state(*, session_id: str = "session-1", root_path: str | None = None):
    record = _make_record(session_id=session_id, root_path=root_path)
    return _STREAMLIT_MODULES.app.StreamlitWorkspaceState(
        sessions={session_id: record},
        session_order=[session_id],
        active_session_id=session_id,
        preferences=_STREAMLIT_MODULES.app.StreamlitPreferences(),
        turn_states={session_id: _STREAMLIT_MODULES.app.StreamlitTurnState()},
    )


class _DeadThread:
    def is_alive(self) -> bool:
        return False


class _FakeRunnerHandle:
    def __init__(self) -> None:
        self.cancelled = False
        self.approvals: list[bool] = []

    def cancel(self) -> None:
        self.cancelled = True

    def resolve_pending_approval(self, approved: bool) -> bool:
        self.approvals.append(approved)
        return True


def test_streamlit_chat_package_imports_without_loading_streamlit() -> None:
    module = importlib.import_module("llm_tools.apps.streamlit_chat")
    main_module = importlib.import_module("llm_tools.apps.streamlit_chat.__main__")

    assert hasattr(module, "main")
    assert hasattr(module, "run_streamlit_chat_app")
    assert hasattr(main_module, "main")


def test_streamlit_chat_config_loading_and_optional_cli(tmp_path: Path) -> None:
    config_path = tmp_path / "chat.yaml"
    config_path.write_text(
        "llm:\n  provider: ollama\n  model_name: base-model\n",
        encoding="utf-8",
    )

    args = build_parser().parse_args(
        ["--config", str(config_path), "--model", "override-model"]
    )
    resolved = _resolve_chat_config(args)
    assert resolved.llm.model_name == "override-model"
    assert _STREAMLIT_MODULES.app._resolve_root_argument(args) is None

    no_config = build_parser().parse_args([])
    default_config = _resolve_chat_config(no_config)
    assert default_config.llm.model_name


def test_chat_runtime_builds_full_tool_catalog_and_optional_root_context(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _STREAMLIT_MODULES.package.TextualChatConfig().model_copy(
        update={
            "session": _STREAMLIT_MODULES.package.TextualChatConfig().session.model_copy(
                update={"max_context_tokens": 10}
            )
        }
    )
    context = _CHAT_RUNTIME_MODULE.build_chat_context(
        root_path=None,
        config=config,
        app_name="streamlit-chat",
    )
    specs = _CHAT_RUNTIME_MODULE.build_available_tool_specs()
    assert context.workspace is None
    assert "write_file" in specs
    assert "run_git_status" in specs
    assert "search_jira" in specs
    prompt = _CHAT_PROMPTS_MODULE.build_chat_system_prompt(
        tool_registry=_CHAT_RUNTIME_MODULE.build_chat_registry(),
        tool_limits=config.tool_limits,
        enabled_tool_names={"run_git_status", "search_jira"},
        workspace_enabled=False,
    )
    assert "run_git_status" in prompt
    assert "search_jira" in prompt
    assert "No workspace root is configured" in prompt


def test_local_tools_require_explicit_workspace() -> None:
    with pytest.raises(ValueError, match="No workspace configured"):
        get_workspace_root(ToolContext(invocation_id="test"))


def test_streamlit_chat_default_enabled_tools_depend_on_root(tmp_path: Path) -> None:
    config = _STREAMLIT_MODULES.package.TextualChatConfig()
    assert resolve_enabled_tool_names(config, root_path=None) == set()
    with_root = resolve_enabled_tool_names(config, root_path=tmp_path)
    assert "read_file" in with_root
    assert "search_text" in with_root
    assert "run_git_status" not in with_root
    assert "search_jira" not in with_root
    assert "write_file" not in with_root


def test_streamlit_chat_initial_render_uses_new_title_and_no_root_defaults(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))

    _STREAMLIT_MODULES.app.run_streamlit_chat_app(
        root_path=None,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )

    assert fake_st.page_config_kwargs[0]["page_title"] == "llm-tools chat"
    assert any("llm-tools chat" in text for text in fake_st.markdown_messages)
    assert "Settings" in fake_st.button_labels
    assert any(
        "<div class='llm-tools-settings-title'>Settings</div>" in text
        for text in fake_st.markdown_messages
    )
    assert any(
        "enable external tools from the same sidebar" in text
        for text in fake_st.markdown_messages
    )
    assert not any("settings rail" in text for text in fake_st.markdown_messages)
    app_state = fake_st.session_state[_STREAMLIT_MODULES.app._APP_STATE_SLOT]
    active = app_state.sessions[app_state.active_session_id]
    assert active.runtime.root_path is None
    assert active.runtime.enabled_tools == []
    assert (
        active.runtime.provider_mode_strategy
        is _STREAMLIT_MODULES.app.ProviderModeStrategy.MD_JSON
    )


def test_streamlit_chat_visible_transcript_hides_hidden_notices_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    app_state = _make_app_state(root_path=str(tmp_path))
    record = app_state.sessions[app_state.active_session_id]
    record.transcript.extend(
        [
            _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(role="user", text="Hi"),
            _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
                role="system",
                text="Workspace root updated",
                show_in_transcript=False,
            ),
            _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
                role="error", text="Problem"
            ),
            _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
                role="assistant", text="Hello"
            ),
        ]
    )
    fake_st.session_state[_STREAMLIT_MODULES.app._APP_STATE_SLOT] = app_state
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))

    _STREAMLIT_MODULES.app.run_streamlit_chat_app(
        root_path=tmp_path,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )

    assert fake_st.chat_roles.count("user") == 1
    assert fake_st.chat_roles.count("assistant") == 2
    assert "System" not in fake_st.caption_messages
    assert "Error" in fake_st.caption_messages


def test_streamlit_chat_settings_render_inside_sidebar_expander(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    app_state = _make_app_state(root_path=None)
    app_state.preferences.settings_panel_open = False
    fake_st.session_state[_STREAMLIT_MODULES.app._APP_STATE_SLOT] = app_state
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))

    _STREAMLIT_MODULES.app.run_streamlit_chat_app(
        root_path=None,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )

    assert "Settings" in fake_st.button_labels
    assert not any(
        "llm-tools-settings-rail-shell" in text for text in fake_st.markdown_messages
    )
    assert app_state.preferences.settings_panel_open is False


def test_streamlit_chat_theme_state_applies_before_sidebar_and_persists_panel(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    fake_st.session_state[_STREAMLIT_MODULES.app._THEME_TOGGLE_KEY] = False
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))

    _STREAMLIT_MODULES.app.run_streamlit_chat_app(
        root_path=None,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )

    app_state = fake_st.session_state[_STREAMLIT_MODULES.app._APP_STATE_SLOT]
    assert app_state.preferences.theme_mode == "light"
    assert app_state.preferences.settings_panel_open is True
    assert any("#edf3fb" in text for text in fake_st.markdown_messages)

    reloaded = _STREAMLIT_MODULES.app._load_workspace_state(
        root_path=None,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )
    assert reloaded.preferences.theme_mode == "light"
    assert reloaded.preferences.settings_panel_open is True


def test_streamlit_chat_persists_multiple_sessions_and_deletes_them(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))

    state = _STREAMLIT_MODULES.app._load_workspace_state(
        root_path=tmp_path,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )
    original_id = state.active_session_id
    _STREAMLIT_MODULES.app._create_session(
        state,
        template_runtime=state.sessions[original_id].runtime,
    )
    reloaded = _STREAMLIT_MODULES.app._load_workspace_state(
        root_path=tmp_path,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )
    assert len(reloaded.session_order) == 2
    session_to_delete = reloaded.session_order[0]
    _STREAMLIT_MODULES.app._delete_session(
        reloaded,
        session_id=session_to_delete,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        root_path=tmp_path,
    )
    assert session_to_delete not in reloaded.session_order
    assert not (_STREAMLIT_MODULES.app._session_path(session_to_delete)).exists()


def test_streamlit_chat_skips_corrupt_persisted_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    storage_root = tmp_path / "state"
    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(storage_root))
    (storage_root / "sessions").mkdir(parents=True)
    (storage_root / "index.json").write_text(
        _STREAMLIT_MODULES.app.StreamlitSessionIndex(
            active_session_id="broken",
            session_order=["broken"],
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    (storage_root / "sessions" / "broken.json").write_text(
        "{not-json", encoding="utf-8"
    )
    state = _STREAMLIT_MODULES.app._load_workspace_state(
        root_path=None,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )
    assert state.session_order
    assert any(
        "Skipped unreadable chat session broken" in text
        for text in state.startup_notices
    )


def test_streamlit_chat_does_not_persist_entered_api_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(
        text_input_values={"api-key:OPENAI_API_KEY": "top-secret"},
    )
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))
    config = _STREAMLIT_MODULES.package.TextualChatConfig().model_copy(
        update={
            "llm": _STREAMLIT_MODULES.package.TextualChatConfig().llm.model_copy(
                update={
                    "provider": _STREAMLIT_MODULES.package.ProviderPreset.OPENAI,
                    "model_name": "gpt-demo",
                }
            )
        }
    )
    _STREAMLIT_MODULES.app.run_streamlit_chat_app(root_path=None, config=config)
    saved_text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in (tmp_path / "state").rglob("*.json")
    )
    assert "top-secret" not in saved_text


def test_streamlit_chat_turn_processes_completed_result(tmp_path: Path) -> None:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "app.py").write_text("needle = 1\n", encoding="utf-8")
    outcome = process_streamlit_chat_turn(
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
                    ).model_dump(mode="json")
                ),
            ]
        ),
        session_state=ChatSessionState(),
        user_message="Where is needle defined?",
        root_path=tmp_path,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )
    assert outcome.transcript_entries[-1].final_response is not None
    assert outcome.transcript_entries[-1].final_response.answer.startswith(
        "It is defined"
    )
    assert len(outcome.session_state.turns) == 1


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
        root_path=None,
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

    assert _STREAMLIT_MODULES.app._launch_streamlit_app(["--config", "chat.yaml"]) == 0
    assert captured[:3] == [
        "streamlit",
        "run",
        str(Path(_STREAMLIT_MODULES.app.__file__).resolve()),
    ]
    assert captured[3:5] == [
        "--browser.gatherUsageStats=false",
        "--client.toolbarMode=minimal",
    ]
    assert captured[5] == "--"

    config_path = tmp_path / "chat.yaml"
    config_path.write_text("llm:\n  provider: ollama\n", encoding="utf-8")
    called: list[tuple[Path | None, object]] = []
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "run_streamlit_chat_app",
        lambda *, root_path, config: called.append((root_path, config)),
    )
    _STREAMLIT_MODULES.app._run_streamlit_script(["--config", str(config_path)])
    assert called[0][0] is None


def test_streamlit_chat_theme_css_covers_widget_surfaces() -> None:
    css = _STREAMLIT_MODULES.app._streamlit_theme_css("light")

    assert 'div[data-baseweb="select"] > div' in css
    assert '[data-testid="stExpander"]' in css
    assert '[data-baseweb="tab"]' in css
    assert '[data-testid="stDownloadButton"] > button' in css
    assert '[data-testid="stFormSubmitButton"] > button' in css
    assert '[data-testid="stCode"]' in css
    assert '[data-testid="stCodeBlock"]' in css
    assert '[data-testid="stHeader"]' in css
    assert ".stAppToolbar" in css
    assert ".stMainBlockContainer" in css
    assert ".stMain .block-container" in css
    assert '[data-testid="InputInstructions"]' in css
    assert "llm-tools-status-line" in css


def test_streamlit_chat_fatal_error_renderer_and_script_wrapper(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)

    boom = RuntimeError("boom")
    _STREAMLIT_MODULES.app._render_fatal_error(boom)
    assert any("unexpected error" in text for text in fake_st.markdown_messages)
    assert "Traceback" in fake_st.button_labels
    assert "fatal-error-traceback" in fake_st.text_area_values
    assert "RuntimeError: boom" in fake_st.error_messages[-1]

    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "run_streamlit_chat_app",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("wrapper boom")),
    )
    called: list[str] = []
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_render_fatal_error",
        lambda exc: called.append(str(exc)),
    )
    _STREAMLIT_MODULES.app._run_streamlit_script([])
    assert called == ["wrapper boom"]


def test_streamlit_chat_helper_defaults_and_model_validators(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    config = _STREAMLIT_MODULES.package.TextualChatConfig()
    assert _STREAMLIT_MODULES.app._dedupe_preserve([" a ", "", "a", "b "]) == ["a", "b"]
    assert (
        _STREAMLIT_MODULES.app._default_model_for_provider(
            config, _STREAMLIT_MODULES.package.ProviderPreset.OPENAI
        )
        == "gpt-4.1-mini"
    )
    assert (
        _STREAMLIT_MODULES.app._default_model_for_provider(
            config, _STREAMLIT_MODULES.package.ProviderPreset.OLLAMA
        )
        == config.llm.model_name
    )
    assert (
        _STREAMLIT_MODULES.app._default_base_url_for_provider(
            config, _STREAMLIT_MODULES.package.ProviderPreset.OPENAI
        )
        is None
    )
    assert (
        _STREAMLIT_MODULES.app._default_base_url_for_provider(
            config, _STREAMLIT_MODULES.package.ProviderPreset.OLLAMA
        )
        == config.llm.api_base_url
    )
    assert (
        _STREAMLIT_MODULES.app._filter_enabled_tools_for_root(
            {"read_file", "run_git_status"}, root_path=None
        )
        == set()
    )

    configured = config.model_copy(
        update={
            "policy": config.policy.model_copy(
                update={"enabled_tools": ["read_file", "run_git_status", "missing"]}
            )
        }
    )
    assert (
        _STREAMLIT_MODULES.app._default_enabled_tool_names(configured, root_path=None)
        == set()
    )

    runtime = _STREAMLIT_MODULES.app._default_runtime_config(config, root_path=tmp_path)
    assert runtime.root_path == str(tmp_path)
    assert "read_file" in runtime.enabled_tools
    assert (
        runtime.provider_mode_strategy
        is _STREAMLIT_MODULES.app.ProviderModeStrategy.MD_JSON
    )
    llm_config = _STREAMLIT_MODULES.app._llm_config_for_runtime(config, runtime)
    assert llm_config.model_name == runtime.model_name

    parser_args = build_parser().parse_args(
        [str(tmp_path), "--directory", str(tmp_path / "override")]
    )
    assert (
        _STREAMLIT_MODULES.app._resolve_root_argument(parser_args)
        == (tmp_path / "override").resolve()
    )

    existing_root, root_error = _STREAMLIT_MODULES.app._resolve_root_text(str(tmp_path))
    missing_root, missing_error = _STREAMLIT_MODULES.app._resolve_root_text(
        str(tmp_path / "missing")
    )
    file_path = tmp_path / "file.txt"
    file_path.write_text("x", encoding="utf-8")
    _, file_error = _STREAMLIT_MODULES.app._resolve_root_text(str(file_path))
    assert existing_root == str(tmp_path.resolve())
    assert root_error is None
    assert missing_root is None and "does not exist" in str(missing_error)
    assert "not a directory" in str(file_error)

    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))
    assert _STREAMLIT_MODULES.app._storage_root() == (tmp_path / "state").resolve()
    assert (
        _STREAMLIT_MODULES.app._read_model_file(
            tmp_path / "absent.json", _STREAMLIT_MODULES.app.StreamlitPreferences
        )
        is None
    )

    validated_runtime = _STREAMLIT_MODULES.app.StreamlitRuntimeConfig(
        model_name="  demo  ",
        api_base_url="  ",
        root_path="  ",
        enabled_tools=[" read_file ", "search_text"],
        provider_mode_strategy="json",
    )
    assert validated_runtime.model_name == "demo"
    assert validated_runtime.api_base_url is None
    assert validated_runtime.root_path is None
    assert validated_runtime.enabled_tools == ["read_file", "search_text"]
    assert (
        validated_runtime.provider_mode_strategy
        is _STREAMLIT_MODULES.app.ProviderModeStrategy.JSON
    )
    with pytest.raises(ValueError):
        _STREAMLIT_MODULES.app.StreamlitRuntimeConfig(model_name="   ")
    with pytest.raises(ValueError):
        _STREAMLIT_MODULES.app.StreamlitRuntimeConfig(enabled_tools=["read_file", " "])

    prefs = _STREAMLIT_MODULES.app.StreamlitPreferences(
        recent_roots=["  /a  ", ""],
        recent_models={" ollama ": [" gemma ", " "]},
        recent_base_urls={" ": ["skip"], "openai": [" https://api.example.com ", " "]},
    )
    assert prefs.recent_roots == ["/a"]
    assert prefs.recent_models == {"ollama": ["gemma"]}
    assert prefs.recent_base_urls == {"openai": ["https://api.example.com"]}


def test_streamlit_chat_build_runner_and_render_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)

    captured: dict[str, object] = {}

    def _fake_build_chat_executor(*, policy):
        captured["policy"] = policy
        return SimpleNamespace(name="registry"), SimpleNamespace(name="executor")

    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "build_chat_executor", _fake_build_chat_executor
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "build_chat_system_prompt", lambda **kwargs: "prompt"
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "build_chat_context",
        lambda **kwargs: {"workspace": kwargs["root_path"]},
    )

    def _fake_run_interactive_chat_session_turn(**kwargs):
        captured["runner_kwargs"] = kwargs
        return SimpleNamespace(name="runner")

    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "run_interactive_chat_session_turn",
        _fake_run_interactive_chat_session_turn,
    )

    runtime = _make_runtime(root_path=str(tmp_path))
    runtime.enabled_tools = ["read_file", "run_git_status", "search_jira"]
    runtime.allow_network = True
    runtime.allow_subprocess = True
    runtime.require_approval_for = {SideEffectClass.LOCAL_WRITE}
    result = _STREAMLIT_MODULES.app._build_chat_runner(
        session_id="session-1",
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        runtime=runtime,
        provider=SimpleNamespace(),
        session_state=ChatSessionState(),
        user_message="hello",
    )
    assert result.name == "runner"
    policy = captured["policy"]
    assert policy.allow_network is True
    assert policy.allow_filesystem is True
    assert policy.allow_subprocess is True
    assert SideEffectClass.LOCAL_READ in policy.allowed_side_effects
    assert SideEffectClass.EXTERNAL_READ in policy.allowed_side_effects

    response = ChatFinalResponse(
        answer="Answer",
        citations=[
            {"source_path": "src/app.py", "line_start": 1, "excerpt": "needle = 1"}
        ],
        confidence=0.9,
        uncertainty=["maybe"],
        missing_information=["more"],
        follow_up_suggestions=["next"],
    )
    export_text = _STREAMLIT_MODULES.app._transcript_export_text(
        [
            _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(role="user", text="Hi"),
            _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
                role="system",
                text="Root changed",
                show_in_transcript=False,
            ),
            _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
                role="error", text="Problem"
            ),
            _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
                role="assistant", text="Answer", final_response=response
            ),
        ]
    )
    assert "Hi" in export_text and "Answer" in export_text
    assert "Root changed" not in export_text
    assert "Problem" in export_text

    _STREAMLIT_MODULES.app._render_theme(
        _STREAMLIT_MODULES.app.StreamlitPreferences(theme_mode="light")
    )
    _STREAMLIT_MODULES.app._render_brand_header()
    _STREAMLIT_MODULES.app._render_final_response(response)
    _STREAMLIT_MODULES.app._render_transcript_entry(
        _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(role="user", text="Question")
    )
    _STREAMLIT_MODULES.app._render_transcript_entry(
        _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
            role="system", text="System note"
        )
    )
    _STREAMLIT_MODULES.app._render_transcript_entry(
        _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(role="error", text="Problem")
    )
    _STREAMLIT_MODULES.app._render_transcript_entry(
        _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
            role="assistant",
            text="Interrupted output",
            assistant_completion_state="interrupted",
        )
    )
    _STREAMLIT_MODULES.app._render_transcript_entry(
        _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
            role="assistant", text="Plain answer"
        )
    )
    record = _make_record(root_path=str(tmp_path))
    record.runtime.provider_mode_strategy = (
        _STREAMLIT_MODULES.app.ProviderModeStrategy.JSON
    )
    record.token_usage = ChatTokenUsage(session_tokens=123, active_context_tokens=45)
    record.confidence = 0.75
    _STREAMLIT_MODULES.app._render_summary_chips(record)
    _STREAMLIT_MODULES.app._render_empty_state(record)
    record.inspector_state.provider_messages.append(
        _STREAMLIT_MODULES.app.StreamlitInspectorEntry(
            label="provider", payload={"x": 1}
        )
    )
    record.runtime.inspector_open = True
    app_state = _make_app_state(root_path=str(tmp_path))
    app_state.sessions["session-1"] = record
    app_state.show_export_for.add("session-1")
    _STREAMLIT_MODULES.app._render_session_details(app_state, session_id="session-1")

    assert any("llm-tools chat" in text for text in fake_st.markdown_messages)
    assert any(
        "instructor: Structured response" in text for text in fake_st.markdown_messages
    )
    assert "Confidence: 0.90" in fake_st.caption_messages
    assert "System" in fake_st.caption_messages
    assert "Error" in fake_st.caption_messages
    assert any(call[0] == "Download transcript" for call in fake_st.download_calls)
    assert fake_st.chat_roles.count("assistant") >= 4


def test_streamlit_chat_workspace_state_persistence_and_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))
    app_state = _make_app_state(root_path=str(tmp_path))
    record = app_state.sessions["session-1"]
    record.transcript.extend(
        [
            _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(role="user", text="Hi"),
            _STREAMLIT_MODULES.app.StreamlitTranscriptEntry(
                role="assistant", text="Hello"
            ),
        ]
    )
    _STREAMLIT_MODULES.app._remember_runtime_preferences(
        app_state.preferences, record.runtime
    )
    _STREAMLIT_MODULES.app._touch_record(record)
    orphan = _STREAMLIT_MODULES.app._sessions_dir() / "orphan.json"
    orphan.parent.mkdir(parents=True, exist_ok=True)
    orphan.write_text("{}", encoding="utf-8")
    _STREAMLIT_MODULES.app._save_workspace_state(app_state)
    assert not orphan.exists()
    assert _STREAMLIT_MODULES.app._active_session(app_state) is record
    assert _STREAMLIT_MODULES.app._turn_state_for(app_state, "missing").busy is False

    session_id = _STREAMLIT_MODULES.app._create_session(
        app_state, template_runtime=record.runtime
    )
    assert session_id == app_state.active_session_id
    _STREAMLIT_MODULES.app._delete_session(
        app_state,
        session_id=session_id,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        root_path=tmp_path,
    )
    assert app_state.session_order

    storage_root = tmp_path / "broken-state"
    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(storage_root))
    storage_root.mkdir(parents=True, exist_ok=True)
    _STREAMLIT_MODULES.app._preferences_path().write_text("{bad", encoding="utf-8")
    _STREAMLIT_MODULES.app._index_path().write_text(
        _STREAMLIT_MODULES.app.StreamlitSessionIndex(
            active_session_id="missing", session_order=["missing"]
        ).model_dump_json(indent=2),
        encoding="utf-8",
    )
    loaded = _STREAMLIT_MODULES.app._load_workspace_state(
        root_path=None,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )
    assert loaded.active_session_id in loaded.sessions
    assert any(
        "Unable to load preferences" in notice for notice in loaded.startup_notices
    )
    assert any(
        "Skipped missing chat session missing." in notice
        for notice in loaded.startup_notices
    )


def test_streamlit_chat_event_reducers_and_drain(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))
    app_state = _make_app_state(root_path=str(tmp_path))
    approval = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-1",
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {"path": "a.txt"}},
            tool_name="read_file",
            tool_version="0.1.0",
            policy_reason="approval required",
            policy_metadata={},
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:05:00Z",
        ),
        tool_name="read_file",
        redacted_arguments={"path": "a.txt"},
        policy_reason="approval required",
        policy_metadata={},
    )
    queued_events = [
        _STREAMLIT_MODULES.app._serialize_workflow_event(
            ChatWorkflowStatusEvent(status="thinking"),
            turn_number=1,
            session_id="session-1",
        ),
        _STREAMLIT_MODULES.app._serialize_workflow_event(
            ChatWorkflowApprovalEvent(approval=approval),
            turn_number=1,
            session_id="session-1",
        ),
        _STREAMLIT_MODULES.app._serialize_workflow_event(
            ChatWorkflowApprovalResolvedEvent(approval=approval, resolution="approved"),
            turn_number=1,
            session_id="session-1",
        ),
        _STREAMLIT_MODULES.app._serialize_workflow_event(
            ChatWorkflowInspectorEvent(
                round_index=1, kind="provider_messages", payload=[{"role": "user"}]
            ),
            turn_number=1,
            session_id="session-1",
        ),
        _STREAMLIT_MODULES.app._serialize_workflow_event(
            ChatWorkflowResultEvent(
                result=ChatWorkflowTurnResult(
                    status="completed",
                    new_messages=[],
                    context_warning="warn",
                    continuation_reason=None,
                    final_response=ChatFinalResponse(answer="done", confidence=0.5),
                    token_usage=ChatTokenUsage(
                        session_tokens=22, active_context_tokens=10
                    ),
                    session_state=ChatSessionState(),
                )
            ),
            turn_number=1,
            session_id="session-1",
        ),
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="error", payload="boom", turn_number=1, session_id="session-1"
        ),
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="complete", payload=None, turn_number=1, session_id="session-1"
        ),
    ]
    queue_obj = _STREAMLIT_MODULES.app.queue.Queue()
    for event in queued_events:
        queue_obj.put(event)
    turn_state = _STREAMLIT_MODULES.app._turn_state_for(app_state, "session-1")
    turn_state.busy = False
    fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
            session_id="session-1",
            runner=_FakeRunnerHandle(),
            event_queue=queue_obj,
            thread=_DeadThread(),
            turn_number=1,
        )
    )
    pending_prompt = _STREAMLIT_MODULES.app._drain_active_turn_events(app_state)
    record = app_state.sessions["session-1"]
    assert pending_prompt is None
    assert any(entry.text == "warn" for entry in record.transcript)
    assert any(
        entry.text == "Approved pending approval request."
        for entry in record.transcript
    )
    assert any(entry.text == "boom" for entry in record.transcript)
    assert record.inspector_state.provider_messages
    assert record.confidence == 0.5
    assert fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] is None

    interrupted_event = ChatWorkflowResultEvent(
        result=ChatWorkflowTurnResult(
            status="interrupted",
            new_messages=[
                ChatMessage(
                    role="assistant",
                    content="partial answer",
                    completion_state="interrupted",
                )
            ],
            interruption_reason="stopped",
        )
    )
    _STREAMLIT_MODULES.app._apply_turn_result(
        app_state, session_id="session-1", event=interrupted_event
    )
    assert record.transcript[-1].assistant_completion_state == "interrupted"

    interrupted_no_message = ChatWorkflowResultEvent(
        result=ChatWorkflowTurnResult(
            status="interrupted",
            new_messages=[],
            interruption_reason="stopped hard",
        )
    )
    _STREAMLIT_MODULES.app._apply_turn_result(
        app_state, session_id="session-1", event=interrupted_no_message
    )
    assert record.transcript[-1].text == "stopped hard"


def test_streamlit_chat_dead_worker_surfaces_error_and_clears_busy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))

    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    turn_state = _STREAMLIT_MODULES.app._turn_state_for(app_state, session_id)
    turn_state.busy = True
    turn_state.status_text = "thinking"
    turn_state.started_at_monotonic = 100.0
    turn_state.last_event_at_monotonic = 100.0

    fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
            session_id=session_id,
            runner=_FakeRunnerHandle(),
            event_queue=_STREAMLIT_MODULES.app.queue.Queue(),
            thread=_DeadThread(),
            turn_number=1,
        )
    )

    pending_prompt = _STREAMLIT_MODULES.app._drain_active_turn_events(app_state)

    assert pending_prompt is None
    assert turn_state.busy is False
    assert turn_state.status_text == ""
    assert turn_state.started_at_monotonic is None
    assert turn_state.last_event_at_monotonic is None
    assert fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] is None
    assert "Prompt text" in app_state.sessions[session_id].transcript[-1].text


def test_streamlit_chat_composer_clears_on_next_render_without_mutating_widget_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit(
        button_values={"send:session-1": True},
        text_input_values={"composer:session-1": "hello there"},
    )
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_submit_streamlit_prompt", lambda **kwargs: None
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_save_workspace_state", lambda app_state: None
    )

    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))
    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app._render_status_and_composer(
            app_state,
            session_id=session_id,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )

    assert session_id in app_state.clear_draft_for
    assert app_state.drafts[session_id] == ""

    fake_st.button_values.clear()
    fake_st.text_input_values.pop(f"composer:{session_id}", None)
    _STREAMLIT_MODULES.app._render_status_and_composer(
        app_state,
        session_id=session_id,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )

    assert session_id not in app_state.clear_draft_for
    assert fake_st.session_state[f"composer:{session_id}"] == ""
    assert app_state.drafts[session_id] == ""


def test_streamlit_chat_complete_event_finishes_cancelled_turn(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)

    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    turn_state = _STREAMLIT_MODULES.app._turn_state_for(app_state, session_id)
    turn_state.busy = True
    turn_state.cancelling = True
    turn_state.status_text = "stopping"
    turn_state.pending_interrupt_draft = "next prompt"

    ignored = _STREAMLIT_MODULES.app._apply_queued_event(
        app_state,
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="status",
            payload=ChatWorkflowStatusEvent(status="thinking").model_dump(mode="json"),
            turn_number=1,
            session_id=session_id,
        ),
    )
    assert ignored is None
    assert turn_state.status_text == "stopping"

    pending_prompt = _STREAMLIT_MODULES.app._apply_queued_event(
        app_state,
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="complete",
            payload=None,
            turn_number=1,
            session_id=session_id,
        ),
    )
    assert pending_prompt == "next prompt"
    assert turn_state.busy is False
    assert turn_state.cancelling is False
    assert turn_state.status_text == ""
    assert app_state.sessions[session_id].transcript[-1].text == "Stopped active turn."


def test_streamlit_chat_cancel_active_turn_clears_stale_busy_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)

    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    turn_state = _STREAMLIT_MODULES.app._turn_state_for(app_state, session_id)
    turn_state.busy = True
    turn_state.status_text = "thinking"

    _STREAMLIT_MODULES.app._cancel_active_turn(app_state, session_id=session_id)

    assert turn_state.busy is False
    assert turn_state.status_text == ""
    assert app_state.sessions[session_id].transcript[-1].text == "Stopped active turn."


def test_streamlit_chat_root_warning_only_shows_when_workspace_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_save_workspace_state", lambda app_state: None
    )

    fake_busy = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_busy)
    busy_state = _make_app_state(root_path=str(tmp_path))
    busy_session = busy_state.active_session_id
    _STREAMLIT_MODULES.app._turn_state_for(busy_state, busy_session).busy = True
    _STREAMLIT_MODULES.app._render_tools_popover(
        busy_state,
        session_id=busy_session,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )
    assert (
        "Select a root directory to enable this tool." not in fake_busy.caption_messages
    )

    fake_missing = _FakeStreamlit()
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_missing
    )
    missing_state = _make_app_state(root_path=None)
    missing_session = missing_state.active_session_id
    _STREAMLIT_MODULES.app._render_tools_popover(
        missing_state,
        session_id=missing_session,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )
    assert (
        "Select a root directory to enable this tool." in fake_missing.caption_messages
    )


def test_streamlit_chat_submit_and_command_helpers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_save_workspace_state", lambda app_state: None
    )
    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    app_state.drafts[session_id] = "/help"

    started: list[str] = []
    create_provider_calls: list[dict[str, object]] = []

    def _fake_create_provider(*args: object, **kwargs: object) -> str:
        del args
        create_provider_calls.append(dict(kwargs))
        return "provider"

    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "create_provider", _fake_create_provider
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_start_streamlit_turn",
        lambda **kwargs: started.append(kwargs["user_message"]),
    )
    cancelled: list[bool] = []
    original_cancel_active_turn = _STREAMLIT_MODULES.app._cancel_active_turn
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_cancel_active_turn",
        lambda app_state, session_id, preserve_pending_prompt=False: cancelled.append(
            preserve_pending_prompt
        ),
    )

    turn_state = _STREAMLIT_MODULES.app._turn_state_for(app_state, session_id)
    _STREAMLIT_MODULES.app._submit_streamlit_prompt(
        app_state=app_state,
        session_id=session_id,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        prompt="   ",
    )
    turn_state.busy = True
    _STREAMLIT_MODULES.app._submit_streamlit_prompt(
        app_state=app_state,
        session_id=session_id,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        prompt="pending",
    )
    turn_state.busy = False
    app_state.sessions[
        session_id
    ].runtime.provider_mode_strategy = (
        _STREAMLIT_MODULES.app.ProviderModeStrategy.MD_JSON
    )
    _STREAMLIT_MODULES.app._submit_streamlit_prompt(
        app_state=app_state,
        session_id=session_id,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        prompt="hello",
    )
    assert cancelled == [True]
    assert started == ["hello"]
    assert create_provider_calls[-1]["mode_strategy"] == "md_json"

    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_cancel_active_turn", original_cancel_active_turn
    )

    handle_runner = _FakeRunnerHandle()
    fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
            session_id=session_id,
            runner=handle_runner,
            event_queue=_STREAMLIT_MODULES.app.queue.Queue(),
            thread=_DeadThread(),
            turn_number=1,
        )
    )
    approval = ChatWorkflowApprovalState(
        approval_request=ApprovalRequest(
            approval_id="approval-1",
            invocation_index=1,
            request={"tool_name": "read_file", "arguments": {"path": "a.txt"}},
            tool_name="read_file",
            tool_version="0.1.0",
            policy_reason="approval required",
            policy_metadata={},
            requested_at="2026-01-01T00:00:00Z",
            expires_at="2026-01-01T00:05:00Z",
        ),
        tool_name="read_file",
        redacted_arguments={"path": "a.txt"},
        policy_reason="approval required",
        policy_metadata={},
    )
    turn_state.pending_approval = approval
    _STREAMLIT_MODULES.app._resolve_active_approval(
        app_state, session_id=session_id, approved=True
    )
    assert handle_runner.approvals == [True]
    _STREAMLIT_MODULES.app._cancel_active_turn(app_state, session_id=session_id)
    assert handle_runner.cancelled is True

    control_state = _STREAMLIT_MODULES.app._runtime_to_control_state(
        _STREAMLIT_MODULES.package.TextualChatConfig(),
        app_state.sessions[session_id].runtime,
    )
    control_state.active_model_name = "replacement"
    control_state.enabled_tools = {"read_file"}
    control_state.require_approval_for = {SideEffectClass.LOCAL_READ}
    control_state.inspector_open = True
    _STREAMLIT_MODULES.app._apply_control_state(
        app_state.sessions[session_id].runtime, control_state
    )
    assert app_state.sessions[session_id].runtime.model_name == "replacement"
    assert app_state.sessions[session_id].runtime.inspector_open is True
    assert _STREAMLIT_MODULES.app._is_command_prompt("/help") is True
    assert _STREAMLIT_MODULES.app._is_command_prompt("exit") is True
    assert _STREAMLIT_MODULES.app._is_command_prompt("hello") is False

    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "handle_chat_command",
        lambda *args, **kwargs: ChatCommandOutcome(
            handled=True,
            notices=[ChatControlNotice(role="system", text="notice")],
            request_copy=True,
        ),
    )
    outcome = _STREAMLIT_MODULES.app._run_streamlit_command(
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        app_state=app_state,
        session_id=session_id,
    )
    _STREAMLIT_MODULES.app._apply_streamlit_command(
        app_state, session_id=session_id, outcome=outcome
    )
    assert app_state.show_export_for == {session_id}
    assert app_state.sessions[session_id].transcript[-1].text == "notice"


def test_streamlit_chat_provider_controls_tools_popover_and_composer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_save_workspace_state", lambda app_state: None
    )

    app_state = _make_app_state(root_path=None)
    session_id = app_state.active_session_id
    record = app_state.sessions[session_id]
    fake_provider_change = _FakeStreamlit(
        selectbox_values={
            f"provider:{session_id}": _STREAMLIT_MODULES.package.ProviderPreset.OPENAI.value
        }
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_provider_change
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_available_model_options",
        lambda *args: (["gpt-4.1-mini"], None),
    )
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app._provider_control_strip(
            app_state,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
            session_id=session_id,
        )
    assert record.runtime.provider is _STREAMLIT_MODULES.package.ProviderPreset.OPENAI
    assert (
        fake_provider_change.session_state[
            _STREAMLIT_MODULES.app._WIDGET_OVERRIDES_STATE_SLOT
        ][f"model:{session_id}"]
        == "gpt-4.1-mini"
    )
    assert (
        fake_provider_change.session_state[
            _STREAMLIT_MODULES.app._WIDGET_OVERRIDES_STATE_SLOT
        ][f"base-url:{session_id}"]
        == ""
    )

    app_state = _make_app_state(root_path=None)
    session_id = app_state.active_session_id
    record = app_state.sessions[session_id]
    fake_mode_change = _FakeStreamlit(
        selectbox_values={
            f"provider-mode:{session_id}": _STREAMLIT_MODULES.app.ProviderModeStrategy.JSON.value
        }
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_mode_change
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_available_model_options",
        lambda *args: (["gemma4:26b"], None),
    )
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app._provider_control_strip(
            app_state,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
            session_id=session_id,
        )
    assert (
        record.runtime.provider_mode_strategy
        is _STREAMLIT_MODULES.app.ProviderModeStrategy.JSON
    )
    assert (
        fake_mode_change.session_state[
            _STREAMLIT_MODULES.app._WIDGET_OVERRIDES_STATE_SLOT
        ][f"provider-mode:{session_id}"]
        == "json"
    )

    app_state = _make_app_state(root_path=None)
    session_id = app_state.active_session_id
    record = app_state.sessions[session_id]
    fake_root_apply = _FakeStreamlit(
        button_values={f"root-apply:{session_id}": True},
        text_input_values={f"root-input:{session_id}": str(tmp_path)},
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_root_apply
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_available_model_options",
        lambda *args: (["gemma4:26b"], None),
    )
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app._provider_control_strip(
            app_state,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
            session_id=session_id,
        )
    assert record.runtime.root_path == str(tmp_path.resolve())
    assert record.transcript[-1].text.startswith("Workspace root updated")

    browse_state = _make_app_state(root_path=None)
    browse_session = browse_state.active_session_id
    browse_record = browse_state.sessions[browse_session]
    fake_browse_root = _FakeStreamlit(
        button_values={f"root-browse:{browse_session}": True},
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_browse_root
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_pick_root_directory",
        lambda: (str(tmp_path.resolve()), None),
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_available_model_options",
        lambda *args: (["gemma4:26b"], None),
    )
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app._provider_control_strip(
            browse_state,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
            session_id=browse_session,
        )
    assert browse_record.runtime.root_path == str(tmp_path.resolve())
    assert browse_record.transcript[-1].text.startswith("Workspace root updated")
    assert fake_browse_root.session_state[
        _STREAMLIT_MODULES.app._WIDGET_OVERRIDES_STATE_SLOT
    ][f"root-input:{browse_session}"] == str(tmp_path.resolve())

    recent_state = _make_app_state(root_path=None)
    recent_session = recent_state.active_session_id
    recent_record = recent_state.sessions[recent_session]
    recent_state.preferences.recent_roots = [str(tmp_path.resolve())]
    fake_recent_root = _FakeStreamlit(
        selectbox_values={f"recent-root:{recent_session}": str(tmp_path.resolve())},
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_recent_root
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_available_model_options",
        lambda *args: (["gemma4:26b"], None),
    )
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app._provider_control_strip(
            recent_state,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
            session_id=recent_session,
        )
    assert recent_record.runtime.root_path == str(tmp_path.resolve())
    assert recent_record.transcript[-1].text.startswith("Workspace root updated")
    assert fake_recent_root.session_state[
        _STREAMLIT_MODULES.app._WIDGET_OVERRIDES_STATE_SLOT
    ][f"recent-root:{recent_session}"] == (_STREAMLIT_MODULES.app._ROOT_SENTINEL)

    stale_state = _make_app_state(root_path=None)
    stale_session = stale_state.active_session_id
    stale_record = stale_state.sessions[stale_session]
    missing_root = str((tmp_path / "missing-root").resolve())
    stale_state.preferences.recent_roots = [missing_root]
    fake_stale_root = _FakeStreamlit(
        selectbox_values={f"recent-root:{stale_session}": missing_root},
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_stale_root
    )
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app._provider_control_strip(
            stale_state,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
            session_id=stale_session,
        )
    assert stale_record.runtime.root_path is None
    assert stale_record.transcript[-1].role == "error"
    assert "does not exist" in stale_record.transcript[-1].text
    assert fake_stale_root.session_state[
        _STREAMLIT_MODULES.app._WIDGET_OVERRIDES_STATE_SLOT
    ][f"recent-root:{stale_session}"] == (_STREAMLIT_MODULES.app._ROOT_SENTINEL)

    fake_tools = _FakeStreamlit(
        button_values={f"apply-tool-preset:{session_id}": True},
        selectbox_values={f"tool-preset:{session_id}": "all_builtins"},
    )
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_tools)
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app._render_tools_popover(
            app_state,
            session_id=session_id,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )
    assert "search_jira" in record.runtime.enabled_tools
    assert record.runtime.allow_network is True
    assert record.runtime.allow_filesystem is True
    assert record.runtime.allow_subprocess is True
    assert "Apply preset" in fake_tools.button_labels
    overrides = fake_tools.session_state[
        _STREAMLIT_MODULES.app._WIDGET_OVERRIDES_STATE_SLOT
    ]
    assert overrides[f"allow-network:{session_id}"] is True
    assert overrides[f"tool:{session_id}:search_jira"] is True

    record.runtime.require_approval_for = {
        _STREAMLIT_MODULES.app.SideEffectClass.LOCAL_READ,
        _STREAMLIT_MODULES.app.SideEffectClass.LOCAL_WRITE,
    }
    fake_default_tools = _FakeStreamlit(
        button_values={f"apply-tool-preset:{session_id}": True},
        selectbox_values={f"tool-preset:{session_id}": "default"},
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_default_tools
    )
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app._render_tools_popover(
            app_state,
            session_id=session_id,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )
    assert record.runtime.allow_network is False
    assert record.runtime.allow_filesystem is True
    assert record.runtime.allow_subprocess is False
    assert record.runtime.require_approval_for == set()

    fake_tools_toggle = _FakeStreamlit(
        checkbox_values={
            f"allow-network:{session_id}": False,
            f"allow-filesystem:{session_id}": True,
            f"allow-subprocess:{session_id}": False,
            f"approval:{session_id}:local_write": True,
            f"tool:{session_id}:search_jira": False,
        }
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_tools_toggle
    )
    _STREAMLIT_MODULES.app._render_tools_popover(
        app_state,
        session_id=session_id,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
    )
    assert SideEffectClass.LOCAL_WRITE in record.runtime.require_approval_for
    assert "search_jira" not in record.runtime.enabled_tools

    busy_state = _make_app_state(root_path=str(tmp_path))
    busy_session = busy_state.active_session_id
    busy_turn = _STREAMLIT_MODULES.app._turn_state_for(busy_state, busy_session)
    busy_turn.busy = True
    fake_stop = _FakeStreamlit(button_values={f"stop:{busy_session}": True})
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_stop)
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_cancel_active_turn", lambda *args, **kwargs: None
    )
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app._render_status_and_composer(
            busy_state,
            session_id=busy_session,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )
    assert any("llm-tools-status-line" in text for text in fake_stop.markdown_messages)
    assert "Tools" not in fake_stop.button_labels

    command_state = _make_app_state(root_path=str(tmp_path))
    command_session = command_state.active_session_id
    fake_command = _FakeStreamlit(
        button_values={f"send:{command_session}": True},
        text_input_values={f"composer:{command_session}": "/help"},
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_command
    )
    command_called: list[str] = []
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_run_streamlit_command",
        lambda **kwargs: (
            command_called.append("command") or ChatCommandOutcome(handled=True)
        ),
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_apply_streamlit_command", lambda *args, **kwargs: None
    )
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app._render_status_and_composer(
            command_state,
            session_id=command_session,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )
    assert command_called == ["command"]

    prompt_state = _make_app_state(root_path=str(tmp_path))
    prompt_session = prompt_state.active_session_id
    fake_prompt = _FakeStreamlit(
        button_values={f"send:{prompt_session}": True},
        text_input_values={f"composer:{prompt_session}": "hello there"},
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_prompt
    )
    submitted: list[str] = []
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_submit_streamlit_prompt",
        lambda **kwargs: submitted.append(kwargs["prompt"]),
    )
    with pytest.raises(_RerunRequestError):
        _STREAMLIT_MODULES.app._render_status_and_composer(
            prompt_state,
            session_id=prompt_session,
            config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        )
    assert submitted == ["hello there"]


def test_chat_runtime_provider_factory_and_executor_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _STREAMLIT_MODULES.package.TextualChatConfig().llm
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
            "provider": _STREAMLIT_MODULES.package.ProviderPreset.OPENAI,
            "api_key_env_var": "OPENAI_API_KEY",
            "api_base_url": None,
        }
    )
    assert (
        _CHAT_RUNTIME_MODULE.create_provider(
            openai_config,
            api_key=None,
            model_name="gpt-4.1-mini",
            mode_strategy=_STREAMLIT_MODULES.app.ProviderModeStrategy.JSON,
        )
        == "openai-provider"
    )
    assert openai_calls[0]["api_key"] == "env-key"
    assert (
        openai_calls[0]["mode_strategy"]
        == _STREAMLIT_MODULES.app.ProviderModeStrategy.JSON
    )

    ollama_config = config.model_copy(
        update={
            "provider": _STREAMLIT_MODULES.package.ProviderPreset.OLLAMA,
            "api_base_url": None,
        }
    )
    assert (
        _CHAT_RUNTIME_MODULE.create_provider(
            ollama_config,
            api_key=None,
            model_name="gemma4:26b",
            mode_strategy=_STREAMLIT_MODULES.app.ProviderModeStrategy.MD_JSON,
        )
        == "ollama-provider"
    )
    assert ollama_calls[0]["base_url"] == "http://127.0.0.1:11434/v1"
    assert ollama_calls[0]["api_key"] == "ollama"
    assert (
        ollama_calls[0]["mode_strategy"]
        == _STREAMLIT_MODULES.app.ProviderModeStrategy.MD_JSON
    )

    custom_config = config.model_copy(
        update={
            "provider": _STREAMLIT_MODULES.package.ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
            "api_base_url": "https://example.invalid/v1",
            "api_key_env_var": "OPENAI_API_KEY",
        }
    )
    custom_provider = _CHAT_RUNTIME_MODULE.create_provider(
        custom_config,
        api_key="provided",
        model_name="custom-model",
        mode_strategy=_STREAMLIT_MODULES.app.ProviderModeStrategy.TOOLS,
    )
    assert custom_provider.kind == "custom-provider"
    assert custom_calls[0]["api_key"] == "provided"
    assert (
        custom_calls[0]["mode_strategy"]
        == _STREAMLIT_MODULES.app.ProviderModeStrategy.TOOLS
    )
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
        _CHAT_RUNTIME_MODULE,
        "WorkflowExecutor",
        _fake_workflow_executor,
    )
    registry, executor = _CHAT_RUNTIME_MODULE.build_chat_executor()
    assert registry == "registry"
    assert executor == "executor"
    policy = captured["executor_kwargs"]["policy"]
    assert policy.allow_network is False
    assert policy.allow_filesystem is True
    assert policy.allow_subprocess is False


def test_streamlit_chat_helper_branch_coverage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            "--provider",
            _STREAMLIT_MODULES.package.ProviderPreset.OPENAI.value,
            "--model",
            "gpt-demo",
            "--temperature",
            "0.4",
            "--api-base-url",
            "https://api.example.invalid/v1",
            "--max-context-tokens",
            "2048",
            "--max-tool-round-trips",
            "2",
            "--max-tool-calls-per-round",
            "3",
            "--max-total-tool-calls-per-turn",
            "4",
            "--max-entries-per-call",
            "5",
            "--max-recursive-depth",
            "6",
            "--max-search-matches",
            "7",
            "--max-read-lines",
            "8",
            "--max-file-size-characters",
            "9",
            "--max-tool-result-chars",
            "10",
        ]
    )
    resolved = _resolve_chat_config(args)
    assert resolved.llm.provider is _STREAMLIT_MODULES.package.ProviderPreset.OPENAI
    assert resolved.llm.model_name == "gpt-demo"
    assert resolved.llm.temperature == 0.4
    assert resolved.llm.api_base_url == "https://api.example.invalid/v1"
    assert resolved.session.max_context_tokens == 2048
    assert resolved.session.max_tool_round_trips == 2
    assert resolved.session.max_tool_calls_per_round == 3
    assert resolved.session.max_total_tool_calls_per_turn == 4
    assert resolved.tool_limits.max_entries_per_call == 5
    assert resolved.tool_limits.max_recursive_depth == 6
    assert resolved.tool_limits.max_search_matches == 7
    assert resolved.tool_limits.max_read_lines == 8
    assert resolved.tool_limits.max_file_size_characters == 9
    assert resolved.tool_limits.max_tool_result_chars == 10

    assert (
        _STREAMLIT_MODULES.app._tool_group(
            _STREAMLIT_MODULES.app.ToolSpec(
                name="misc",
                description="misc",
                input_schema={},
                output_schema={},
                tags=["misc"],
            )
        )
        == "Other"
    )
    assert (
        _STREAMLIT_MODULES.app._default_model_for_provider(
            _STREAMLIT_MODULES.package.TextualChatConfig(),
            _STREAMLIT_MODULES.package.ProviderPreset.OLLAMA,
        )
        == "gemma4:26b"
    )
    assert (
        _STREAMLIT_MODULES.app._default_model_for_provider(
            _STREAMLIT_MODULES.package.TextualChatConfig(),
            _STREAMLIT_MODULES.package.ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
        )
        == "gpt-4.1-mini"
    )
    assert (
        _STREAMLIT_MODULES.app._default_base_url_for_provider(
            _STREAMLIT_MODULES.package.TextualChatConfig(),
            _STREAMLIT_MODULES.package.ProviderPreset.OLLAMA,
        )
        == "http://127.0.0.1:11434/v1"
    )

    assert _STREAMLIT_MODULES.app._title_from_prompt("short prompt") == "short prompt"
    assert _STREAMLIT_MODULES.app._title_from_prompt("word " * 20).endswith("...")
    assert _STREAMLIT_MODULES.app._resolve_root_text("   ") == (None, None)
    assert (
        _STREAMLIT_MODULES.app._session_matches(
            _make_record(root_path=str(tmp_path)),
            "gemma4",
        )
        is True
    )

    monkeypatch.delenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, raising=False)
    monkeypatch.setattr(_STREAMLIT_MODULES.app.Path, "home", lambda: tmp_path)
    assert (
        _STREAMLIT_MODULES.app._storage_root()
        == (tmp_path / ".llm-tools" / "chat" / "streamlit").resolve()
    )


def test_streamlit_chat_widget_delete_and_event_edge_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_st = _FakeStreamlit()
    fake_st.session_state[_STREAMLIT_MODULES.app._WIDGET_OVERRIDES_STATE_SLOT] = "bad"
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)

    overrides = _STREAMLIT_MODULES.app._widget_overrides_state()
    assert overrides == {}
    _STREAMLIT_MODULES.app._queue_widget_override("demo", "value")
    _STREAMLIT_MODULES.app._prime_widget_value("demo", "fallback")
    assert fake_st.session_state["demo"] == "value"
    _STREAMLIT_MODULES.app._prime_widget_value("fresh", "fallback")
    assert fake_st.session_state["fresh"] == "fallback"

    monkeypatch.setenv(_STREAMLIT_MODULES.app._STORAGE_ENV_VAR, str(tmp_path / "state"))
    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_streamlit_module",
        lambda: (_ for _ in ()).throw(ModuleNotFoundError()),
    )
    _STREAMLIT_MODULES.app._delete_session(
        app_state,
        session_id=session_id,
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        root_path=tmp_path,
    )
    assert app_state.session_order
    assert app_state.active_session_id in app_state.sessions
    assert app_state.active_session_id != session_id

    with pytest.raises(TypeError):
        _STREAMLIT_MODULES.app._serialize_workflow_event(
            object(), turn_number=1, session_id="session-1"
        )

    app_state = _make_app_state(root_path=str(tmp_path))
    session_id = app_state.active_session_id
    with pytest.raises(ValueError, match="Unsupported queued event kind"):
        _STREAMLIT_MODULES.app._apply_queued_event(
            app_state,
            _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
                kind="mystery",
                payload=object(),
                turn_number=99,
                session_id=session_id,
            ),
        )

    turn_state = _STREAMLIT_MODULES.app._turn_state_for(app_state, session_id)
    turn_state.busy = True
    turn_state.cancelling = True
    turn_state.pending_interrupt_draft = "next"
    queue_obj = _STREAMLIT_MODULES.app.queue.Queue()
    queue_obj.put(
        _STREAMLIT_MODULES.app.StreamlitQueuedEvent(
            kind="complete", payload=None, turn_number=1, session_id=session_id
        )
    )
    fake_st = _FakeStreamlit()
    fake_st.session_state[_STREAMLIT_MODULES.app._ACTIVE_TURN_STATE_SLOT] = (
        _STREAMLIT_MODULES.app.StreamlitActiveTurnHandle(
            session_id=session_id,
            runner=_FakeRunnerHandle(),
            event_queue=queue_obj,
            thread=_DeadThread(),
            turn_number=1,
        )
    )
    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_streamlit_module", lambda: fake_st)
    assert _STREAMLIT_MODULES.app._drain_active_turn_events(app_state) == "next"


def test_streamlit_chat_wsl_and_powershell_picker_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = str(tmp_path.resolve())

    monkeypatch.delenv("WSL_DISTRO_NAME", raising=False)
    monkeypatch.delenv("WSL_INTEROP", raising=False)
    assert _STREAMLIT_MODULES.app._running_in_wsl() is False
    monkeypatch.setenv("WSL_INTEROP", "1")
    assert _STREAMLIT_MODULES.app._running_in_wsl() is True
    monkeypatch.delenv("WSL_INTEROP", raising=False)

    monkeypatch.setattr(_STREAMLIT_MODULES.app.shutil, "which", lambda name: None)
    assert _STREAMLIT_MODULES.app._pick_root_directory_via_powershell() == (
        None,
        "Native directory picker is not available in this environment.",
    )

    monkeypatch.setattr(
        _STREAMLIT_MODULES.app.shutil,
        "which",
        lambda name: "powershell.exe" if name == "powershell.exe" else None,
    )

    def _raise_run(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("boom")

    monkeypatch.setattr(_STREAMLIT_MODULES.app.subprocess, "run", _raise_run)
    selected, error = _STREAMLIT_MODULES.app._pick_root_directory_via_powershell()
    assert selected is None
    assert error == "Unable to open native directory picker: boom"

    def _fake_run(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        if cmd[0] == "powershell.exe":
            return SimpleNamespace(returncode=1, stderr="bad picker", stdout="")
        raise AssertionError(cmd)

    monkeypatch.setattr(_STREAMLIT_MODULES.app.subprocess, "run", _fake_run)
    selected, error = _STREAMLIT_MODULES.app._pick_root_directory_via_powershell()
    assert selected is None
    assert error == "Unable to open native directory picker: bad picker"

    def _fake_empty(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        del cmd, kwargs
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(_STREAMLIT_MODULES.app.subprocess, "run", _fake_empty)
    assert _STREAMLIT_MODULES.app._pick_root_directory_via_powershell() == (None, None)

    def _fake_success(cmd: list[str], **kwargs: object) -> SimpleNamespace:
        del kwargs
        if cmd[0] == "powershell.exe":
            return SimpleNamespace(returncode=0, stderr="", stdout="C:\\repo")
        if cmd[0] == "wslpath":
            return SimpleNamespace(returncode=0, stderr="", stdout=existing)
        raise AssertionError(cmd)

    monkeypatch.setattr(
        _STREAMLIT_MODULES.app.shutil,
        "which",
        lambda name: "powershell.exe" if name == "powershell.exe" else "wslpath",
    )
    monkeypatch.setattr(_STREAMLIT_MODULES.app.subprocess, "run", _fake_success)
    assert _STREAMLIT_MODULES.app._pick_root_directory_via_powershell() == (
        existing,
        None,
    )


class _FakeTkRoot:
    def __init__(self, *, fail_attributes: bool = False) -> None:
        self.fail_attributes = fail_attributes
        self.destroyed = False

    def withdraw(self) -> None:
        return None

    def attributes(self, *args: object) -> None:
        del args
        if self.fail_attributes:
            raise RuntimeError("ignore me")

    def destroy(self) -> None:
        self.destroyed = True


class _FakeTkModule:
    def __init__(self, root: _FakeTkRoot | Exception) -> None:
        self._root = root
        self.Tk = self._build_root

    def _build_root(self) -> _FakeTkRoot:
        if isinstance(self._root, Exception):
            raise self._root
        return self._root


def test_streamlit_chat_tk_picker_branches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = str(tmp_path.resolve())

    import builtins

    real_import = builtins.__import__

    def _missing_tk(name: str, *args: object, **kwargs: object):
        if name == "tkinter":
            raise ImportError("no tk")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _missing_tk)
    assert _STREAMLIT_MODULES.app._pick_root_directory_via_tk() == (
        None,
        "Native directory picker is not available in this environment.",
    )

    fake_dialog = SimpleNamespace(askdirectory=lambda mustexist=True: existing)

    def _import_fake_tk(name: str, *args: object, **kwargs: object):
        if name == "tkinter":
            module = _FakeTkModule(_FakeTkRoot(fail_attributes=True))
            module.filedialog = fake_dialog
            return module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_fake_tk)
    assert _STREAMLIT_MODULES.app._pick_root_directory_via_tk() == (existing, None)

    def _import_broken_tk(name: str, *args: object, **kwargs: object):
        if name == "tkinter":
            module = _FakeTkModule(RuntimeError("tk boom"))
            module.filedialog = fake_dialog
            return module
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _import_broken_tk)
    selected, error = _STREAMLIT_MODULES.app._pick_root_directory_via_tk()
    assert selected is None
    assert error == "Unable to open native directory picker: tk boom"


def test_streamlit_chat_picker_fallback_and_clear_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    existing = str(tmp_path.resolve())

    monkeypatch.setattr(_STREAMLIT_MODULES.app, "_running_in_wsl", lambda: True)
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_pick_root_directory_via_powershell",
        lambda: (existing, None),
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_pick_root_directory_via_tk", lambda: (None, "unused")
    )
    assert _STREAMLIT_MODULES.app._pick_root_directory() == (existing, None)

    monkeypatch.setattr(
        _STREAMLIT_MODULES.app,
        "_pick_root_directory_via_powershell",
        lambda: (None, "retry"),
    )
    monkeypatch.setattr(
        _STREAMLIT_MODULES.app, "_pick_root_directory_via_tk", lambda: (existing, None)
    )
    assert _STREAMLIT_MODULES.app._pick_root_directory() == (existing, None)

    runtime = _make_runtime(root_path=existing)
    record = _make_record(root_path=existing)
    _STREAMLIT_MODULES.app._apply_runtime_root(
        config=_STREAMLIT_MODULES.package.TextualChatConfig(),
        preferences=_STREAMLIT_MODULES.app.StreamlitPreferences(),
        record=record,
        runtime=runtime,
        resolved_root=None,
    )
    assert record.transcript[-1].text == "Workspace root cleared."
    assert runtime.root_path is None


def test_chat_runtime_available_tool_names_and_context_limits(tmp_path: Path) -> None:
    names = _CHAT_RUNTIME_MODULE.build_available_tool_names()
    specs = _CHAT_RUNTIME_MODULE.build_available_tool_specs()
    assert names
    assert names == set(specs)

    config = _STREAMLIT_MODULES.package.TextualChatConfig().model_copy(
        update={
            "session": _STREAMLIT_MODULES.package.TextualChatConfig().session.model_copy(
                update={"max_context_tokens": 25}
            )
        }
    )
    context = _CHAT_RUNTIME_MODULE.build_chat_context(
        root_path=tmp_path,
        config=config,
        app_name="streamlit-chat",
    )
    assert context.workspace == str(tmp_path)
    assert context.metadata["tool_limits"]["max_read_file_chars"] == 100
    assert context.metadata["source_filters"] == config.source_filters.model_dump(
        mode="json"
    )
