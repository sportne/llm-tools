from __future__ import annotations

import queue
import runpy
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal

import pytest
from pydantic import BaseModel

from llm_tools.apps.assistant_app import app as nicegui_app_module
from llm_tools.apps.assistant_app.app import (
    NICEGUI_APPROVAL_LABELS,
    NICEGUI_APPROVAL_OPTIONS,
    NICEGUI_PROVIDER_OPTIONS,
    _branding_favicon_href,
    _branding_head_html,
    _can_admin_disable_user,
    _composer_action_icon,
    _default_protection_corrections_path,
    _discover_model_names,
    _event_payload_text,
    _extract_model_names_from_models_payload,
    _first_nonempty_text,
    _format_information_security_label,
    _format_information_security_level,
    _format_workbench_duration,
    _is_admin_user,
    _is_tool_url_setting,
    _models_endpoint_url,
    _parse_information_security_categories,
    _protection_corpus_readiness_text,
    _provider_api_key_env_var,
    _runtime_summary_parts,
    _runtime_summary_text,
    _sidebar_container_classes,
    _workbench_container_classes,
    build_assistant_ui,
    build_parser,
    clear_hosted_session,
    main,
    resolve_assistant_config,
    resolve_root_argument,
)
from llm_tools.apps.assistant_app.auth import LocalAuthProvider, validate_hosted_startup
from llm_tools.apps.assistant_app.controller import (
    PROVIDER_API_KEY_FIELD,
    NiceGUIActiveTurnHandle,
    NiceGUIChatController,
    NiceGUIQueuedEvent,
    _interaction_protocol,
    _nicegui_protection_is_ready,
    _remember_status,
    _seconds_between_iso,
    _serialize_workflow_event,
    _workbench_inspector_title,
    _worker_resume_harness,
    _worker_run_harness,
    _worker_run_turn,
)
from llm_tools.apps.assistant_app.models import (
    AssistantBranding,
    NiceGUIHostedConfig,
    NiceGUIPreferences,
    NiceGUIRuntimeConfig,
)
from llm_tools.apps.assistant_app.store import SQLiteNiceGUIChatStore
from llm_tools.apps.assistant_config import AssistantConfig
from llm_tools.apps.chat_config import ProviderPreset
from llm_tools.harness_api import ApprovalResolution
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import SideEffectClass, ToolInvocationRequest
from llm_tools.workflow_api import (
    ChatMessage,
    ChatWorkflowInspectorEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
    ProtectionConfig,
    ProtectionPendingPrompt,
)


class _FakeProvider:
    def __init__(self, responses: list[ParsedModelResponse]) -> None:
        self._responses = list(responses)

    def run(
        self,
        *,
        adapter: ActionEnvelopeAdapter,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> ParsedModelResponse:
        del adapter, messages, response_model, request_params
        return self._responses.pop(0)

    def run_structured(
        self,
        *,
        messages: list[dict[str, Any]],
        response_model: type[BaseModel],
        request_params: dict[str, Any] | None = None,
    ) -> object:
        del messages, request_params
        return response_model.model_validate({"answer": "forced final"})

    def uses_staged_schema_protocol(self) -> bool:
        return False

    def run_text(
        self,
        *,
        messages: list[dict[str, Any]],
        request_params: dict[str, Any] | None = None,
    ) -> str:
        del messages, request_params
        return "forced final"


class _FakeEvent:
    def __init__(self, args: object) -> None:
        self.args = args


class _ProtocolProvider(_FakeProvider):
    def __init__(
        self,
        *,
        prompt_tools: bool = False,
        staged_schema: bool = False,
    ) -> None:
        super().__init__([])
        self._prompt_tools = prompt_tools
        self._staged_schema = staged_schema

    def uses_prompt_tool_protocol(self) -> bool:
        return self._prompt_tools

    def uses_staged_schema_protocol(self) -> bool:
        return self._staged_schema


def _controller(
    tmp_path: Path,
    provider: _FakeProvider,
) -> NiceGUIChatController:
    store = _chat_store(tmp_path)
    store.initialize()
    return NiceGUIChatController(
        store=store,
        config=AssistantConfig(),
        root_path=tmp_path,
        provider_factory=lambda _runtime: provider,
    )


def _chat_store(tmp_path: Path) -> SQLiteNiceGUIChatStore:
    return SQLiteNiceGUIChatStore(
        tmp_path / "chat.sqlite3",
        db_key_file=tmp_path / "db.key",
        user_key_file=tmp_path / "user-kek.key",
    )


def _drain_until_idle(controller: NiceGUIChatController) -> None:
    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline:
        controller.drain_events()
        if not controller.active_turns:
            controller.drain_events()
            return
        time.sleep(0.01)
    raise AssertionError("turn did not finish")


def test_package_import_and_cli_parser() -> None:
    import llm_tools.apps.assistant_app as assistant_app

    parser = build_parser()
    args = parser.parse_args(
        [
            "--provider",
            "ollama",
            "--model",
            "qwen3:8b",
            "--provider-mode-strategy",
            "json",
            "--api-base-url",
            "http://127.0.0.1:11434/v1",
            "--db-path",
            "chat.sqlite3",
        ]
    )
    config = resolve_assistant_config(args)

    assert assistant_app.main is not None
    assert config.llm.model_name == "qwen3:8b"
    assert config.llm.provider_mode_strategy.value == "json"
    assert args.db_path == Path("chat.sqlite3")
    assert args.auth_mode == "local"

    no_auth_args = parser.parse_args(["--auth-mode", "none"])
    assert no_auth_args.auth_mode == "none"


def test_branding_defaults_and_head_metadata() -> None:
    branding = AssistantBranding()

    href = _branding_favicon_href(branding)
    head_html = _branding_head_html(branding)

    assert branding.app_name == "LLM Tools Assistant"
    assert branding.short_name == "Assistant"
    assert href.startswith("data:image/svg+xml,")
    assert "<title>LLM Tools Assistant</title>" in head_html
    assert 'rel="icon"' in head_html
    assert "viewBox" in href


def test_branding_rejects_empty_values() -> None:
    with pytest.raises(ValueError):
        AssistantBranding(app_name=" ")


def test_hosted_startup_validation() -> None:
    loopback = validate_hosted_startup(
        auth_mode="none",
        host="127.0.0.1",
        public_base_url=None,
        tls_certfile=None,
        tls_keyfile=None,
        allow_insecure_hosted_secrets=False,
    )
    assert loopback.config.auth_mode == "none"
    assert loopback.config.secret_entry_enabled is True

    hosted_http = validate_hosted_startup(
        auth_mode="local",
        host="192.0.2.1",
        public_base_url="http://example.test",
        tls_certfile=None,
        tls_keyfile=None,
        allow_insecure_hosted_secrets=False,
    )
    assert hosted_http.config.secret_entry_enabled is False
    assert hosted_http.config.insecure_hosted_warning

    with pytest.raises(ValueError, match="requires --auth-mode local"):
        validate_hosted_startup(
            auth_mode="none",
            host="192.0.2.1",
            public_base_url=None,
            tls_certfile=None,
            tls_keyfile=None,
            allow_insecure_hosted_secrets=False,
        )


def test_hosted_storage_helpers(monkeypatch: pytest.MonkeyPatch) -> None:
    storage = SimpleNamespace(user={})
    monkeypatch.setattr(
        nicegui_app_module, "nicegui_app", SimpleNamespace(storage=storage)
    )

    assert nicegui_app_module._hosted_storage_values() == (None, None)

    nicegui_app_module._set_hosted_storage_values("session-1", "token-1")
    assert nicegui_app_module._hosted_storage_values() == ("session-1", "token-1")

    nicegui_app_module._clear_hosted_storage_values()
    assert nicegui_app_module._hosted_storage_values() == (None, None)


def test_clear_hosted_session_revokes_and_clears_storage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = SimpleNamespace(user={})
    monkeypatch.setattr(
        nicegui_app_module, "nicegui_app", SimpleNamespace(storage=storage)
    )
    store = _chat_store(tmp_path)
    store.initialize()
    auth = LocalAuthProvider(store)
    password = "admin-" + "value"
    user = auth.create_user(username="admin", password=password, role="admin")
    session_id, token = auth.create_session(user.user_id)
    nicegui_app_module._set_hosted_storage_values(session_id, token)

    clear_hosted_session(auth)

    assert nicegui_app_module._hosted_storage_values() == (None, None)
    assert auth.user_for_session(session_id, token) is None


def test_admin_user_helpers_prevent_self_disable(tmp_path: Path) -> None:
    store = _chat_store(tmp_path)
    store.initialize()
    auth = LocalAuthProvider(store)
    password = "admin-" + "value"
    admin = auth.create_user(username="admin", password=password, role="admin")
    other_admin = auth.create_user(
        username="other-admin", password=password, role="admin"
    )
    user = auth.create_user(username="user", password=password, role="user")

    assert _is_admin_user(admin) is True
    assert _is_admin_user(user) is False
    assert _is_admin_user(None) is False
    assert _can_admin_disable_user(admin, admin) is False
    assert _can_admin_disable_user(admin, other_admin) is True
    assert _can_admin_disable_user(admin, user) is True
    assert _can_admin_disable_user(user, admin) is False


def test_hosted_page_routes_first_admin_login_and_chat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    storage = SimpleNamespace(user={})
    monkeypatch.setattr(
        nicegui_app_module, "nicegui_app", SimpleNamespace(storage=storage)
    )
    store = _chat_store(tmp_path)
    store.initialize()
    auth = LocalAuthProvider(store)
    calls: list[str] = []
    branded_titles: list[str] = []
    monkeypatch.setattr(
        nicegui_app_module,
        "_apply_branding_head",
        lambda branding: branded_titles.append(branding.app_name),
    )
    monkeypatch.setattr(
        nicegui_app_module,
        "render_first_admin_page",
        lambda _auth: calls.append("first-admin"),
    )
    monkeypatch.setattr(
        nicegui_app_module,
        "render_login_page",
        lambda _auth: calls.append("login"),
    )
    monkeypatch.setattr(
        nicegui_app_module,
        "build_assistant_ui",
        lambda controller: calls.append(f"chat:{controller.current_user.username}"),
    )

    nicegui_app_module.render_hosted_nicegui_page(
        store=store,
        config=AssistantConfig(),
        root_path=tmp_path,
        hosted_config=NiceGUIHostedConfig(auth_mode="local"),
        auth_provider=auth,
    )

    admin_password = "admin-" + "value"
    user = auth.create_user(username="admin", password=admin_password, role="admin")
    nicegui_app_module.render_hosted_nicegui_page(
        store=store,
        config=AssistantConfig(),
        root_path=tmp_path,
        hosted_config=NiceGUIHostedConfig(auth_mode="local"),
        auth_provider=auth,
    )

    session_id, token = auth.create_session(user.user_id)
    nicegui_app_module._set_hosted_storage_values(session_id, token)
    nicegui_app_module.render_hosted_nicegui_page(
        store=store,
        config=AssistantConfig(),
        root_path=tmp_path,
        hosted_config=NiceGUIHostedConfig(auth_mode="local"),
        auth_provider=auth,
    )

    assert calls == ["first-admin", "login", "chat:admin"]
    assert branded_titles == [
        "LLM Tools Assistant",
        "LLM Tools Assistant",
        "LLM Tools Assistant",
    ]


def test_composer_text_helpers_accept_server_and_js_payloads() -> None:
    assert _first_nonempty_text("", None, "  hello  ") == "  hello  "
    assert _first_nonempty_text("   ", "") == ""
    assert _event_payload_text(_FakeEvent("typed text")) == "typed text"
    assert _event_payload_text(_FakeEvent({"value": "from value"})) == "from value"
    assert (
        _event_payload_text(_FakeEvent({"target": {"value": "from target"}}))
        == "from target"
    )
    assert _event_payload_text(_FakeEvent({"target": {}})) == ""
    assert _event_payload_text(_FakeEvent(123)) == ""


def test_layout_container_class_helpers() -> None:
    assert _sidebar_container_classes(collapsed=False) == "llmt-sidebar"
    assert _sidebar_container_classes(collapsed=True) == "llmt-sidebar collapsed"
    assert _workbench_container_classes() == "llmt-workbench"
    assert _workbench_container_classes(open=False) == "llmt-workbench closed"


def test_composer_runtime_helpers() -> None:
    runtime = NiceGUIRuntimeConfig(model_name="model-a", root_path="/repo")
    assert _runtime_summary_parts(runtime) == [
        ("provider", "ollama"),
        ("model", "model-a"),
        ("mode", "auto"),
        ("workspace", "/repo"),
    ]
    assert _runtime_summary_text(runtime) == "ollama | model-a | auto | /repo"
    assert _composer_action_icon(busy=False) == "send"
    assert _composer_action_icon(busy=True) == "stop"
    assert _format_workbench_duration(None) == "duration unknown"
    assert _format_workbench_duration(0.25) == "250 ms"
    assert _format_workbench_duration(1.25) == "1.2 s"
    assert _format_workbench_duration(61.0) == "1m 1s"
    assert NICEGUI_PROVIDER_OPTIONS == ["ollama", "custom_openai_compatible"]
    assert NICEGUI_APPROVAL_OPTIONS == [
        SideEffectClass.LOCAL_READ,
        SideEffectClass.LOCAL_WRITE,
        SideEffectClass.EXTERNAL_READ,
        SideEffectClass.EXTERNAL_WRITE,
    ]


def test_information_security_helpers(tmp_path: Path) -> None:
    assert _parse_information_security_categories(" TRIVIAL\nMINOR,TRIVIAL\n") == [
        "TRIVIAL",
        "MINOR",
    ]
    assert _default_protection_corrections_path(str(tmp_path)) == str(
        tmp_path / ".llm-tools-protection-corrections.json"
    )
    disabled = ProtectionConfig(enabled=False)
    assert _format_information_security_level(disabled) == "Undefined"
    assert _format_information_security_label(disabled).endswith("Undefined")
    incomplete = ProtectionConfig(
        enabled=True,
        allowed_sensitivity_labels=["TRIVIAL", "MINOR"],
    )
    assert _format_information_security_level(incomplete) == "Undefined"
    configured = ProtectionConfig(
        enabled=True,
        document_paths=[str(tmp_path)],
        allowed_sensitivity_labels=["TRIVIAL", "MINOR"],
    )
    assert _format_information_security_level(configured) == "Undefined"
    guidance = tmp_path / "guidance.md"
    guidance.write_text("TRIVIAL information may be discussed.", encoding="utf-8")
    assert _format_information_security_level(configured) == "TRIVIAL/MINOR"


def test_protection_readiness_requires_labels_and_corpus(tmp_path: Path) -> None:
    document = tmp_path / "guidance.md"
    document.write_text("TRIVIAL information may be discussed.", encoding="utf-8")
    ready = ProtectionConfig(
        enabled=True,
        document_paths=[str(tmp_path)],
        allowed_sensitivity_labels=["TRIVIAL"],
    )
    assert _nicegui_protection_is_ready(ready) is True
    assert _protection_corpus_readiness_text(ready).startswith("Ready:")
    no_labels = ready.model_copy(update={"allowed_sensitivity_labels": []})
    assert _nicegui_protection_is_ready(no_labels) is False
    assert "Add at least one allowed category" in _protection_corpus_readiness_text(
        no_labels
    )


def test_workbench_inspector_titles_are_directional() -> None:
    assert (
        _workbench_inspector_title(
            turn_number=1, round_index=3, kind="provider_messages"
        )
        == "T1R3: User To LLM"
    )
    assert (
        _workbench_inspector_title(
            turn_number=2, round_index=1, kind="provider_response"
        )
        == "T2R1: From LLM"
    )
    assert (
        _workbench_inspector_title(turn_number=2, round_index=1, kind="parsed_response")
        == "T2R1: Parsed LLM Response"
    )
    assert (
        _workbench_inspector_title(turn_number=2, round_index=1, kind="tool_execution")
        == "T2R1: Tool To LLM"
    )
    assert NICEGUI_APPROVAL_LABELS["local_write"] == "Local write"
    assert (
        _provider_api_key_env_var(
            ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
            None,
        )
        == "OPENAI_API_KEY"
    )
    assert _provider_api_key_env_var(ProviderPreset.OLLAMA, None) == ""
    assert _is_tool_url_setting("BITBUCKET_BASE_URL") is True
    assert _is_tool_url_setting("GITLAB_API_TOKEN") is False


def test_controller_helper_edges(tmp_path: Path) -> None:
    assert _seconds_between_iso(None, "2026-04-26T10:00:00+00:00") is None
    assert _seconds_between_iso("bad date", "2026-04-26T10:00:00+00:00") is None
    assert (
        _seconds_between_iso(
            "2026-04-26T10:00:02+00:00",
            "2026-04-26T10:00:01+00:00",
        )
        == 0.0
    )

    controller = _controller(tmp_path, _FakeProvider([]))
    turn_state = controller.active_turn_state
    _remember_status(turn_state, "")
    assert turn_state.status_history == []
    _remember_status(turn_state, "thinking")
    _remember_status(turn_state, "thinking")
    assert turn_state.status_history == ["thinking"]

    assert _interaction_protocol(_ProtocolProvider(prompt_tools=True)) == "prompt_tools"
    assert _interaction_protocol(_ProtocolProvider(staged_schema=True)) == "staged_json"
    assert _interaction_protocol(_ProtocolProvider()) == "native_tools"


def test_model_discovery_payload_helpers() -> None:
    assert _models_endpoint_url("http://127.0.0.1:11434/v1/") == (
        "http://127.0.0.1:11434/v1/models"
    )
    assert _models_endpoint_url("  ") == ""
    assert _extract_model_names_from_models_payload(
        {
            "data": [
                {"id": "z-model"},
                {"id": "models/a-model"},
                {"id": "z-model"},
            ]
        }
    ) == ["a-model", "z-model"]
    assert _extract_model_names_from_models_payload(
        {"models": [{"name": "ollama-a"}, {"name": "ollama-b"}]}
    ) == ["ollama-a", "ollama-b"]
    assert _extract_model_names_from_models_payload({"data": "not-list"}) == []
    assert _extract_model_names_from_models_payload(["not-dict"]) == []
    assert _extract_model_names_from_models_payload({"data": ["plain-model"]}) == [
        "plain-model"
    ]


def test_model_discovery_fetches_openai_compatible_models(
    monkeypatch: Any,
) -> None:
    import llm_tools.apps.assistant_app.app as app_module

    class _Response:
        def __enter__(self) -> _Response:
            return self

        def __exit__(self, *args: object) -> None:
            del args

        def read(self) -> bytes:
            return b'{"data": [{"id": "model-b"}, {"id": "model-a"}]}'

    calls: list[tuple[str, str | None, float]] = []

    def fake_urlopen(request: Any, *, timeout: float) -> _Response:
        calls.append(
            (
                request.full_url,
                request.get_header("Authorization"),
                timeout,
            )
        )
        return _Response()

    monkeypatch.setenv("CUSTOM_OPENAI_KEY", "test-key")
    monkeypatch.setattr(app_module, "urlopen", fake_urlopen)

    assert _discover_model_names(
        provider=ProviderPreset.CUSTOM_OPENAI_COMPATIBLE,
        base_url="https://example.test/v1",
        api_key="typed-key",
        api_key_env_var="CUSTOM_OPENAI_KEY",
        timeout=1.5,
    ) == ["model-a", "model-b"]
    assert calls == [
        ("https://example.test/v1/models", "Bearer typed-key", 1.5),
    ]


def test_model_discovery_rejects_invalid_or_failed_endpoints(
    monkeypatch: Any,
) -> None:
    import llm_tools.apps.assistant_app.app as app_module

    def failing_urlopen(request: Any, *, timeout: float) -> object:
        del request, timeout
        raise OSError("offline")

    monkeypatch.setattr(app_module, "urlopen", failing_urlopen)

    assert _discover_model_names(provider=ProviderPreset.OLLAMA, base_url=None) == []
    assert _discover_model_names(provider=ProviderPreset.OLLAMA, base_url="  ") == []
    assert (
        _discover_model_names(provider=ProviderPreset.OLLAMA, base_url="file:///tmp")
        == []
    )
    assert (
        _discover_model_names(
            provider=ProviderPreset.OLLAMA,
            base_url="http://127.0.0.1:11434/v1",
        )
        == []
    )


def test_nicegui_preferences_default_to_closed_workbench() -> None:
    prefs = NiceGUIPreferences()

    assert prefs.theme_mode == "light"
    assert prefs.workbench_open is False


def test_cli_config_resolution_covers_runtime_overrides(tmp_path: Path) -> None:
    parser = build_parser()
    args = parser.parse_args(
        [
            str(tmp_path),
            "--directory",
            str(tmp_path / "override"),
            "--provider",
            "ollama",
            "--model",
            "model-b",
            "--provider-mode-strategy",
            "prompt_tools",
            "--temperature",
            "0.2",
            "--api-base-url",
            "http://127.0.0.1:11434/v1",
            "--api-key-env-var",
            "OLLAMA_API_KEY",
            "--max-context-tokens",
            "1000",
            "--max-tool-round-trips",
            "3",
            "--max-tool-calls-per-round",
            "1",
            "--max-total-tool-calls-per-turn",
            "4",
            "--max-entries-per-call",
            "5",
            "--max-recursive-depth",
            "2",
            "--max-search-matches",
            "6",
            "--max-read-lines",
            "7",
            "--max-file-size-characters",
            "800",
            "--max-tool-result-chars",
            "900",
        ]
    )

    config = resolve_assistant_config(args)
    root = resolve_root_argument(args, config)

    assert config.llm.model_name == "model-b"
    assert config.llm.api_key_env_var == "OLLAMA_API_KEY"
    assert config.llm.temperature == 0.2
    assert config.session.max_context_tokens == 1000
    assert config.tool_limits.max_tool_result_chars == 900
    assert root == (tmp_path / "override").resolve()


def test_root_resolution_uses_config_default_and_none() -> None:
    parser = build_parser()
    no_root_args = parser.parse_args([])
    config_with_root = AssistantConfig.model_validate(
        {"workspace": {"default_root": "."}}
    )

    assert resolve_root_argument(no_root_args, AssistantConfig()) is None
    assert resolve_root_argument(no_root_args, config_with_root) == Path(".").resolve()


def test_main_resolves_arguments_and_delegates_run(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("llm_tools.apps.assistant_app.app.run_assistant_app", fake_run)

    assert (
        main(
            [
                str(tmp_path),
                "--model",
                "model-c",
                "--db-path",
                "chat.sqlite3",
                "--host",
                "127.0.0.1",
                "--port",
                "9999",
                "--no-browser",
            ]
        )
        == 0
    )
    assert calls[0]["root_path"] == tmp_path.resolve()
    assert calls[0]["port"] == 9999
    assert calls[0]["show"] is False


def test_main_allows_no_workspace_root(monkeypatch: Any) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr("llm_tools.apps.assistant_app.app.run_assistant_app", fake_run)

    assert main(["--no-browser"]) == 0
    assert calls[0]["root_path"] is None


def test_module_entrypoint_uses_package_main(monkeypatch: Any) -> None:
    import llm_tools.apps.assistant_app as assistant_app

    calls = []
    monkeypatch.setattr(assistant_app, "main", lambda: calls.append("called"))

    runpy.run_module("llm_tools.apps.assistant_app.__main__", run_name="__main__")

    assert calls == ["called"]


def test_app_builder_renders_with_temporary_sqlite_db(tmp_path: Path) -> None:
    provider = _FakeProvider(
        [
            ParsedModelResponse(
                final_response={
                    "answer": "unused",
                    "citations": [],
                    "confidence": 0.5,
                    "uncertainty": [],
                    "missing_information": [],
                    "follow_up_suggestions": [],
                }
            )
        ]
    )
    controller = _controller(tmp_path, provider)

    build_assistant_ui(controller)

    assert controller.active_record.summary.title == "New chat"


def test_app_builder_applies_initial_layout_preferences(tmp_path: Path) -> None:
    from nicegui import ui

    controller = _controller(tmp_path, _FakeProvider([]))
    controller.preferences.sidebar_collapsed = True
    controller.preferences.workbench_open = False

    build_assistant_ui(controller)

    elements = list(ui.context.client.elements.values())
    sidebar = [element for element in elements if "llmt-sidebar" in element.classes][-1]
    workbench = [
        element for element in elements if "llmt-workbench" in element.classes
    ][-1]
    assert "collapsed" in sidebar.classes
    assert workbench.visible is False


def test_controller_direct_answer_persists_user_and_assistant(
    tmp_path: Path,
) -> None:
    provider = _FakeProvider(
        [
            ParsedModelResponse(
                final_response={
                    "answer": "Plain answer",
                    "citations": [],
                    "confidence": 0.7,
                    "uncertainty": [],
                    "missing_information": [],
                    "follow_up_suggestions": [],
                }
            )
        ]
    )
    controller = _controller(tmp_path, provider)

    assert controller.submit_prompt("hello") is None
    _drain_until_idle(controller)

    loaded = controller.store.load_session(controller.active_session_id)
    assert loaded is not None
    assert [entry.role for entry in loaded.transcript] == ["user", "assistant"]
    assert loaded.transcript[-1].text == "Plain answer"
    assert loaded.workflow_session_state.turns
    assert loaded.token_usage is not None


def test_controller_interaction_mode_locks_after_first_message(
    tmp_path: Path,
) -> None:
    provider = _FakeProvider(
        [
            ParsedModelResponse(
                final_response={
                    "answer": "Plain answer",
                    "citations": [],
                    "confidence": 0.7,
                    "uncertainty": [],
                    "missing_information": [],
                    "follow_up_suggestions": [],
                }
            )
        ]
    )
    controller = _controller(tmp_path, provider)

    assert controller.deep_task_mode_enabled() is False
    assert controller.set_interaction_mode("deep_task") is False
    controller.set_deep_task_mode_enabled(True)
    assert controller.set_interaction_mode("deep_task") is True
    assert controller.active_record.runtime.interaction_mode == "deep_task"
    assert controller.set_interaction_mode("chat") is True
    assert controller.submit_prompt("hello") is None

    assert controller.interaction_mode_locked() is True
    assert controller.set_interaction_mode("deep_task") is False
    assert controller.active_record.runtime.interaction_mode == "chat"
    _drain_until_idle(controller)


def test_controller_persists_admin_branding(tmp_path: Path) -> None:
    controller = _controller(tmp_path, _FakeProvider([]))

    controller.set_branding(
        AssistantBranding(
            app_name="Team Assistant",
            short_name="Team",
            icon_name="hub",
            favicon_svg='<svg viewBox="0 0 1 1"></svg>',
        )
    )

    assert controller.admin_settings.branding.app_name == "Team Assistant"
    store = _chat_store(tmp_path)
    store.initialize()
    assert store.load_admin_settings().branding.short_name == "Team"


def test_controller_disabled_deep_task_mode_is_coerced_to_chat(
    tmp_path: Path,
) -> None:
    provider = _FakeProvider(
        [
            ParsedModelResponse(
                final_response={
                    "answer": "Plain answer",
                    "citations": [],
                    "confidence": 0.7,
                    "uncertainty": [],
                    "missing_information": [],
                    "follow_up_suggestions": [],
                }
            )
        ]
    )
    controller = _controller(tmp_path, provider)
    controller.set_deep_task_mode_enabled(True)
    assert controller.set_interaction_mode("deep_task") is True
    controller.set_deep_task_mode_enabled(False)
    assert controller.active_record.runtime.interaction_mode == "chat"

    controller.active_record.runtime.interaction_mode = "deep_task"
    assert controller.submit_prompt("hello") is None
    assert controller.active_record.runtime.interaction_mode == "chat"
    _drain_until_idle(controller)


def test_controller_deep_task_runs_harness_and_persists_summary(
    tmp_path: Path,
) -> None:
    provider = _FakeProvider(
        [
            ParsedModelResponse(
                final_response={
                    "answer": "Deep task answer",
                    "citations": [],
                    "confidence": 0.7,
                    "uncertainty": [],
                    "missing_information": [],
                    "follow_up_suggestions": [],
                }
            )
        ]
    )
    controller = _controller(tmp_path, provider)
    controller.set_deep_task_mode_enabled(True)
    assert controller.set_interaction_mode("deep_task") is True

    assert controller.submit_prompt("investigate this") is None
    _drain_until_idle(controller)

    roles = [entry.role for entry in controller.active_record.transcript]
    assert roles == ["user", "system", "assistant"]
    assert "Started deep task" in controller.active_record.transcript[1].text
    assert "Deep task session:" in controller.active_record.transcript[2].text
    assert controller.active_record.workbench_items[-1].kind == "result"
    assert controller.active_record.workbench_items[-1].title.endswith("Deep Task")


def test_controller_restores_deep_task_pending_approval(
    tmp_path: Path,
) -> None:
    provider = _FakeProvider(
        [
            ParsedModelResponse(
                invocations=[
                    {"tool_name": "list_directory", "arguments": {"path": "."}},
                ]
            )
        ]
    )
    controller = _controller(tmp_path, provider)
    controller.set_deep_task_mode_enabled(True)
    controller.active_record.runtime.enabled_tools = ["list_directory"]
    controller.active_record.runtime.require_approval_for = {SideEffectClass.LOCAL_READ}
    assert controller.set_interaction_mode("deep_task") is True
    session_id = controller.active_session_id

    assert controller.submit_prompt("inspect the workspace") is None
    _drain_until_idle(controller)

    assert controller.turn_state_for(session_id).pending_approval is not None
    assert controller.turn_state_for(session_id).pending_harness_session_id is not None

    reloaded = NiceGUIChatController(
        store=controller.store,
        config=AssistantConfig(),
        root_path=tmp_path,
        provider_factory=lambda _runtime: _FakeProvider([]),
    )
    reloaded.select_session(session_id)

    assert reloaded.turn_state_for(session_id).pending_approval is not None
    assert reloaded.turn_state_for(session_id).pending_harness_session_id is not None


def test_controller_deep_task_approval_resume_build_failure_records_error(
    tmp_path: Path,
) -> None:
    controller = _controller(
        tmp_path,
        _FakeProvider([]),
    )
    controller.provider_factory = lambda _runtime: (_ for _ in ()).throw(
        RuntimeError("provider unavailable")
    )
    session_id = controller.active_session_id
    controller.turn_state_for(session_id).pending_harness_session_id = "harness-1"

    assert controller.resolve_approval(approved=True) is False
    assert controller.active_record.transcript[-1].role == "error"
    assert "provider unavailable" in controller.active_record.transcript[-1].text


def test_controller_session_management_filters_and_temporary_chats(
    tmp_path: Path,
) -> None:
    controller = _controller(
        tmp_path,
        _FakeProvider(
            [
                ParsedModelResponse(
                    final_response={
                        "answer": "unused",
                        "citations": [],
                        "confidence": 0.5,
                        "uncertainty": [],
                        "missing_information": [],
                        "follow_up_suggestions": [],
                    }
                )
            ]
        ),
    )
    original_id = controller.active_session_id
    controller.rename_session(original_id, "Original repo chat")

    temporary = controller.create_session(temporary=True)

    assert temporary.summary.temporary is True
    assert temporary.summary.session_id not in [
        summary.session_id for summary in controller.list_session_summaries()
    ]
    assert controller.select_session("missing") is False
    assert controller.select_session(original_id) is True
    assert [
        summary.title for summary in controller.list_session_summaries(query="repo")
    ] == ["Original repo chat"]

    controller.delete_session(temporary.summary.session_id)

    assert temporary.summary.session_id not in controller.sessions


def test_controller_session_secrets_are_in_memory_and_session_scoped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    ambient_value = "ambient-" + "value"
    monkeypatch.setenv("GITLAB_API_TOKEN", ambient_value)
    controller = _controller(tmp_path, _FakeProvider([]))
    first_session_id = controller.active_session_id
    provider_value = "provider-value"
    tool_value = "tool-value"
    controller.set_session_secret(PROVIDER_API_KEY_FIELD, provider_value)
    controller.set_session_secret("GITLAB_API_TOKEN", tool_value)
    controller.active_record.runtime.tool_urls = {
        "GITLAB_BASE_URL": "https://gitlab.example.test"
    }

    assert controller.provider_api_key() == provider_value
    assert controller.session_tool_env() == {"GITLAB_API_TOKEN": tool_value}
    assert controller.effective_tool_env()["GITLAB_API_TOKEN"] == tool_value
    assert controller.effective_tool_env()["GITLAB_API_TOKEN"] != ambient_value
    assert (
        controller.effective_tool_env()["GITLAB_BASE_URL"]
        == "https://gitlab.example.test"
    )
    assert controller.tool_env_overrides() == {
        "GITLAB_API_TOKEN": tool_value,
        "GITLAB_BASE_URL": "https://gitlab.example.test",
    }

    second = controller.create_session(temporary=False)
    assert controller.provider_api_key() is None
    assert controller.session_tool_env() == {}

    controller.select_session(first_session_id)
    controller.clear_session_secret("GITLAB_API_TOKEN")
    assert controller.session_tool_env() == {}
    assert controller.provider_api_key() == provider_value

    reloaded = NiceGUIChatController(
        store=controller.store,
        config=AssistantConfig(),
        root_path=tmp_path,
        provider_factory=lambda _runtime: _FakeProvider([]),
    )
    reloaded.select_session(first_session_id)
    assert reloaded.provider_api_key() is None
    assert reloaded.session_tool_env() == {}
    assert second.summary.session_id in controller.sessions


def test_controller_hosted_secrets_are_memory_only_and_env_isolated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GITLAB_API_TOKEN", "ambient-token")
    store = _chat_store(tmp_path)
    store.initialize()
    auth = LocalAuthProvider(store)
    credential_value = "secret"
    user = auth.create_user(username="admin", password=credential_value, role="admin")
    controller = NiceGUIChatController(
        store=store,
        config=AssistantConfig(),
        root_path=tmp_path,
        provider_factory=lambda _runtime: _FakeProvider([]),
        current_user=user,
        auth_provider=auth,
    )
    session_id = controller.active_session_id

    assert "GITLAB_API_TOKEN" not in controller.effective_tool_env()

    controller.set_session_secret(PROVIDER_API_KEY_FIELD, "provider-token")
    controller.set_session_secret("GITLAB_API_TOKEN", "session-token")

    assert controller.provider_api_key() == "provider-token"
    assert controller.session_tool_env() == {"GITLAB_API_TOKEN": "session-token"}

    reloaded = NiceGUIChatController(
        store=store,
        config=AssistantConfig(),
        root_path=tmp_path,
        provider_factory=lambda _runtime: _FakeProvider([]),
        current_user=user,
        auth_provider=auth,
    )
    reloaded.select_session(session_id)
    assert reloaded.provider_api_key() is None
    assert reloaded.session_tool_env() == {}


def test_controller_switch_database_copies_durable_sessions(
    tmp_path: Path, monkeypatch: Any
) -> None:
    from llm_tools.apps.assistant_app import controller as controller_module

    remembered_paths: list[Path] = []
    monkeypatch.setattr(
        controller_module,
        "remember_default_db_path",
        lambda path: remembered_paths.append(Path(path)),
    )
    controller = _controller(tmp_path, _FakeProvider([]))
    session_id = controller.active_session_id
    controller.rename_session(session_id, "Migrated chat")
    new_db_path = tmp_path / "nested" / "new-chat.sqlite3"

    controller.switch_database(new_db_path)

    assert controller.store.db_path == new_db_path
    loaded = controller.store.load_session(session_id)
    assert loaded is not None
    assert loaded.summary.title == "Migrated chat"
    assert controller.store.load_preferences().active_session_id == session_id
    assert remembered_paths == [new_db_path]


def test_hosted_controller_cannot_switch_database(tmp_path: Path) -> None:
    store = _chat_store(tmp_path)
    store.initialize()
    auth = LocalAuthProvider(store)
    password = "hosted-" + "value"
    user = auth.create_user(username="admin", password=password, role="admin")
    controller = NiceGUIChatController(
        store=store,
        config=AssistantConfig(),
        root_path=tmp_path,
        provider_factory=lambda _runtime: _FakeProvider([]),
        current_user=user,
        auth_provider=auth,
    )

    with pytest.raises(RuntimeError, match="Hosted sessions cannot switch"):
        controller.switch_database(tmp_path / "new.sqlite3")


def test_controller_loads_existing_session_and_deletes_active_session(
    tmp_path: Path,
) -> None:
    store = _chat_store(tmp_path)
    store.initialize()
    stored = store.create_session(NiceGUIRuntimeConfig(), title="Stored chat")

    controller = NiceGUIChatController(
        store=store,
        config=AssistantConfig(),
        root_path=tmp_path,
        provider_factory=lambda _runtime: _FakeProvider([]),
    )
    controller.sessions.pop(stored.summary.session_id)
    controller.turn_states.pop(stored.summary.session_id)

    assert controller.select_session(stored.summary.session_id) is True
    assert controller.active_record.summary.title == "Stored chat"

    controller.delete_session(stored.summary.session_id)

    assert controller.active_session_id != stored.summary.session_id


def test_controller_provider_creation_failure_persists_error(tmp_path: Path) -> None:
    store = _chat_store(tmp_path)
    store.initialize()
    controller = NiceGUIChatController(
        store=store,
        config=AssistantConfig(),
        root_path=tmp_path,
        provider_factory=lambda _runtime: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    error = controller.submit_prompt("hello")

    assert error == "boom"
    assert [entry.role for entry in controller.active_record.transcript] == [
        "user",
        "error",
    ]


def test_controller_empty_submit_and_idle_controls(tmp_path: Path) -> None:
    controller = _controller(tmp_path, _FakeProvider([]))

    assert controller.submit_prompt("   ") == "Enter a message first."
    assert controller.resolve_approval(approved=True) is False
    assert controller.cancel_active_turn() is False
    assert (
        controller.submit_protection_accept() == "No protection challenge is pending."
    )
    assert (
        controller.submit_protection_overrule(
            expected_sensitivity_label="MINOR",
            rationale="safe enough",
        )
        == "No protection challenge is pending."
    )


def test_controller_cancel_active_turn_branches(tmp_path: Path) -> None:
    controller = _controller(tmp_path, _FakeProvider([]))
    session_id = controller.active_session_id

    class _Runner:
        cancelled = False

        def cancel(self) -> None:
            self.cancelled = True

    runner = _Runner()
    controller.active_turns[session_id] = NiceGUIActiveTurnHandle(
        session_id=session_id,
        mode="chat",
        event_queue=queue.Queue(),
        thread=threading.Thread(target=lambda: None),
        turn_number=1,
        runner=runner,  # type: ignore[arg-type]
    )

    assert controller.cancel_active_turn() is True
    assert runner.cancelled is True

    class _FailingHarnessService:
        def stop_session(self, request: object) -> None:
            del request
            raise RuntimeError("stop failed")

    controller.active_turns[session_id] = NiceGUIActiveTurnHandle(
        session_id=session_id,
        mode="deep_task",
        event_queue=queue.Queue(),
        thread=threading.Thread(target=lambda: None),
        turn_number=2,
        harness_service=_FailingHarnessService(),  # type: ignore[arg-type]
        harness_session_id="harness-1",
    )

    assert controller.cancel_active_turn() is False


def test_controller_worker_error_paths_enqueue_error_and_complete() -> None:
    chat_queue: queue.Queue[NiceGUIQueuedEvent] = queue.Queue()
    _worker_run_turn(
        NiceGUIActiveTurnHandle(
            session_id="session-1",
            mode="chat",
            event_queue=chat_queue,
            thread=threading.Thread(target=lambda: None),
            turn_number=1,
        )
    )
    assert chat_queue.get_nowait().kind == "error"
    assert chat_queue.get_nowait().kind == "complete"

    harness_queue: queue.Queue[NiceGUIQueuedEvent] = queue.Queue()
    _worker_run_harness(
        NiceGUIActiveTurnHandle(
            session_id="session-1",
            mode="deep_task",
            event_queue=harness_queue,
            thread=threading.Thread(target=lambda: None),
            turn_number=2,
        )
    )
    assert harness_queue.get_nowait().kind == "error"
    assert harness_queue.get_nowait().kind == "complete"

    resume_queue: queue.Queue[NiceGUIQueuedEvent] = queue.Queue()
    _worker_resume_harness(
        NiceGUIActiveTurnHandle(
            session_id="session-1",
            mode="deep_task",
            event_queue=resume_queue,
            thread=threading.Thread(target=lambda: None),
            turn_number=3,
        ),
        ApprovalResolution.APPROVE,
    )
    assert resume_queue.get_nowait().kind == "error"
    assert resume_queue.get_nowait().kind == "complete"


def test_controller_protection_accept_and_overrule_feedback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path, _FakeProvider([]))
    pending = ProtectionPendingPrompt(
        original_user_message="show the protected plan",
        reasoning="The request may exceed the allowed category.",
        predicted_sensitivity_label="MAJOR",
    )
    controller.active_record.workflow_session_state.pending_protection_prompt = pending
    submitted: list[str] = []

    def capture_prompt(prompt: str, **_kwargs: object) -> None:
        submitted.append(prompt)

    monkeypatch.setattr(
        controller,
        "submit_prompt",
        capture_prompt,
    )

    assert controller.pending_protection_prompt() == pending
    assert controller.submit_protection_accept() is None
    assert submitted[-1] == "analysis_is_correct: true"
    assert (
        controller.submit_protection_overrule(
            expected_sensitivity_label="MINOR",
            rationale="This is already cleared for MINOR.",
        )
        is None
    )
    assert "analysis_is_correct: false" in submitted[-1]
    assert "expected_sensitivity_label: MINOR" in submitted[-1]
    assert controller.active_turn_state.queued_follow_up_prompt == (
        "show the protected plan"
    )


def test_controller_protection_overrule_requires_fields(tmp_path: Path) -> None:
    controller = _controller(tmp_path, _FakeProvider([]))
    controller.active_record.workflow_session_state.pending_protection_prompt = (
        ProtectionPendingPrompt(
            original_user_message="show the protected plan",
            reasoning="The request may exceed the allowed category.",
            predicted_sensitivity_label="MAJOR",
        )
    )

    assert (
        controller.submit_protection_overrule(
            expected_sensitivity_label="",
            rationale="because",
        )
        == "Expected category is required."
    )
    assert (
        controller.submit_protection_overrule(
            expected_sensitivity_label="MINOR",
            rationale="",
        )
        == "Explanation is required."
    )


def test_controller_protection_overrule_clears_queue_on_submit_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    controller = _controller(tmp_path, _FakeProvider([]))
    controller.active_record.workflow_session_state.pending_protection_prompt = (
        ProtectionPendingPrompt(
            original_user_message="show the protected plan",
            reasoning="The request may exceed the allowed category.",
            predicted_sensitivity_label="MAJOR",
        )
    )
    controller.active_turn_state.queued_follow_up_prompt = "stale prompt"

    def fail_submit(_prompt: str, **_kwargs: object) -> str:
        return "provider failed"

    monkeypatch.setattr(controller, "submit_prompt", fail_submit)

    assert (
        controller.submit_protection_overrule(
            expected_sensitivity_label="MINOR",
            rationale="This is already cleared.",
        )
        == "provider failed"
    )
    assert controller.active_turn_state.queued_follow_up_prompt is None


def test_controller_busy_turn_queues_follow_up(tmp_path: Path) -> None:
    controller = _controller(
        tmp_path,
        _FakeProvider(
            [
                ParsedModelResponse(
                    final_response={
                        "answer": "unused",
                        "citations": [],
                        "confidence": 0.5,
                        "uncertainty": [],
                        "missing_information": [],
                        "follow_up_suggestions": [],
                    }
                )
            ]
        ),
    )
    controller.active_turn_state.busy = True

    assert controller.submit_prompt("next") is None
    assert controller.active_turn_state.queued_follow_up_prompt == "next"


def test_controller_manual_events_cover_non_completed_paths(tmp_path: Path) -> None:
    controller = _controller(
        tmp_path,
        _FakeProvider(
            [
                ParsedModelResponse(
                    final_response={
                        "answer": "unused",
                        "citations": [],
                        "confidence": 0.5,
                        "uncertainty": [],
                        "missing_information": [],
                        "follow_up_suggestions": [],
                    }
                )
            ]
        ),
    )
    session_id = controller.active_session_id
    turn_state = controller.active_turn_state
    turn_state.active_turn_started_at = "2026-04-26T10:00:00+00:00"
    turn_state.last_workbench_entry_at = turn_state.active_turn_started_at

    ignored = controller._apply_queued_event(
        NiceGUIQueuedEvent(
            kind="status",
            payload=ChatWorkflowStatusEvent(status="thinking").model_dump(mode="json"),
            turn_number=1,
            session_id="missing",
        )
    )
    assert ignored is None

    controller._apply_queued_event(
        NiceGUIQueuedEvent(
            kind="status",
            payload=ChatWorkflowStatusEvent(status="thinking").model_dump(mode="json"),
            turn_number=1,
            session_id=session_id,
        )
    )
    assert turn_state.status_text == "thinking"

    turn_state.cancelling = True
    controller._apply_queued_event(
        NiceGUIQueuedEvent(
            kind="status",
            payload=ChatWorkflowStatusEvent(status="ignored").model_dump(mode="json"),
            turn_number=1,
            session_id=session_id,
        )
    )
    assert turn_state.status_text == "thinking"
    turn_state.cancelling = False

    inspector_event_times: dict[
        Literal["provider_messages", "parsed_response"], str
    ] = {
        "provider_messages": "2026-04-26T10:00:03+00:00",
        "parsed_response": "2026-04-26T10:00:05+00:00",
    }
    for kind, created_at in inspector_event_times.items():
        controller._apply_queued_event(
            NiceGUIQueuedEvent(
                kind="inspector",
                payload=ChatWorkflowInspectorEvent(
                    round_index=1,
                    kind=kind,
                    payload={"kind": kind},
                ).model_dump(mode="json"),
                turn_number=1,
                session_id=session_id,
                created_at=created_at,
            )
        )
    assert controller.active_record.inspector_state.provider_messages
    assert controller.active_record.inspector_state.parsed_responses
    assert controller.active_record.workbench_items[0].duration_seconds == 3.0
    assert controller.active_record.workbench_items[1].duration_seconds == 2.0

    continuation = ChatWorkflowResultEvent(
        result=ChatWorkflowTurnResult(
            status="needs_continuation",
            continuation_reason="Need one more step.",
            context_warning="Context was compacted.",
        )
    )
    controller._apply_queued_event(
        NiceGUIQueuedEvent(
            kind="result",
            payload=continuation.model_dump(mode="json"),
            turn_number=1,
            session_id=session_id,
        )
    )
    assert controller.active_record.transcript[-2].text == "Context was compacted."
    assert controller.active_record.transcript[-1].text == "Need one more step."

    interrupted = ChatWorkflowResultEvent(
        result=ChatWorkflowTurnResult(
            status="interrupted",
            new_messages=[
                ChatMessage(
                    role="assistant",
                    content="Partial answer",
                    completion_state="interrupted",
                )
            ],
            interruption_reason="Stopped.",
        )
    )
    controller._apply_queued_event(
        NiceGUIQueuedEvent(
            kind="result",
            payload=interrupted.model_dump(mode="json"),
            turn_number=1,
            session_id=session_id,
        )
    )
    assert controller.active_record.transcript[-1].text == "Partial answer"

    controller.active_turn_state.busy = True
    controller.active_turn_state.cancelling = True
    controller._apply_queued_event(
        NiceGUIQueuedEvent(
            kind="complete",
            payload=None,
            turn_number=1,
            session_id=session_id,
        )
    )
    assert controller.active_record.transcript[-1].text == "Stopped the active turn."

    controller._apply_queued_event(
        NiceGUIQueuedEvent(
            kind="error",
            payload="All provider mode attempts failed.",
            turn_number=1,
            session_id=session_id,
        )
    )
    assert controller.active_record.transcript[-1].role == "error"

    with pytest.raises(ValueError, match="Unsupported queued event kind"):
        controller._apply_queued_event(
            NiceGUIQueuedEvent(
                kind="unknown",
                payload=None,
                turn_number=1,
                session_id=session_id,
            )
        )

    try:
        _serialize_workflow_event(object(), turn_number=1, session_id=session_id)
    except TypeError as exc:
        assert "Unsupported workflow event" in str(exc)


def test_controller_tool_turn_updates_state_and_inspector(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello from repo", encoding="utf-8")
    provider = _FakeProvider(
        [
            ParsedModelResponse(
                invocations=[
                    ToolInvocationRequest(
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                    )
                ]
            ),
            ParsedModelResponse(
                final_response={
                    "answer": "README says hello from repo.",
                    "citations": [],
                    "confidence": 0.9,
                    "uncertainty": [],
                    "missing_information": [],
                    "follow_up_suggestions": [],
                }
            ),
        ]
    )
    controller = _controller(tmp_path, provider)
    controller.active_record.runtime = controller.active_record.runtime.model_copy(
        update={
            "root_path": str(tmp_path),
            "enabled_tools": ["read_file"],
            "allow_filesystem": True,
        }
    )
    controller.save_active_session()

    assert controller.submit_prompt("read the README") is None
    _drain_until_idle(controller)

    loaded = controller.store.load_session(controller.active_session_id)
    assert loaded is not None
    assert loaded.transcript[-1].text == "README says hello from repo."
    assert (
        loaded.workflow_session_state.turns[-1].tool_results[0].tool_name == "read_file"
    )
    assert loaded.inspector_state.tool_executions
    assert loaded.workbench_items


def test_controller_approval_pauses_resumes_and_records_resolution(
    tmp_path: Path,
) -> None:
    (tmp_path / "README.md").write_text("approval path", encoding="utf-8")
    provider = _FakeProvider(
        [
            ParsedModelResponse(
                invocations=[
                    ToolInvocationRequest(
                        tool_name="read_file",
                        arguments={"path": "README.md"},
                    )
                ]
            ),
            ParsedModelResponse(
                final_response={
                    "answer": "Approved read finished.",
                    "citations": [],
                    "confidence": 0.9,
                    "uncertainty": [],
                    "missing_information": [],
                    "follow_up_suggestions": [],
                }
            ),
        ]
    )
    controller = _controller(tmp_path, provider)
    controller.active_record.runtime = NiceGUIRuntimeConfig(
        root_path=str(tmp_path),
        enabled_tools=["read_file"],
        allow_filesystem=True,
        require_approval_for={SideEffectClass.LOCAL_READ},
    )
    controller.save_active_session()

    assert controller.submit_prompt("read the README") is None
    deadline = time.monotonic() + 3.0
    while controller.active_turn_state.pending_approval is None:
        controller.drain_events()
        if time.monotonic() > deadline:
            raise AssertionError("approval was not requested")
        time.sleep(0.01)

    assert controller.active_turn_state.pending_approval.tool_name == "read_file"
    assert controller.resolve_approval(approved=True) is True
    _drain_until_idle(controller)

    texts = [entry.text for entry in controller.active_record.transcript]
    assert any("Approval required" in text for text in texts)
    assert any("approved" in text for text in texts)
    assert texts[-1] == "Approved read finished."


def test_controller_cancellation_records_interrupted_state(tmp_path: Path) -> None:
    provider = _FakeProvider(
        [
            ParsedModelResponse(
                invocations=[
                    ToolInvocationRequest(
                        tool_name="read_file",
                        arguments={"path": "missing.md"},
                    )
                ]
            ),
            ParsedModelResponse(
                final_response={
                    "answer": "This should not be reached.",
                    "citations": [],
                    "confidence": 0.1,
                    "uncertainty": [],
                    "missing_information": [],
                    "follow_up_suggestions": [],
                }
            ),
        ]
    )
    controller = _controller(tmp_path, provider)
    controller.active_record.runtime = NiceGUIRuntimeConfig(
        root_path=str(tmp_path),
        enabled_tools=["read_file"],
        allow_filesystem=True,
    )
    controller.save_active_session()

    assert controller.submit_prompt("try to read") is None
    assert controller.cancel_active_turn() is True
    _drain_until_idle(controller)

    assert any(
        entry.assistant_completion_state == "interrupted"
        or "Interrupted" in entry.text
        or "Stopped" in entry.text
        for entry in controller.active_record.transcript
    )
