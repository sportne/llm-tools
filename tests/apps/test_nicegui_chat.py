from __future__ import annotations

import runpy
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from llm_tools.apps.assistant_config import StreamlitAssistantConfig
from llm_tools.apps.chat_config import ProviderPreset
from llm_tools.apps.nicegui_chat.app import (
    _composer_action_icon,
    _discover_model_names,
    _event_payload_text,
    _extract_model_names_from_models_payload,
    _first_nonempty_text,
    _models_endpoint_url,
    _runtime_summary_parts,
    _runtime_summary_text,
    _sidebar_container_classes,
    _workbench_container_classes,
    build_nicegui_chat_ui,
    build_parser,
    main,
    resolve_assistant_config,
    resolve_root_argument,
)
from llm_tools.apps.nicegui_chat.controller import (
    NiceGUIChatController,
    NiceGUIQueuedEvent,
    _serialize_workflow_event,
)
from llm_tools.apps.nicegui_chat.models import (
    NiceGUIPreferences,
    NiceGUIRuntimeConfig,
)
from llm_tools.apps.nicegui_chat.store import SQLiteNiceGUIChatStore
from llm_tools.llm_adapters import ActionEnvelopeAdapter, ParsedModelResponse
from llm_tools.tool_api import SideEffectClass, ToolInvocationRequest
from llm_tools.workflow_api import (
    ChatMessage,
    ChatWorkflowInspectorEvent,
    ChatWorkflowResultEvent,
    ChatWorkflowStatusEvent,
    ChatWorkflowTurnResult,
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


def _controller(
    tmp_path: Path,
    provider: _FakeProvider,
) -> NiceGUIChatController:
    store = SQLiteNiceGUIChatStore(tmp_path / "chat.sqlite3")
    store.initialize()
    return NiceGUIChatController(
        store=store,
        config=StreamlitAssistantConfig(),
        root_path=tmp_path,
        provider_factory=lambda _runtime: provider,
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
    import llm_tools.apps.nicegui_chat as nicegui_chat

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

    assert nicegui_chat.main is not None
    assert config.llm.model_name == "qwen3:8b"
    assert config.llm.provider_mode_strategy.value == "json"
    assert args.db_path == Path("chat.sqlite3")


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
    import llm_tools.apps.nicegui_chat.app as app_module

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

    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr(app_module, "urlopen", fake_urlopen)

    assert _discover_model_names(
        provider=ProviderPreset.OPENAI,
        base_url="https://example.test/v1",
        timeout=1.5,
    ) == ["model-a", "model-b"]
    assert calls == [
        ("https://example.test/v1/models", "Bearer test-key", 1.5),
    ]


def test_model_discovery_rejects_invalid_or_failed_endpoints(
    monkeypatch: Any,
) -> None:
    import llm_tools.apps.nicegui_chat.app as app_module

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
    assert config.llm.temperature == 0.2
    assert config.session.max_context_tokens == 1000
    assert config.tool_limits.max_tool_result_chars == 900
    assert root == (tmp_path / "override").resolve()


def test_root_resolution_uses_config_default_and_none() -> None:
    parser = build_parser()
    no_root_args = parser.parse_args([])
    config_with_root = StreamlitAssistantConfig.model_validate(
        {"workspace": {"default_root": "."}}
    )

    assert resolve_root_argument(no_root_args, StreamlitAssistantConfig()) is None
    assert resolve_root_argument(no_root_args, config_with_root) == Path(".").resolve()


def test_main_resolves_arguments_and_delegates_run(
    monkeypatch: Any, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_run(**kwargs: object) -> None:
        calls.append(kwargs)

    monkeypatch.setattr(
        "llm_tools.apps.nicegui_chat.app.run_nicegui_chat_app", fake_run
    )

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

    monkeypatch.setattr(
        "llm_tools.apps.nicegui_chat.app.run_nicegui_chat_app", fake_run
    )

    assert main(["--no-browser"]) == 0
    assert calls[0]["root_path"] is None


def test_module_entrypoint_uses_package_main(monkeypatch: Any) -> None:
    import llm_tools.apps.nicegui_chat as nicegui_chat

    calls = []
    monkeypatch.setattr(nicegui_chat, "main", lambda: calls.append("called"))

    runpy.run_module("llm_tools.apps.nicegui_chat.__main__", run_name="__main__")

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

    build_nicegui_chat_ui(controller)

    assert controller.active_record.summary.title == "New chat"


def test_app_builder_applies_initial_layout_preferences(tmp_path: Path) -> None:
    from nicegui import ui

    controller = _controller(tmp_path, _FakeProvider([]))
    controller.preferences.sidebar_collapsed = True
    controller.preferences.workbench_open = False

    build_nicegui_chat_ui(controller)

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


def test_controller_loads_existing_session_and_deletes_active_session(
    tmp_path: Path,
) -> None:
    store = SQLiteNiceGUIChatStore(tmp_path / "chat.sqlite3")
    store.initialize()
    stored = store.create_session(NiceGUIRuntimeConfig(), title="Stored chat")

    controller = NiceGUIChatController(
        store=store,
        config=StreamlitAssistantConfig(),
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
    store = SQLiteNiceGUIChatStore(tmp_path / "chat.sqlite3")
    store.initialize()
    controller = NiceGUIChatController(
        store=store,
        config=StreamlitAssistantConfig(),
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

    for kind in ("provider_messages", "parsed_response"):
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
            )
        )
    assert controller.active_record.inspector_state.provider_messages
    assert controller.active_record.inspector_state.parsed_responses

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
