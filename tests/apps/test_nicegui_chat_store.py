from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import update

from llm_tools.apps.nicegui_chat.models import (
    NiceGUIInspectorEntry,
    NiceGUIPreferences,
    NiceGUIRuntimeConfig,
    NiceGUITranscriptEntry,
    NiceGUIWorkbenchItem,
)
from llm_tools.apps.nicegui_chat.store import (
    SQLiteNiceGUIChatStore,
    chat_sessions,
)
from llm_tools.workflow_api import ChatFinalResponse


def _store(tmp_path: Path) -> SQLiteNiceGUIChatStore:
    store = SQLiteNiceGUIChatStore(tmp_path / "chat.sqlite3")
    store.initialize()
    return store


def test_store_initializes_and_round_trips_session(tmp_path: Path) -> None:
    store = _store(tmp_path)
    runtime = NiceGUIRuntimeConfig(model_name="test-model", root_path=str(tmp_path))

    record = store.create_session(runtime, title="Repo chat")
    record.transcript.append(
        NiceGUITranscriptEntry(role="user", text="What is here?", created_at="1")
    )
    record.transcript.append(
        NiceGUITranscriptEntry(
            role="assistant",
            text="A repo.",
            final_response=ChatFinalResponse(answer="A repo.", confidence=0.8),
            created_at="2",
        )
    )
    record.inspector_state.provider_messages.append(
        NiceGUIInspectorEntry(label="messages", payload={"count": 2}, created_at="2")
    )
    record.workbench_items.append(
        NiceGUIWorkbenchItem(
            item_id="item-1",
            kind="inspector",
            title="Messages",
            payload={"count": 2},
            version=2,
            active=True,
            created_at="2",
            updated_at="2",
        )
    )
    store.save_session(record)

    loaded = store.load_session(record.summary.session_id)

    assert loaded is not None
    assert loaded.summary.title == "Repo chat"
    assert loaded.summary.message_count == 2
    assert loaded.runtime.model_name == "test-model"
    assert loaded.transcript[0].text == "What is here?"
    assert loaded.transcript[1].final_response is not None
    assert loaded.transcript[1].final_response.answer == "A repo."
    assert loaded.inspector_state.provider_messages[0].payload == {"count": 2}
    assert loaded.workbench_items[0].version == 2
    assert loaded.workbench_items[0].active is True


def test_store_lists_searches_appends_and_deletes_sessions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.create_session(
        NiceGUIRuntimeConfig(model_name="alpha"), title="Alpha"
    )
    second = store.create_session(NiceGUIRuntimeConfig(model_name="beta"), title="Beta")

    store.append_message(
        first.summary.session_id,
        NiceGUITranscriptEntry(role="user", text="hello"),
    )

    all_summaries = store.list_sessions()
    beta_matches = store.list_sessions(query="beta")

    assert [summary.title for summary in all_summaries] == ["Alpha", "Beta"]
    assert [summary.session_id for summary in beta_matches] == [
        second.summary.session_id
    ]
    assert store.load_session(first.summary.session_id).transcript[0].text == "hello"  # type: ignore[union-attr]

    store.delete_session(first.summary.session_id)

    assert store.load_session(first.summary.session_id) is None
    assert [summary.session_id for summary in store.list_sessions()] == [
        second.summary.session_id
    ]


def test_preferences_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)
    preferences = NiceGUIPreferences(
        active_session_id="chat-1",
        sidebar_collapsed=True,
        workbench_open=False,
        recent_roots=["/repo"],
        recent_models={"ollama": ["model-a"]},
    )

    store.save_preferences(preferences)

    loaded = store.load_preferences()
    assert loaded.active_session_id == "chat-1"
    assert loaded.sidebar_collapsed is True
    assert loaded.workbench_open is False
    assert loaded.recent_models == {"ollama": ["model-a"]}


def test_temporary_sessions_are_not_persisted(tmp_path: Path) -> None:
    store = _store(tmp_path)

    record = store.create_session(NiceGUIRuntimeConfig(), temporary=True)

    assert record.summary.temporary is True
    assert store.list_sessions() == []
    assert store.load_session(record.summary.session_id) is None


def test_corrupt_json_fails_clearly(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.create_session(NiceGUIRuntimeConfig())

    with store.engine.begin() as connection:
        connection.execute(
            update(chat_sessions)
            .where(chat_sessions.c.session_id == record.summary.session_id)
            .values(runtime_json="{bad json")
        )

    with pytest.raises(Exception, match="runtime_json"):
        store.load_session(record.summary.session_id)
