from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from sqlalchemy import select, update

from llm_tools.apps.assistant_app.auth import (
    LocalAuthProvider,
    ensure_secret_file,
    validate_hosted_startup,
    verify_password,
)
from llm_tools.apps.assistant_app.models import (
    AssistantBranding,
    NiceGUIAdminSettings,
    NiceGUIInspectorEntry,
    NiceGUIPreferences,
    NiceGUIRuntimeConfig,
    NiceGUITranscriptEntry,
    NiceGUIWorkbenchItem,
)
from llm_tools.apps.assistant_app.paths import expanded_path_text
from llm_tools.apps.assistant_app.store import (
    NICEGUI_DB_ENV_VAR,
    SQLiteNiceGUIChatStore,
    chat_messages,
    chat_sessions,
    default_db_path,
    remember_default_db_path,
    workbench_items,
)
from llm_tools.harness_api.models import BudgetPolicy
from llm_tools.harness_api.replay import build_stored_artifacts
from llm_tools.harness_api.store import CURRENT_HARNESS_STATE_SCHEMA_VERSION
from llm_tools.harness_api.tasks import create_root_task
from llm_tools.workflow_api import (
    ChatContextSummary,
    ChatFinalResponse,
    ChatMessage,
    ChatSessionTurnRecord,
)


def _store(tmp_path: Path) -> SQLiteNiceGUIChatStore:
    store = SQLiteNiceGUIChatStore(
        tmp_path / "chat.sqlite3",
        db_key_file=tmp_path / "db.key",
        user_key_file=tmp_path / "user-kek.key",
    )
    store.initialize()
    return store


def test_default_db_path_prefers_env_then_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_tools.apps.assistant_app import store as store_module

    fallback_path = tmp_path / "fallback.sqlite3"
    pointer_path = tmp_path / "pointer.txt"
    selected_path = tmp_path / "selected" / "chat.sqlite3"
    env_path = tmp_path / "env.sqlite3"
    monkeypatch.setattr(store_module, "_DEFAULT_DB_PATH", fallback_path)
    monkeypatch.setattr(store_module, "_DB_POINTER_PATH", pointer_path)
    monkeypatch.delenv(NICEGUI_DB_ENV_VAR, raising=False)

    assert default_db_path() == fallback_path

    remember_default_db_path(selected_path)

    assert pointer_path.read_text(encoding="utf-8") == str(selected_path)
    assert default_db_path() == selected_path

    monkeypatch.setenv(NICEGUI_DB_ENV_VAR, str(env_path))

    assert default_db_path() == env_path


def test_assistant_store_paths_expand_user(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_tools.apps.assistant_app import store as store_module

    home = tmp_path / "home"
    pointer_path = tmp_path / "pointer.txt"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    monkeypatch.delenv("HOMEDRIVE", raising=False)
    monkeypatch.delenv("HOMEPATH", raising=False)
    monkeypatch.setattr(store_module, "_DB_POINTER_PATH", pointer_path)
    monkeypatch.delenv(NICEGUI_DB_ENV_VAR, raising=False)

    remember_default_db_path("~/chosen/chat.sqlite3")

    assert default_db_path() == home / "chosen" / "chat.sqlite3"
    assert expanded_path_text("~/chosen/chat.sqlite3") == str(
        home / "chosen" / "chat.sqlite3"
    )

    store = SQLiteNiceGUIChatStore(
        "~/state/chat.sqlite3",
        db_key_file="~/keys/db.key",
        user_key_file="~/keys/user-kek.key",
    )
    assert store.db_path == home / "state" / "chat.sqlite3"
    assert store.db_key_file == home / "keys" / "db.key"
    assert store.user_key_file == home / "keys" / "user-kek.key"

    startup = validate_hosted_startup(
        auth_mode="local",
        host="127.0.0.1",
        public_base_url=None,
        tls_certfile=None,
        tls_keyfile=None,
        allow_insecure_hosted_secrets=False,
        secret_key_path=Path("~/keys/app.secret"),
    )
    assert startup.config.secret_key_path == str(home / "keys" / "app.secret")


def test_store_initializes_and_round_trips_session(tmp_path: Path) -> None:
    store = _store(tmp_path)
    runtime = NiceGUIRuntimeConfig(selected_model="test-model", root_path=str(tmp_path))

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
            started_at="1",
            finished_at="2",
            duration_seconds=1.25,
            created_at="2",
            updated_at="2",
        )
    )
    record.workflow_session_state.context_summary = ChatContextSummary(
        content="Earlier context summary.",
        covered_turn_count=1,
        created_at="1",
        updated_at="2",
        compaction_count=1,
    )
    record.workflow_session_state.turns.append(
        ChatSessionTurnRecord(
            status="completed",
            new_messages=[
                ChatMessage(role="user", content="What is here?"),
                ChatMessage(role="assistant", content="A repo."),
            ],
            final_response=ChatFinalResponse(answer="A repo."),
        )
    )
    store.save_session(record)

    loaded = store.load_session(record.summary.session_id)

    assert loaded is not None
    assert loaded.summary.title == "Repo chat"
    assert loaded.summary.message_count == 2
    assert loaded.runtime.selected_model == "test-model"
    assert loaded.transcript[0].text == "What is here?"
    assert loaded.transcript[1].final_response is not None
    assert loaded.transcript[1].final_response.answer == "A repo."
    assert loaded.inspector_state.provider_messages[0].payload == {"count": 2}
    assert loaded.workbench_items[0].version == 2
    assert loaded.workbench_items[0].active is True
    assert loaded.workbench_items[0].duration_seconds == 1.25
    assert loaded.workflow_session_state.context_summary is not None
    assert loaded.workflow_session_state.context_summary.content == (
        "Earlier context summary."
    )


def test_harness_store_round_trips_encrypted_state(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.create_session(
        NiceGUIRuntimeConfig(selected_model="test-model"),
        title="Deep task chat",
        owner_user_id="user-a",
    )
    harness_store = store.harness_store(
        chat_session_id=record.summary.session_id,
        owner_user_id=record.summary.owner_user_id,
    )
    state = create_root_task(
        schema_version=CURRENT_HARNESS_STATE_SCHEMA_VERSION,
        session_id="harness-1",
        root_task_id="task-1",
        title="Secret task",
        intent="Investigate the secret repository",
        budget_policy=BudgetPolicy(max_turns=2),
        started_at="2026-01-01T00:00:00Z",
    )

    saved = harness_store.save_session(
        state, artifacts=build_stored_artifacts(state=state)
    )
    loaded = harness_store.load_session("harness-1")

    assert loaded is not None
    assert saved.revision == "1"
    assert loaded.state.tasks[0].title == "Secret task"
    assert loaded.artifacts.summary is not None
    other_user_store = store.harness_store(
        chat_session_id=record.summary.session_id,
        owner_user_id="user-b",
    )
    assert other_user_store.load_session("harness-1") is None
    assert other_user_store.list_sessions() == []
    with store.engine.begin() as connection:
        row = (
            connection.execute(
                select(workbench_items).where(workbench_items.c.item_id == "missing")
            )
            .mappings()
            .first()
        )
        harness_row = (
            connection.exec_driver_sql(
                "SELECT state_json, artifacts_json FROM harness_sessions"
            )
            .mappings()
            .first()
        )

    assert row is None
    assert harness_row is not None
    assert "Secret task" not in str(harness_row["state_json"])
    assert "Investigate the secret repository" not in str(harness_row["state_json"])

    store.delete_session(record.summary.session_id)
    assert harness_store.load_session("harness-1") is None


def test_sqlcipher_database_rejects_plain_sqlite_and_wrong_key(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "chat.sqlite3"
    store = _store(tmp_path)
    store.create_session(
        NiceGUIRuntimeConfig(selected_model="secret-model"), title="Alpha"
    )

    with pytest.raises(sqlite3.DatabaseError):
        sqlite3.connect(db_path).execute(
            "SELECT count(*) FROM chat_sessions"
        ).fetchone()

    wrong_key = tmp_path / "wrong-db.key"
    wrong_key.write_text("wrong-passphrase", encoding="utf-8")
    with pytest.raises(Exception, match="initialize|file is encrypted|not a database"):
        SQLiteNiceGUIChatStore(
            db_path,
            db_key_file=wrong_key,
            user_key_file=tmp_path / "user-kek.key",
        )


def test_existing_encrypted_database_requires_original_key_files(
    tmp_path: Path,
) -> None:
    db_path = tmp_path / "chat.sqlite3"
    db_key = tmp_path / "db.key"
    user_key = tmp_path / "user-kek.key"
    store = SQLiteNiceGUIChatStore(
        db_path,
        db_key_file=db_key,
        user_key_file=user_key,
    )
    store.initialize()
    auth = LocalAuthProvider(store)
    user = auth.create_user(username="admin", password="secret-" + "value")
    store.create_session(
        NiceGUIRuntimeConfig(selected_model="secret-model"),
        title="Alpha",
        owner_user_id=user.user_id,
    )
    db_key_text = db_key.read_text(encoding="utf-8")

    db_key.unlink()
    with pytest.raises(Exception, match="database without key file"):
        SQLiteNiceGUIChatStore(
            db_path,
            db_key_file=db_key,
            user_key_file=user_key,
        )

    db_key.write_text(db_key_text, encoding="utf-8")
    user_key.unlink()
    with pytest.raises(Exception, match="Required key file"):
        SQLiteNiceGUIChatStore(
            db_path,
            db_key_file=db_key,
            user_key_file=user_key,
        )


def test_user_owned_fields_are_encrypted_inside_open_database(tmp_path: Path) -> None:
    store = _store(tmp_path)
    record = store.create_session(
        NiceGUIRuntimeConfig(selected_model="visible-model", root_path="/secret/root"),
        title="Secret title",
        owner_user_id="user-a",
    )
    record.transcript.append(
        NiceGUITranscriptEntry(role="user", text="Secret message", created_at="1")
    )
    record.workbench_items.append(
        NiceGUIWorkbenchItem(
            item_id="item-secret",
            kind="inspector",
            title="Secret workbench",
            payload={"secret": "payload"},
            created_at="1",
            updated_at="1",
        )
    )
    store.save_session(record)

    with store.engine.begin() as connection:
        session_row = connection.execute(select(chat_sessions)).mappings().first()
        message_row = connection.execute(select(chat_messages)).mappings().first()
        workbench_row = connection.execute(select(workbench_items)).mappings().first()

    assert session_row is not None
    assert "Secret title" not in str(session_row["title"])
    assert "/secret/root" not in str(session_row["root_path"])
    assert "Secret message" not in str(message_row["text"])
    assert "Secret workbench" not in str(workbench_row["title"])
    assert "payload" not in str(workbench_row["payload_json"])
    assert session_row["provider_protocol"] == "openai_api"
    assert session_row["selected_model"] == "visible-model"


def test_per_user_encryption_detects_owner_swapping(tmp_path: Path) -> None:
    store = _store(tmp_path)
    auth = LocalAuthProvider(store)
    password = "secret-" + "value"
    user_a = auth.create_user(username="alice", password=password)
    user_b = auth.create_user(username="bob", password=password)
    record = store.create_session(
        NiceGUIRuntimeConfig(selected_model="alpha"),
        title="Alice chat",
        owner_user_id=user_a.user_id,
    )

    with store.engine.begin() as connection:
        connection.execute(
            update(chat_sessions)
            .where(chat_sessions.c.session_id == record.summary.session_id)
            .values(owner_user_id=user_b.user_id)
        )

    with pytest.raises(Exception, match="decrypt"):
        store.load_session(record.summary.session_id, owner_user_id=user_b.user_id)


def test_store_lists_searches_appends_and_deletes_sessions(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.create_session(
        NiceGUIRuntimeConfig(selected_model="alpha"), title="Alpha"
    )
    second = store.create_session(
        NiceGUIRuntimeConfig(selected_model="beta"), title="Beta"
    )

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


def test_store_filters_sessions_and_preferences_by_owner(tmp_path: Path) -> None:
    store = _store(tmp_path)
    alpha = store.create_session(
        NiceGUIRuntimeConfig(selected_model="alpha"),
        title="Alpha",
        owner_user_id="user-a",
    )
    store.create_session(
        NiceGUIRuntimeConfig(selected_model="beta"),
        title="Beta",
        owner_user_id="user-b",
    )
    store.save_preferences(
        NiceGUIPreferences(active_session_id=alpha.summary.session_id),
        owner_user_id="user-a",
    )

    assert [item.title for item in store.list_sessions(owner_user_id="user-a")] == [
        "Alpha"
    ]
    assert store.load_session(alpha.summary.session_id, owner_user_id="user-b") is None
    assert (
        store.load_preferences(owner_user_id="user-a").active_session_id
        == alpha.summary.session_id
    )
    assert store.load_preferences(owner_user_id="user-b").active_session_id is None


def test_local_auth_provider_lifecycle(tmp_path: Path) -> None:
    store = _store(tmp_path)
    auth = LocalAuthProvider(store)
    credential_value = "secret"
    wrong_credential_value = "wrong"
    admin = auth.create_user(username="admin", password=credential_value, role="admin")

    assert auth.authenticate(username="admin", password=wrong_credential_value) is None
    assert auth.authenticate(username="admin", password=credential_value) is not None
    changed_credential_value = "new-" + "value"
    auth.reset_password(user_id=admin.user_id, password=changed_credential_value)
    assert auth.authenticate(username="admin", password=credential_value) is None
    assert (
        auth.authenticate(username="admin", password=changed_credential_value)
        is not None
    )
    session_id, token = auth.create_session(admin.user_id)
    assert auth.user_for_session(session_id, token).user_id == admin.user_id
    auth.revoke_session(session_id)
    assert auth.user_for_session(session_id, token) is None


def test_hosted_auth_validation_and_session_edges(tmp_path: Path) -> None:
    secret_path = tmp_path / "hosted" / "app.secret"
    first_secret = ensure_secret_file(secret_path, token_bytes=8)
    assert first_secret
    assert ensure_secret_file(secret_path, token_bytes=8) == first_secret
    assert verify_password("not-an-argon2-hash", "irrelevant") is False

    with pytest.raises(ValueError, match="auth_mode"):
        validate_hosted_startup(
            auth_mode="bad",
            host="127.0.0.1",
            public_base_url=None,
            tls_certfile=None,
            tls_keyfile=None,
            allow_insecure_hosted_secrets=False,
        )
    with pytest.raises(ValueError, match="provided together"):
        validate_hosted_startup(
            auth_mode="local",
            host="127.0.0.1",
            public_base_url=None,
            tls_certfile="cert.pem",
            tls_keyfile=None,
            allow_insecure_hosted_secrets=False,
        )
    tls = validate_hosted_startup(
        auth_mode="local",
        host="192.0.2.1",
        public_base_url="https://chat.example.test",
        tls_certfile=None,
        tls_keyfile=None,
        allow_insecure_hosted_secrets=False,
    )
    assert tls.tls_enabled is True
    assert tls.config.secret_entry_enabled is True
    insecure_override = validate_hosted_startup(
        auth_mode="local",
        host="192.0.2.1",
        public_base_url="http://chat.example.test",
        tls_certfile=None,
        tls_keyfile=None,
        allow_insecure_hosted_secrets=True,
    )
    assert insecure_override.config.secret_entry_enabled is True

    store = _store(tmp_path)
    auth = LocalAuthProvider(store, session_days=-1)
    auth_password = "auth-" + "value"
    with pytest.raises(ValueError, match="username"):
        auth.create_user(username=" ", password=auth_password)
    with pytest.raises(ValueError, match="password"):
        auth.create_user(username="user", password="")
    with pytest.raises(ValueError, match="role"):
        auth.create_user(username="user", password=auth_password, role="owner")
    user = auth.create_user(username="user", password=auth_password)
    session_id, token = auth.create_session(user.user_id)
    assert auth.user_for_session(session_id, token) is None

    active_auth = LocalAuthProvider(store)
    active_session_id, active_token = active_auth.create_session(user.user_id)
    assert active_auth.user_for_session(active_session_id, "wrong") is None
    store.set_user_disabled(user.user_id, True)
    assert active_auth.authenticate(username="user", password=auth_password) is None
    assert active_auth.user_for_session(active_session_id, active_token) is None
    with pytest.raises(ValueError, match="password"):
        active_auth.reset_password(user_id=user.user_id, password="")


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


def test_admin_settings_round_trip(tmp_path: Path) -> None:
    store = _store(tmp_path)

    assert store.load_admin_settings().deep_task_mode_enabled is False
    assert store.load_admin_settings().skills_enabled is False
    assert store.load_admin_settings().branding.app_name == "LLM Tools Assistant"

    store.save_admin_settings(
        NiceGUIAdminSettings(
            deep_task_mode_enabled=True,
            skills_enabled=True,
            ollama_native_provider_enabled=True,
            ask_sage_native_provider_enabled=True,
            branding=AssistantBranding(
                app_name="Custom Assistant",
                short_name="Custom",
                icon_name="auto_awesome",
                favicon_svg='<svg viewBox="0 0 1 1"></svg>',
            ),
        )
    )

    loaded = store.load_admin_settings()
    assert loaded.deep_task_mode_enabled is True
    assert loaded.skills_enabled is True
    assert loaded.ollama_native_provider_enabled is True
    assert loaded.ask_sage_native_provider_enabled is True
    assert loaded.branding.app_name == "Custom Assistant"
    assert loaded.branding.short_name == "Custom"
    assert loaded.branding.icon_name == "auto_awesome"


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
