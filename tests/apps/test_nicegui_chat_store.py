from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import update

from llm_tools.apps.nicegui_chat.auth import (
    LocalAuthProvider,
    ensure_secret_file,
    validate_hosted_startup,
    verify_password,
)
from llm_tools.apps.nicegui_chat.models import (
    NiceGUIInspectorEntry,
    NiceGUIPreferences,
    NiceGUIRuntimeConfig,
    NiceGUITranscriptEntry,
    NiceGUIWorkbenchItem,
)
from llm_tools.apps.nicegui_chat.secrets import (
    NiceGUISecretStoreError,
    SQLiteSecretStore,
    ensure_master_key,
)
from llm_tools.apps.nicegui_chat.store import (
    NICEGUI_DB_ENV_VAR,
    SQLiteNiceGUIChatStore,
    chat_sessions,
    default_db_path,
    remember_default_db_path,
)
from llm_tools.workflow_api import ChatFinalResponse


def _store(tmp_path: Path) -> SQLiteNiceGUIChatStore:
    store = SQLiteNiceGUIChatStore(tmp_path / "chat.sqlite3")
    store.initialize()
    return store


def test_default_db_path_prefers_env_then_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llm_tools.apps.nicegui_chat import store as store_module

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
            started_at="1",
            finished_at="2",
            duration_seconds=1.25,
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
    assert loaded.workbench_items[0].duration_seconds == 1.25


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


def test_store_filters_sessions_and_preferences_by_owner(tmp_path: Path) -> None:
    store = _store(tmp_path)
    alpha = store.create_session(
        NiceGUIRuntimeConfig(model_name="alpha"),
        title="Alpha",
        owner_user_id="user-a",
    )
    store.create_session(
        NiceGUIRuntimeConfig(model_name="beta"),
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


def test_local_auth_and_encrypted_secret_store(tmp_path: Path) -> None:
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

    secret_store = SQLiteSecretStore(store, master_key_path=tmp_path / "master.key")
    secret_store.set_secret(
        owner_user_id=admin.user_id,
        session_id="chat-1",
        name="OPENAI_API_KEY",
        value="plain-secret",
    )
    record = store.get_secret_record(
        owner_user_id=admin.user_id,
        session_id="chat-1",
        name="OPENAI_API_KEY",
    )
    assert record is not None
    assert "plain-secret" not in record.ciphertext
    assert (
        secret_store.get_secret(
            owner_user_id=admin.user_id,
            session_id="chat-1",
            name="OPENAI_API_KEY",
        )
        == "plain-secret"
    )


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


def test_encrypted_secret_store_edges(tmp_path: Path) -> None:
    key_path = tmp_path / "master.key"
    key = ensure_master_key(key_path)
    assert key == ensure_master_key(key_path)
    bad_key_path = tmp_path / "bad.key"
    bad_key_path.write_text("not-a-fernet-key", encoding="utf-8")
    with pytest.raises(NiceGUISecretStoreError, match="Invalid"):
        ensure_master_key(bad_key_path)

    store = _store(tmp_path)
    auth = LocalAuthProvider(store)
    owner_password = "owner-" + "value"
    user = auth.create_user(username="owner", password=owner_password)
    session = store.create_session(
        NiceGUIRuntimeConfig(), owner_user_id=user.user_id
    ).summary.session_id
    secret_store = SQLiteSecretStore(store, master_key_path=key_path)
    with pytest.raises(ValueError, match="secret name"):
        secret_store.set_secret(
            owner_user_id="user-1",
            session_id="chat-1",
            name=" ",
            value="value",
        )
    secret_store.set_secret(
        owner_user_id="user-1",
        session_id="chat-1",
        name="EMPTY",
        value=" ",
    )
    assert (
        secret_store.has_secret(
            owner_user_id="user-1", session_id="chat-1", name="EMPTY"
        )
        is False
    )
    secret_store.set_secret(
        owner_user_id=user.user_id,
        session_id=session,
        name="__provider_api_key__",
        value="provider",
    )
    secret_store.set_secret(
        owner_user_id=user.user_id,
        session_id=session,
        name="GITLAB_API_TOKEN",
        value="tool-token",
    )
    assert secret_store.secret_names(
        owner_user_id=user.user_id, session_id=session
    ) == [
        "GITLAB_API_TOKEN",
        "__provider_api_key__",
    ]
    assert secret_store.tool_env(owner_user_id=user.user_id, session_id=session) == {
        "GITLAB_API_TOKEN": "tool-token"
    }
    wrong_key_store = SQLiteSecretStore(store, master_key_path=tmp_path / "other.key")
    with pytest.raises(NiceGUISecretStoreError, match="Could not decrypt"):
        wrong_key_store.get_secret(
            owner_user_id=user.user_id,
            session_id=session,
            name="GITLAB_API_TOKEN",
        )
    secret_store.delete_secret(
        owner_user_id=user.user_id, session_id=session, name="GITLAB_API_TOKEN"
    )
    assert (
        secret_store.get_secret(
            owner_user_id=user.user_id, session_id=session, name="GITLAB_API_TOKEN"
        )
        is None
    )


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
