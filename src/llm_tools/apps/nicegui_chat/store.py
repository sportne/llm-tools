"""SQLite persistence for the NiceGUI chat app."""

from __future__ import annotations

import json
import os
from contextlib import suppress
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError
from sqlalchemy import (
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    UniqueConstraint,
    create_engine,
    delete,
    event,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError

from llm_tools.apps.nicegui_chat.models import (
    NiceGUIInspectorState,
    NiceGUIPreferences,
    NiceGUIRuntimeConfig,
    NiceGUISecretRecord,
    NiceGUISessionRecord,
    NiceGUISessionSummary,
    NiceGUITranscriptEntry,
    NiceGUIUser,
    NiceGUIUserRole,
    NiceGUIUserSession,
    NiceGUIWorkbenchItem,
)
from llm_tools.workflow_api import ChatFinalResponse, ChatSessionState, ChatTokenUsage

NICEGUI_DB_ENV_VAR = "LLM_TOOLS_NICEGUI_DB"
PREFERENCES_KEY = "preferences"
_DEFAULT_DB_PATH = Path.home() / ".llm-tools" / "assistant" / "nicegui" / "chat.sqlite3"
_DB_POINTER_PATH = Path.home() / ".llm-tools" / "assistant" / "nicegui" / "db_path.txt"

metadata = MetaData()

chat_sessions = Table(
    "chat_sessions",
    metadata,
    Column("session_id", String, primary_key=True),
    Column("title", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    Column("root_path", String, nullable=True),
    Column("provider", String, nullable=False),
    Column("model_name", String, nullable=False),
    Column("runtime_json", Text, nullable=False),
    Column("workflow_session_state_json", Text, nullable=False),
    Column("token_usage_json", Text, nullable=True),
    Column("inspector_state_json", Text, nullable=False),
    Column("confidence", Float, nullable=True),
    Column("temporary", Boolean, nullable=False, default=False),
    Column("project_id", String, nullable=True),
    Column("owner_user_id", String, nullable=True),
)

chat_messages = Table(
    "chat_messages",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column(
        "session_id",
        String,
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("ordinal", Integer, nullable=False),
    Column("role", String, nullable=False),
    Column("text", Text, nullable=False),
    Column("final_response_json", Text, nullable=True),
    Column("assistant_completion_state", String, nullable=False),
    Column("show_in_transcript", Boolean, nullable=False, default=True),
    Column("created_at", String, nullable=False),
    UniqueConstraint("session_id", "ordinal", name="uq_chat_messages_session_ordinal"),
)

workbench_items = Table(
    "workbench_items",
    metadata,
    Column("item_id", String, primary_key=True),
    Column(
        "session_id",
        String,
        ForeignKey("chat_sessions.session_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("kind", String, nullable=False),
    Column("title", String, nullable=False),
    Column("payload_json", Text, nullable=False),
    Column("version", Integer, nullable=False),
    Column("active", Boolean, nullable=False, default=False),
    Column("started_at", String, nullable=True),
    Column("finished_at", String, nullable=True),
    Column("duration_seconds", Float, nullable=True),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
)

app_preferences = Table(
    "app_preferences",
    metadata,
    Column("key", String, primary_key=True),
    Column("payload_json", Text, nullable=False),
)

users = Table(
    "users",
    metadata,
    Column("user_id", String, primary_key=True),
    Column("username", String, nullable=False, unique=True),
    Column("password_hash", Text, nullable=False),
    Column("role", String, nullable=False),
    Column("disabled", Boolean, nullable=False, default=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    Column("last_login_at", String, nullable=True),
)

user_sessions = Table(
    "user_sessions",
    metadata,
    Column("session_id", String, primary_key=True),
    Column(
        "user_id",
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("token_hash", String, nullable=False),
    Column("created_at", String, nullable=False),
    Column("expires_at", String, nullable=False),
    Column("revoked_at", String, nullable=True),
)

secret_records = Table(
    "secret_records",
    metadata,
    Column("secret_id", String, primary_key=True),
    Column(
        "owner_user_id",
        String,
        ForeignKey("users.user_id", ondelete="CASCADE"),
        nullable=False,
    ),
    Column("session_id", String, nullable=False),
    Column("name", String, nullable=False),
    Column("ciphertext", Text, nullable=False),
    Column("created_at", String, nullable=False),
    Column("updated_at", String, nullable=False),
    UniqueConstraint(
        "owner_user_id",
        "session_id",
        "name",
        name="uq_secret_records_owner_session_name",
    ),
)

auth_events = Table(
    "auth_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("user_id", String, nullable=True),
    Column("event_type", String, nullable=False),
    Column("detail_json", Text, nullable=False),
    Column("created_at", String, nullable=False),
)


class NiceGUIChatStoreError(RuntimeError):
    """Base error raised by the NiceGUI SQLite store."""


class NiceGUIChatStoreCorruptionError(NiceGUIChatStoreError):
    """Raised when persisted JSON no longer validates."""


def default_db_path() -> Path:
    """Return the configured NiceGUI SQLite database path."""
    configured = os.getenv(NICEGUI_DB_ENV_VAR)
    if configured and configured.strip():
        return Path(configured).expanduser()
    with suppress(OSError):
        configured_path = _DB_POINTER_PATH.read_text(encoding="utf-8").strip()
        if configured_path:
            return Path(configured_path).expanduser()
    return _DEFAULT_DB_PATH


def remember_default_db_path(path: Path | str) -> None:
    """Persist the preferred NiceGUI SQLite database path for future launches."""
    db_path = Path(path).expanduser()
    _DB_POINTER_PATH.parent.mkdir(parents=True, exist_ok=True)
    _DB_POINTER_PATH.write_text(str(db_path), encoding="utf-8")


def create_sqlite_engine(path: Path | str) -> Engine:
    """Create a SQLite engine with app defaults."""
    path_text = str(path)
    url = "sqlite:///:memory:" if path_text == ":memory:" else f"sqlite:///{path_text}"
    if path_text != ":memory:":
        Path(path_text).expanduser().parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(
        url,
        connect_args={"check_same_thread": False},
        future=True,
    )

    @event.listens_for(engine, "connect")
    def _configure_sqlite(dbapi_connection: Any, _connection_record: object) -> None:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        if path_text != ":memory:":
            with suppress(Exception):
                cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    return engine


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _preferences_key(owner_user_id: str | None = None) -> str:
    if owner_user_id is None:
        return PREFERENCES_KEY
    return f"{PREFERENCES_KEY}:{owner_user_id}"


def _dump_model(model: BaseModel) -> str:
    return model.model_dump_json()


def _dump_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def _load_model_json(
    *,
    raw: str | None,
    model_type: type[BaseModel],
    field_name: str,
) -> BaseModel:
    try:
        if raw is None:
            raise ValueError("missing JSON payload")
        return model_type.model_validate_json(raw)
    except (PydanticValidationError, ValueError) as exc:
        raise NiceGUIChatStoreCorruptionError(
            f"Invalid persisted JSON for {field_name}: {exc}"
        ) from exc


def _load_payload_json(*, raw: str | None, field_name: str) -> object:
    try:
        return json.loads(raw or "null")
    except json.JSONDecodeError as exc:
        raise NiceGUIChatStoreCorruptionError(
            f"Invalid persisted JSON for {field_name}: {exc}"
        ) from exc


def _sync_summary_fields(record: NiceGUISessionRecord) -> None:
    record.summary.root_path = record.runtime.root_path
    record.summary.provider = record.runtime.provider
    record.summary.model_name = record.runtime.model_name
    record.summary.message_count = len(
        [entry for entry in record.transcript if entry.show_in_transcript]
    )
    if record.transcript:
        latest = record.transcript[-1].created_at or record.summary.updated_at
        record.summary.updated_at = latest


def _new_session_record(
    *,
    runtime_config: NiceGUIRuntimeConfig,
    title: str | None,
    temporary: bool,
    owner_user_id: str | None = None,
) -> NiceGUISessionRecord:
    timestamp = _now_iso()
    session_id = f"chat-{uuid4().hex}"
    runtime = runtime_config.model_copy(deep=True)
    summary = NiceGUISessionSummary(
        session_id=session_id,
        title=title or ("Temporary chat" if temporary else "New chat"),
        created_at=timestamp,
        updated_at=timestamp,
        root_path=runtime.root_path,
        provider=runtime.provider,
        model_name=runtime.model_name,
        temporary=temporary,
        owner_user_id=owner_user_id,
    )
    return NiceGUISessionRecord(summary=summary, runtime=runtime)


class SQLiteNiceGUIChatStore:
    """SQLAlchemy-backed store for NiceGUI chat sessions."""

    def __init__(self, db_path: Path | str | None = None) -> None:
        self.db_path = (
            Path(db_path).expanduser() if db_path is not None else default_db_path()
        )
        self.engine = create_sqlite_engine(self.db_path)

    def initialize(self) -> None:
        """Create missing tables."""
        try:
            metadata.create_all(self.engine)
            self._ensure_schema_columns()
        except SQLAlchemyError as exc:  # pragma: no cover - defensive wrapper
            raise NiceGUIChatStoreError(
                f"Failed to initialize SQLite store: {exc}"
            ) from exc

    def _ensure_schema_columns(self) -> None:
        """Add additive columns for existing SQLite databases."""
        workbench_columns = {
            "started_at": "TEXT",
            "finished_at": "TEXT",
            "duration_seconds": "FLOAT",
        }
        chat_session_columns = {"owner_user_id": "TEXT"}
        with self.engine.begin() as connection:
            self._ensure_table_columns(
                connection,
                table_name="workbench_items",
                required_columns=workbench_columns,
            )
            self._ensure_table_columns(
                connection,
                table_name="chat_sessions",
                required_columns=chat_session_columns,
            )

    def _ensure_table_columns(
        self, connection: Any, *, table_name: str, required_columns: dict[str, str]
    ) -> None:
        existing_columns = {
            str(row[1])
            for row in connection.exec_driver_sql(f"PRAGMA table_info({table_name})")
        }
        for column_name, column_type in required_columns.items():
            if column_name not in existing_columns:
                connection.exec_driver_sql(
                    f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}"
                )

    def list_sessions(
        self,
        *,
        limit: int | None = None,
        query: str | None = None,
        owner_user_id: str | None = None,
    ) -> list[NiceGUISessionSummary]:
        """Return durable session summaries newest first."""
        stmt = select(chat_sessions).where(chat_sessions.c.temporary.is_(False))
        if owner_user_id is not None:
            stmt = stmt.where(chat_sessions.c.owner_user_id == owner_user_id)
        cleaned_query = (query or "").strip()
        if cleaned_query:
            like_query = f"%{cleaned_query}%"
            stmt = stmt.where(
                chat_sessions.c.title.like(like_query)
                | chat_sessions.c.model_name.like(like_query)
                | chat_sessions.c.root_path.like(like_query)
            )
        stmt = stmt.order_by(chat_sessions.c.updated_at.desc())
        if limit is not None:
            stmt = stmt.limit(limit)
        with self.engine.begin() as connection:
            rows = connection.execute(stmt).mappings().all()
        return [
            NiceGUISessionSummary(
                session_id=str(row["session_id"]),
                title=str(row["title"]),
                created_at=str(row["created_at"]),
                updated_at=str(row["updated_at"]),
                root_path=row["root_path"],
                provider=row["provider"],
                model_name=str(row["model_name"]),
                message_count=self._message_count(str(row["session_id"])),
                temporary=bool(row["temporary"]),
                project_id=row["project_id"],
                owner_user_id=row["owner_user_id"],
            )
            for row in rows
        ]

    def create_session(
        self,
        runtime_config: NiceGUIRuntimeConfig,
        *,
        title: str | None = None,
        temporary: bool = False,
        owner_user_id: str | None = None,
    ) -> NiceGUISessionRecord:
        """Create and optionally persist a new session record."""
        record = _new_session_record(
            runtime_config=runtime_config,
            title=title,
            temporary=temporary,
            owner_user_id=owner_user_id,
        )
        if not temporary:
            self.save_session(record)
        return record

    def load_session(
        self, session_id: str, *, owner_user_id: str | None = None
    ) -> NiceGUISessionRecord | None:
        """Load a durable session by id."""
        with self.engine.begin() as connection:
            session_stmt = select(chat_sessions).where(
                chat_sessions.c.session_id == session_id
            )
            if owner_user_id is not None:
                session_stmt = session_stmt.where(
                    chat_sessions.c.owner_user_id == owner_user_id
                )
            session_row = connection.execute(session_stmt).mappings().first()
            if session_row is None:
                return None
            message_rows = (
                connection.execute(
                    select(chat_messages)
                    .where(chat_messages.c.session_id == session_id)
                    .order_by(chat_messages.c.ordinal.asc())
                )
                .mappings()
                .all()
            )
            workbench_rows = (
                connection.execute(
                    select(workbench_items)
                    .where(workbench_items.c.session_id == session_id)
                    .order_by(workbench_items.c.updated_at.asc())
                )
                .mappings()
                .all()
            )
        runtime = _load_model_json(
            raw=session_row["runtime_json"],
            model_type=NiceGUIRuntimeConfig,
            field_name="runtime_json",
        )
        workflow_state = _load_model_json(
            raw=session_row["workflow_session_state_json"],
            model_type=ChatSessionState,
            field_name="workflow_session_state_json",
        )
        token_usage = (
            None
            if session_row["token_usage_json"] is None
            else _load_model_json(
                raw=session_row["token_usage_json"],
                model_type=ChatTokenUsage,
                field_name="token_usage_json",
            )
        )
        inspector_state = _load_model_json(
            raw=session_row["inspector_state_json"],
            model_type=NiceGUIInspectorState,
            field_name="inspector_state_json",
        )
        transcript = [self._transcript_entry_from_row(row) for row in message_rows]
        record = NiceGUISessionRecord(
            summary=NiceGUISessionSummary(
                session_id=str(session_row["session_id"]),
                title=str(session_row["title"]),
                created_at=str(session_row["created_at"]),
                updated_at=str(session_row["updated_at"]),
                root_path=session_row["root_path"],
                provider=session_row["provider"],
                model_name=str(session_row["model_name"]),
                message_count=len(
                    [entry for entry in transcript if entry.show_in_transcript]
                ),
                temporary=bool(session_row["temporary"]),
                project_id=session_row["project_id"],
                owner_user_id=session_row["owner_user_id"],
            ),
            runtime=runtime,  # type: ignore[arg-type]
            transcript=transcript,
            workflow_session_state=workflow_state,  # type: ignore[arg-type]
            token_usage=token_usage,  # type: ignore[arg-type]
            inspector_state=inspector_state,  # type: ignore[arg-type]
            workbench_items=[
                NiceGUIWorkbenchItem(
                    item_id=str(row["item_id"]),
                    kind=row["kind"],
                    title=str(row["title"]),
                    payload=_load_payload_json(
                        raw=row["payload_json"], field_name="workbench.payload_json"
                    ),
                    version=int(row["version"]),
                    active=bool(row["active"]),
                    started_at=row["started_at"],
                    finished_at=row["finished_at"],
                    duration_seconds=(
                        None
                        if row["duration_seconds"] is None
                        else float(row["duration_seconds"])
                    ),
                    created_at=str(row["created_at"]),
                    updated_at=str(row["updated_at"]),
                )
                for row in workbench_rows
            ],
            confidence=session_row["confidence"],
        )
        _sync_summary_fields(record)
        return record

    def save_session(self, record: NiceGUISessionRecord) -> None:
        """Persist a full session record."""
        if record.summary.temporary:
            return
        _sync_summary_fields(record)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(chat_sessions.c.session_id).where(
                    chat_sessions.c.session_id == record.summary.session_id
                )
            ).first()
            values = self._session_values(record)
            if existing is None:
                connection.execute(insert(chat_sessions).values(values))
            else:
                connection.execute(
                    update(chat_sessions)
                    .where(chat_sessions.c.session_id == record.summary.session_id)
                    .values(values)
                )
            connection.execute(
                delete(chat_messages).where(
                    chat_messages.c.session_id == record.summary.session_id
                )
            )
            for ordinal, entry in enumerate(record.transcript):
                connection.execute(
                    insert(chat_messages).values(
                        self._message_values(
                            record.summary.session_id,
                            ordinal=ordinal,
                            entry=entry,
                        )
                    )
                )
            connection.execute(
                delete(workbench_items).where(
                    workbench_items.c.session_id == record.summary.session_id
                )
            )
            for item in record.workbench_items:
                connection.execute(
                    insert(workbench_items).values(
                        self._workbench_values(record.summary.session_id, item)
                    )
                )

    def append_message(
        self, session_id: str, entry: NiceGUITranscriptEntry
    ) -> NiceGUITranscriptEntry:
        """Append one transcript entry to a durable session."""
        if entry.created_at is None:
            entry = entry.model_copy(update={"created_at": _now_iso()})
        with self.engine.begin() as connection:
            max_ordinal = connection.execute(
                select(func.max(chat_messages.c.ordinal)).where(
                    chat_messages.c.session_id == session_id
                )
            ).scalar_one_or_none()
            ordinal = 0 if max_ordinal is None else int(max_ordinal) + 1
            connection.execute(
                insert(chat_messages).values(
                    self._message_values(session_id, ordinal=ordinal, entry=entry)
                )
            )
            connection.execute(
                update(chat_sessions)
                .where(chat_sessions.c.session_id == session_id)
                .values(updated_at=entry.created_at)
            )
        return entry

    def delete_session(self, session_id: str) -> None:
        """Delete a durable session and its dependent rows."""
        with self.engine.begin() as connection:
            connection.execute(
                delete(chat_sessions).where(chat_sessions.c.session_id == session_id)
            )

    def load_preferences(
        self, *, owner_user_id: str | None = None
    ) -> NiceGUIPreferences:
        """Load app preferences or return defaults."""
        key = _preferences_key(owner_user_id)
        with self.engine.begin() as connection:
            row = (
                connection.execute(
                    select(app_preferences.c.payload_json).where(
                        app_preferences.c.key == key
                    )
                )
                .mappings()
                .first()
            )
        if row is None:
            return NiceGUIPreferences()
        return _load_model_json(
            raw=row["payload_json"],
            model_type=NiceGUIPreferences,
            field_name="app_preferences.payload_json",
        )  # type: ignore[return-value]

    def save_preferences(
        self, preferences: NiceGUIPreferences, *, owner_user_id: str | None = None
    ) -> None:
        """Persist app preferences."""
        key = _preferences_key(owner_user_id)
        payload = _dump_model(preferences)
        with self.engine.begin() as connection:
            existing = connection.execute(
                select(app_preferences.c.key).where(app_preferences.c.key == key)
            ).first()
            if existing is None:
                connection.execute(
                    insert(app_preferences).values(key=key, payload_json=payload)
                )
            else:
                connection.execute(
                    update(app_preferences)
                    .where(app_preferences.c.key == key)
                    .values(payload_json=payload)
                )

    def create_user(
        self,
        *,
        username: str,
        password_hash: str,
        role: NiceGUIUserRole = "user",
    ) -> NiceGUIUser:
        """Create a local hosted-mode user."""
        timestamp = _now_iso()
        user = NiceGUIUser(
            user_id=f"user-{uuid4().hex}",
            username=username.strip(),
            password_hash=password_hash,
            role=role,
            created_at=timestamp,
            updated_at=timestamp,
        )
        with self.engine.begin() as connection:
            connection.execute(
                insert(users).values(
                    user_id=user.user_id,
                    username=user.username,
                    password_hash=user.password_hash,
                    role=user.role,
                    disabled=user.disabled,
                    created_at=user.created_at,
                    updated_at=user.updated_at,
                    last_login_at=user.last_login_at,
                )
            )
        return user

    def list_users(self) -> list[NiceGUIUser]:
        """Return local users ordered by username."""
        with self.engine.begin() as connection:
            rows = (
                connection.execute(select(users).order_by(users.c.username.asc()))
                .mappings()
                .all()
            )
        return [self._user_from_row(row) for row in rows]

    def get_user_by_username(self, username: str) -> NiceGUIUser | None:
        """Return one local user by username."""
        with self.engine.begin() as connection:
            row = (
                connection.execute(
                    select(users).where(users.c.username == username.strip())
                )
                .mappings()
                .first()
            )
        return None if row is None else self._user_from_row(row)

    def get_user(self, user_id: str) -> NiceGUIUser | None:
        """Return one local user by id."""
        with self.engine.begin() as connection:
            row = (
                connection.execute(select(users).where(users.c.user_id == user_id))
                .mappings()
                .first()
            )
        return None if row is None else self._user_from_row(row)

    def update_user_login(self, user_id: str) -> None:
        """Record a successful login timestamp."""
        with self.engine.begin() as connection:
            connection.execute(
                update(users)
                .where(users.c.user_id == user_id)
                .values(last_login_at=_now_iso(), updated_at=_now_iso())
            )

    def set_user_disabled(self, user_id: str, disabled: bool) -> None:
        """Enable or disable one local user."""
        with self.engine.begin() as connection:
            connection.execute(
                update(users)
                .where(users.c.user_id == user_id)
                .values(disabled=disabled, updated_at=_now_iso())
            )

    def update_user_password_hash(self, user_id: str, password_hash: str) -> None:
        """Replace one local user's password hash."""
        with self.engine.begin() as connection:
            connection.execute(
                update(users)
                .where(users.c.user_id == user_id)
                .values(password_hash=password_hash, updated_at=_now_iso())
            )

    def create_user_session(
        self, *, user_id: str, token_hash: str, expires_at: str
    ) -> NiceGUIUserSession:
        """Create a durable hosted-mode browser session record."""
        session = NiceGUIUserSession(
            session_id=f"session-{uuid4().hex}",
            user_id=user_id,
            token_hash=token_hash,
            created_at=_now_iso(),
            expires_at=expires_at,
        )
        with self.engine.begin() as connection:
            connection.execute(
                insert(user_sessions).values(
                    session_id=session.session_id,
                    user_id=session.user_id,
                    token_hash=session.token_hash,
                    created_at=session.created_at,
                    expires_at=session.expires_at,
                    revoked_at=session.revoked_at,
                )
            )
        return session

    def get_user_session(self, session_id: str) -> NiceGUIUserSession | None:
        """Return one durable hosted-mode browser session."""
        with self.engine.begin() as connection:
            row = (
                connection.execute(
                    select(user_sessions).where(
                        user_sessions.c.session_id == session_id
                    )
                )
                .mappings()
                .first()
            )
        return None if row is None else self._user_session_from_row(row)

    def revoke_user_session(self, session_id: str) -> None:
        """Mark one hosted-mode browser session as revoked."""
        with self.engine.begin() as connection:
            connection.execute(
                update(user_sessions)
                .where(user_sessions.c.session_id == session_id)
                .values(revoked_at=_now_iso())
            )

    def upsert_secret_record(
        self,
        *,
        owner_user_id: str,
        session_id: str,
        name: str,
        ciphertext: str,
    ) -> NiceGUISecretRecord:
        """Create or replace one encrypted secret record."""
        timestamp = _now_iso()
        with self.engine.begin() as connection:
            existing = (
                connection.execute(
                    select(secret_records).where(
                        secret_records.c.owner_user_id == owner_user_id,
                        secret_records.c.session_id == session_id,
                        secret_records.c.name == name,
                    )
                )
                .mappings()
                .first()
            )
            if existing is None:
                secret_id = f"secret-{uuid4().hex}"
                connection.execute(
                    insert(secret_records).values(
                        secret_id=secret_id,
                        owner_user_id=owner_user_id,
                        session_id=session_id,
                        name=name,
                        ciphertext=ciphertext,
                        created_at=timestamp,
                        updated_at=timestamp,
                    )
                )
                return NiceGUISecretRecord(
                    secret_id=secret_id,
                    owner_user_id=owner_user_id,
                    session_id=session_id,
                    name=name,
                    ciphertext=ciphertext,
                    created_at=timestamp,
                    updated_at=timestamp,
                )
            connection.execute(
                update(secret_records)
                .where(secret_records.c.secret_id == existing["secret_id"])
                .values(ciphertext=ciphertext, updated_at=timestamp)
            )
        return NiceGUISecretRecord(
            secret_id=str(existing["secret_id"]),
            owner_user_id=owner_user_id,
            session_id=session_id,
            name=name,
            ciphertext=ciphertext,
            created_at=str(existing["created_at"]),
            updated_at=timestamp,
        )

    def get_secret_record(
        self, *, owner_user_id: str, session_id: str, name: str
    ) -> NiceGUISecretRecord | None:
        """Return one encrypted secret record."""
        with self.engine.begin() as connection:
            row = (
                connection.execute(
                    select(secret_records).where(
                        secret_records.c.owner_user_id == owner_user_id,
                        secret_records.c.session_id == session_id,
                        secret_records.c.name == name,
                    )
                )
                .mappings()
                .first()
            )
        return None if row is None else self._secret_record_from_row(row)

    def list_secret_names(self, *, owner_user_id: str, session_id: str) -> list[str]:
        """Return configured secret names for one user chat session."""
        with self.engine.begin() as connection:
            rows = (
                connection.execute(
                    select(secret_records.c.name)
                    .where(
                        secret_records.c.owner_user_id == owner_user_id,
                        secret_records.c.session_id == session_id,
                    )
                    .order_by(secret_records.c.name.asc())
                )
                .scalars()
                .all()
            )
        return [str(name) for name in rows]

    def delete_secret_record(
        self, *, owner_user_id: str, session_id: str, name: str
    ) -> None:
        """Delete one encrypted secret record."""
        with self.engine.begin() as connection:
            connection.execute(
                delete(secret_records).where(
                    secret_records.c.owner_user_id == owner_user_id,
                    secret_records.c.session_id == session_id,
                    secret_records.c.name == name,
                )
            )

    def assign_unowned_sessions(self, owner_user_id: str) -> None:
        """Assign legacy local sessions to the first hosted-mode admin."""
        with self.engine.begin() as connection:
            connection.execute(
                update(chat_sessions)
                .where(chat_sessions.c.owner_user_id.is_(None))
                .values(owner_user_id=owner_user_id)
            )

    def record_auth_event(
        self, *, user_id: str | None, event_type: str, detail: object | None = None
    ) -> None:
        """Append a minimal hosted-mode auth audit event."""
        with self.engine.begin() as connection:
            connection.execute(
                insert(auth_events).values(
                    user_id=user_id,
                    event_type=event_type,
                    detail_json=_dump_json(detail or {}),
                    created_at=_now_iso(),
                )
            )

    def _user_from_row(self, row: Any) -> NiceGUIUser:
        return NiceGUIUser(
            user_id=str(row["user_id"]),
            username=str(row["username"]),
            password_hash=str(row["password_hash"]),
            role=row["role"],
            disabled=bool(row["disabled"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
            last_login_at=row["last_login_at"],
        )

    def _user_session_from_row(self, row: Any) -> NiceGUIUserSession:
        return NiceGUIUserSession(
            session_id=str(row["session_id"]),
            user_id=str(row["user_id"]),
            token_hash=str(row["token_hash"]),
            created_at=str(row["created_at"]),
            expires_at=str(row["expires_at"]),
            revoked_at=row["revoked_at"],
        )

    def _secret_record_from_row(self, row: Any) -> NiceGUISecretRecord:
        return NiceGUISecretRecord(
            secret_id=str(row["secret_id"]),
            owner_user_id=str(row["owner_user_id"]),
            session_id=str(row["session_id"]),
            name=str(row["name"]),
            ciphertext=str(row["ciphertext"]),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )

    def _message_count(self, session_id: str) -> int:
        with self.engine.begin() as connection:
            count = connection.execute(
                select(func.count())
                .select_from(chat_messages)
                .where(
                    chat_messages.c.session_id == session_id,
                    chat_messages.c.show_in_transcript.is_(True),
                )
            ).scalar_one()
        return int(count)

    def _session_values(self, record: NiceGUISessionRecord) -> dict[str, object]:
        return {
            "session_id": record.summary.session_id,
            "title": record.summary.title,
            "created_at": record.summary.created_at,
            "updated_at": record.summary.updated_at,
            "root_path": record.runtime.root_path,
            "provider": record.runtime.provider.value,
            "model_name": record.runtime.model_name,
            "runtime_json": _dump_model(record.runtime),
            "workflow_session_state_json": _dump_model(record.workflow_session_state),
            "token_usage_json": (
                None if record.token_usage is None else _dump_model(record.token_usage)
            ),
            "inspector_state_json": _dump_model(record.inspector_state),
            "confidence": record.confidence,
            "temporary": record.summary.temporary,
            "project_id": record.summary.project_id,
            "owner_user_id": record.summary.owner_user_id,
        }

    def _message_values(
        self,
        session_id: str,
        *,
        ordinal: int,
        entry: NiceGUITranscriptEntry,
    ) -> dict[str, object]:
        return {
            "session_id": session_id,
            "ordinal": ordinal,
            "role": entry.role,
            "text": entry.text,
            "final_response_json": (
                None
                if entry.final_response is None
                else _dump_model(entry.final_response)
            ),
            "assistant_completion_state": entry.assistant_completion_state,
            "show_in_transcript": entry.show_in_transcript,
            "created_at": entry.created_at or _now_iso(),
        }

    def _transcript_entry_from_row(self, row: Any) -> NiceGUITranscriptEntry:
        final_response = (
            None
            if row["final_response_json"] is None
            else _load_model_json(
                raw=row["final_response_json"],
                model_type=ChatFinalResponse,
                field_name="chat_messages.final_response_json",
            )
        )
        return NiceGUITranscriptEntry(
            role=row["role"],
            text=str(row["text"]),
            final_response=final_response,  # type: ignore[arg-type]
            assistant_completion_state=row["assistant_completion_state"],
            show_in_transcript=bool(row["show_in_transcript"]),
            created_at=str(row["created_at"]),
        )

    def _workbench_values(
        self, session_id: str, item: NiceGUIWorkbenchItem
    ) -> dict[str, object]:
        return {
            "item_id": item.item_id,
            "session_id": session_id,
            "kind": item.kind,
            "title": item.title,
            "payload_json": _dump_json(item.payload),
            "version": item.version,
            "active": item.active,
            "started_at": item.started_at,
            "finished_at": item.finished_at,
            "duration_seconds": item.duration_seconds,
            "created_at": item.created_at,
            "updated_at": item.updated_at,
        }


__all__ = [
    "NICEGUI_DB_ENV_VAR",
    "NiceGUIChatStoreCorruptionError",
    "NiceGUIChatStoreError",
    "SQLiteNiceGUIChatStore",
    "app_preferences",
    "auth_events",
    "chat_messages",
    "chat_sessions",
    "create_sqlite_engine",
    "default_db_path",
    "metadata",
    "remember_default_db_path",
    "secret_records",
    "user_sessions",
    "users",
    "workbench_items",
]
