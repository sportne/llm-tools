"""Compatibility facade for assistant app SQLite persistence internals."""

from __future__ import annotations

import sys

from llm_tools.apps.assistant_app import store_sqlite as _store_sqlite
from llm_tools.apps.assistant_app.store_sqlite import (
    NICEGUI_DB_ENV_VAR,
    NiceGUIChatStoreCorruptionError,
    NiceGUIChatStoreError,
    SQLiteNiceGUIChatStore,
    SQLiteNiceGUIHarnessStateStore,
    app_preferences,
    auth_events,
    chat_messages,
    chat_sessions,
    create_sqlcipher_engine,
    default_db_path,
    harness_sessions,
    remember_default_db_path,
    user_key_records,
    user_sessions,
    users,
    workbench_items,
)

__all__ = [
    "NICEGUI_DB_ENV_VAR",
    "NiceGUIChatStoreCorruptionError",
    "NiceGUIChatStoreError",
    "SQLiteNiceGUIChatStore",
    "SQLiteNiceGUIHarnessStateStore",
    "app_preferences",
    "auth_events",
    "chat_messages",
    "chat_sessions",
    "create_sqlcipher_engine",
    "default_db_path",
    "harness_sessions",
    "remember_default_db_path",
    "user_key_records",
    "user_sessions",
    "users",
    "workbench_items",
]

sys.modules[__name__] = _store_sqlite
