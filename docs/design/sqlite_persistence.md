# SQLite Persistence

## Goal

The NiceGUI client uses SQLite through SQLAlchemy 2.x for durable chat state.
This replaces the file-per-session approach for the new app only; Streamlit
storage remains unchanged.

## Location

Default database:

```text
~/.llm-tools/assistant/nicegui/chat.sqlite3
```

Override:

```text
LLM_TOOLS_NICEGUI_DB=/path/to/chat.sqlite3
```

The store uses synchronous SQLAlchemy with one short-lived database transaction
per operation. SQLite foreign keys are enabled and WAL is requested for normal
file-backed databases.

## Tables

### `chat_sessions`

Stores durable session metadata plus validated JSON fields:

- `session_id`
- `owner_user_id`
- `title`
- `created_at`
- `updated_at`
- `root_path`
- `provider`
- `model_name`
- `runtime_json`
- `workflow_session_state_json`
- `token_usage_json`
- `inspector_state_json`
- `confidence`
- `temporary`
- `project_id`

`owner_user_id` is nullable for legacy local loopback sessions and populated for
hosted-mode private user data. `project_id` is nullable and reserved so future
project workspaces can be added without rewriting the core session/message
schema.

### `chat_messages`

Stores transcript entries in ordinal order:

- `id`
- `session_id`
- `ordinal`
- `role`
- `text`
- `final_response_json`
- `assistant_completion_state`
- `show_in_transcript`
- `created_at`

### `workbench_items`

Stores inspector/artifact-style records:

- `item_id`
- `session_id`
- `kind`
- `title`
- `payload_json`
- `version`
- `active`
- `created_at`
- `updated_at`
- `started_at`
- `finished_at`
- `duration_seconds`

Only inspector-style workbench records are produced in v1. The table shape is
ready for future Canvas/Artifacts-like outputs.

### `app_preferences`

Stores a singleton JSON preferences payload:

- theme/sidebar/workbench settings
- active session id
- recent roots
- recent models
- recent base URLs

Preferences are keyed by owner in hosted mode and by a default local key in
loopback mode.

### `users`

Stores hosted-mode local users:

- `user_id`
- `username`
- `password_hash`
- `role`
- `disabled`
- `created_at`
- `updated_at`
- `last_login_at`

Passwords are Argon2 hashes. Users are admin-created only in v1.

### `user_sessions`

Stores hosted-mode browser sessions:

- `session_id`
- `user_id`
- `token_hash`
- `created_at`
- `expires_at`
- `revoked_at`

Only token hashes are stored in SQLite.

### `secret_records`

Stores hosted-mode encrypted secrets:

- `secret_id`
- `owner_user_id`
- `session_id`
- `name`
- `ciphertext`
- `created_at`
- `updated_at`

The encrypted value is scoped to one user and one chat session. The master key is
stored outside SQLite under the hosted key path.

### `auth_events`

Stores minimal local auth audit events:

- `event_id`
- `user_id`
- `event_type`
- `detail_json`
- `created_at`

## Validation

Relational columns store queryable metadata. Complex fields are stored as JSON
but validated with Pydantic models on write and on read:

- runtime config
- workflow session state
- token usage
- transcript final responses
- inspector payload wrappers
- preferences
- hosted users, browser sessions, and encrypted secret metadata

Malformed persisted JSON raises a store corruption error with the affected
field name rather than silently dropping data.

## Store API

The v1 API is intentionally small:

- `initialize()`
- `list_sessions(limit=None, query=None)`
- `create_session(runtime_config, title=None, temporary=False, owner_user_id=None)`
- `load_session(session_id, owner_user_id=None)`
- `save_session(record)`
- `append_message(session_id, entry)`
- `delete_session(session_id)`
- `load_preferences(owner_user_id=None)`
- `save_preferences(preferences, owner_user_id=None)`
- user, user-session, encrypted-secret, and auth-event helpers for hosted mode

Temporary sessions are returned as normal typed records but are not inserted into
the database.
