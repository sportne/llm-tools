# SQLite Persistence

## Goal

The NiceGUI client uses SQLCipher-backed SQLite through SQLAlchemy 2.x for
durable chat state. NiceGUI is the supported interactive chat client; legacy
file-per-session app storage is not migrated in v1.

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

The database path must be treated as local to the running assistant server.
Multiple local app instances on different machines must not share one SQLite
database file through a network drive. SQLite WAL coordination depends on local
filesystem semantics, and this app does not implement a cross-machine database
proxy or conflict-resolution layer. Multi-user deployments should run one
assistant server process with the database on storage local to that server.

The database file is always encrypted with SQLCipher. The app creates or reads
the SQLCipher key from:

```text
~/.llm-tools/assistant/nicegui/hosted/db.key
```

User-owned fields are also encrypted inside the opened database using per-user
data keys. Those user keys are wrapped by a local server key stored at:

```text
~/.llm-tools/assistant/nicegui/hosted/user-kek.key
```

Losing either key file makes the corresponding encrypted data unrecoverable.

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

`owner_user_id` is populated for normal authenticated local and hosted sessions.
It is nullable only for the explicit `--auth-mode none` development/test escape
hatch, which uses an internal local key identifier for field encryption.
`project_id` is nullable and reserved so future project workspaces can be added
without rewriting the core session/message schema.

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

Preferences are keyed by owner in authenticated local and hosted mode, and by a
default local key only when `--auth-mode none` is explicitly used.

### `users`

Stores local assistant users:

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

Stores authenticated browser sessions:

- `session_id`
- `user_id`
- `token_hash`
- `created_at`
- `expires_at`
- `revoked_at`

Only token hashes are stored in SQLite.

### `auth_events`

Stores minimal local auth audit events:

- `event_id`
- `user_id`
- `event_type`
- `detail_json`
- `created_at`

### `user_key_records`

Stores wrapped per-user encryption keys:

- `owner_key_id`
- `wrapped_key`
- `algorithm`
- `key_version`
- `created_at`
- `rotated_at`

Raw user data keys are never stored in SQLite.

## Validation

Relational columns store the minimum queryable metadata needed for ownership,
ordering, and status. User-owned content fields are encrypted envelopes and are
validated with Pydantic models after decryption:

- runtime config
- workflow session state
- token usage
- transcript final responses
- inspector payload wrappers
- preferences
- hosted users and browser sessions

Malformed persisted JSON raises a store corruption error with the affected
field name rather than silently dropping data.

Encrypted envelopes use authenticated encryption. The authentication context
includes the table, row id, owner key id, column, and key version so copied or
swapped encrypted values fail during decryption.

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
- user, user-session, and auth-event helpers for hosted mode

Temporary sessions are returned as normal typed records but are not inserted into
the database.

Provider API keys and tool credentials are not part of the SQLite schema. The
assistant app keeps typed credential values in server memory for the current
browser/app session only.
