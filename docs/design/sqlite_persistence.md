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

Only inspector-style workbench records are produced in v1. The table shape is
ready for future Canvas/Artifacts-like outputs.

### `app_preferences`

Stores a singleton JSON preferences payload:

- theme/sidebar/workbench settings
- active session id
- recent roots
- recent models
- recent base URLs

## Validation

Relational columns store queryable metadata. Complex fields are stored as JSON
but validated with Pydantic models on write and on read:

- runtime config
- workflow session state
- token usage
- transcript final responses
- inspector payload wrappers
- preferences

Malformed persisted JSON raises a store corruption error with the affected
field name rather than silently dropping data.

## Store API

The v1 API is intentionally small:

- `initialize()`
- `list_sessions(limit=None, query=None)`
- `create_session(runtime_config, title=None, temporary=False)`
- `load_session(session_id)`
- `save_session(record)`
- `append_message(session_id, entry)`
- `delete_session(session_id)`
- `load_preferences()`
- `save_preferences(preferences)`

Temporary sessions are returned as normal typed records but are not inserted into
the database.
