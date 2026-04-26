# NiceGUI Chat Client

## Goal

The NiceGUI chat client is a new app surface for `llm-tools` that follows the
conventional web-chat layout used by ChatGPT, Gemini, and Claude. It is not a
Streamlit migration. The first version reuses the existing assistant runtime and
adds a purpose-built browser UI plus SQLite persistence.

## Product Shape

The app is organized around four stable regions:

- A left chat rail for New Chat, Temporary Chat, search, recent sessions, and
  rename/delete affordances.
- A header for the current session title, provider/model controls, workspace
  root, turn status, settings, and the workbench toggle.
- A central transcript for user, assistant, system, and error messages,
  including continuation notices, tool timeline/status, copy, retry, and
  edit-last-user affordances as the implementation matures.
- A sticky composer with multiline input, send/stop controls, attachment/tool
  placeholders, provider mode indicator, and busy/disabled states.

The right workbench is an inspector/debug shell in v1. It intentionally reserves
the space and persistence shape for future Canvas/Artifacts-style outputs
without implementing artifact editing or version navigation yet.

## Runtime Contract

The client reuses the shared assistant runtime:

- `create_provider`
- assistant tool registry and executor helpers
- assistant policy/context helpers
- `ChatSessionTurnRunner`
- `ChatSessionState`, `ChatFinalResponse`, and `ChatTokenUsage`

The UI owns only presentation state, persistence, and background turn
coordination. Tool policy, approval handling, redaction, provider fallback,
prompt-tool parsing, native tools, and structured JSON execution stay in the
existing workflow layer.

## Turn Flow

1. Persist the user message immediately for durable sessions.
2. Build a provider and `ChatSessionTurnRunner` from the active session runtime.
3. Run the turn in a background thread.
4. Push workflow events onto a thread-safe queue.
5. Drain events on a NiceGUI timer and update transcript, status, approvals,
   inspector/workbench data, workflow state, and token usage.
6. Persist each durable state change through the SQLite store.

Temporary chats use the same in-memory models and runtime flow but are not
written to SQLite and do not appear in the recent-session rail.

## v1 Scope

Included:

- ChatGPT-like app shell.
- Durable sessions backed by SQLite.
- Temporary chat sessions.
- Provider/model/header controls.
- Background turns, stop, status, approvals, and inspector workbench.
- Settings sufficient to alter subsequent turns.

Deferred:

- Streamlit JSON-file migration.
- Project/folder workspaces.
- Full Canvas/Artifacts editing, export, and version browsing.
- Attachment ingestion.
- Multi-user auth.

