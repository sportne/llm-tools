# LLM Tools Assistant App

## Goal

The assistant app is a new app surface for `llm-tools` that follows the
conventional web-chat layout used by ChatGPT, Gemini, and Claude. It reuses the
existing assistant runtime and
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
  placeholders, response mode indicator, and busy/disabled states.

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
- Local admin-created users and per-user sessions for normal local and hosted use.

Deferred:

- Legacy JSON-file migration.
- Project/folder workspaces.
- Full Canvas/Artifacts editing, export, and version browsing.
- Attachment ingestion.
- External smart-card/OIDC auth.

## Local And Hosted Auth

The normal assistant app uses local username/password auth on both loopback and
hosted bindings. The first launch shows a local admin creation screen. Admins
create later users from the admin screen; there is no public self-registration
in v1.

```text
llm-tools-assistant
```

`--auth-mode none` remains available only as an explicit development/test
escape hatch. In normal authenticated use, each user sees only their own chats,
preferences, workbench records, temporary sessions, and in-memory credentials.
assistant persistence is always SQLCipher encrypted, and user-owned chat fields
are encrypted with per-user keys wrapped by a local server key.

For production-style hosting, put the app behind a TLS-terminating reverse proxy
such as Caddy, nginx, or Traefik and set `--public-base-url` to the HTTPS URL.
Direct `--tls-certfile` and `--tls-keyfile` are supported for short-term or
self-signed certificate testing. Non-loopback HTTP can run, but secret entry is
blocked unless `--allow-insecure-hosted-secrets` is passed.

The supported multi-user shape is one assistant server process with browser
clients. Running separate local app instances on multiple machines against a
shared SQLite file on a network drive is not supported. If the database is
remote from users, keep SQLite access on the server side rather than exposing
the database file as shared storage.

Provider API keys and tool credentials are not read from environment variables in
hosted mode. They are typed into the app and held in server memory for the
current browser/app session only, matching local loopback mode. Non-secret URLs
remain normal runtime configuration. In-memory credentials expire after 2 hours
by default and are re-entered through a focused prompt the next time the expired
credential is needed.

The SQLCipher database key and per-user key wrapping key are local server files.
They must be protected and backed up; deleting them makes encrypted NiceGUI data
unrecoverable.
