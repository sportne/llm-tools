# Security

`llm-tools` includes multiple sensitive execution surfaces: local filesystem
tools, subprocess-backed Git helpers, remote enterprise read integrations,
structured provider calls, a LLM Tools Assistant client, and durable
harness-backed research sessions. Treat configs, examples, caches, and
persisted session state as part of the security surface.

## Current posture

The current baseline is:

- architectural layering is enforced by tests
- tool execution remains runtime-mediated through `tool_api`
- central policy, approval, and redaction controls exist in the lower layers
- harness persistence now treats stored summaries and traces as cache-only
  derived views
- persisted approval waits and interrupted turns now fail closed by default
- malformed file-backed harness session records are treated as corruption

Security-sensitive shipped surfaces include:

- filesystem, subprocess, and network-capable tools
- OpenAI-compatible provider transport
- persisted harness sessions and replay data
- the LLM Tools Assistant and harness CLI entrypoints
- the assistant app, including optional hosted multi-user mode

## Operational guidance

### Dependency surface

Runtime dependency exposure includes:

- `openai` and `instructor` for the OpenAI-compatible provider layer and
  structured response parsing
- `atlassian-python-api` for Jira, Bitbucket, and Confluence tools
- `python-gitlab` for GitLab read tools
- `markitdown` for office and document conversion during read-oriented
  filesystem access
- `mpxj` plus Java for Microsoft Project file reads
- `nicegui` and `PyYAML` for the shipped assistant UX and config loading

Only enable the integrations you actually need in the current environment.

### Assistant permissions and approvals

In `llm_tools.apps.assistant_app`:

- selecting a workspace root only picks the directory available to local file
  and subprocess tools
- fresh sessions start with network, filesystem, and subprocess permissions off
- write-capable side effects are approval-gated by default, even after a tool is
  enabled
- additional approval requirements can come from `policy.require_approval_for`
  in assistant config

The sidebar runtime controls are session-scoped. "New session from current
setup" clones the current model, enabled tools, permissions, and approval
settings into the new session.

### Secret Handling

Prefer one of these patterns for credentials:

- environment variables consumed by providers or integrations
- the LLM Tools Assistant's session-only API-key entry
- the assistant app's session-scoped credential entry

Do not commit secrets into assistant YAML configs, scripted harness payloads, or
examples. If you need local overrides, keep them in ignored files.

The assistant app intentionally does not use process environment variables
as implicit provider or tool credentials. Provider API keys and tool credentials
must be typed into the app for the active browser/app session. In both normal
local loopback use and hosted use those values are held in server memory only.
They are not written to SQLite and are not restored after a server restart,
browser session reset, or logout.

Non-secret service URLs, such as Atlassian or GitLab base URLs, are runtime
configuration and may be persisted as normal chat settings. Keep bearer tokens,
passwords, API keys, and PATs in the credential fields instead.

assistant persistence uses encrypted SQLite only. The database is opened through
SQLCipher with a local database key file, and user-owned chat fields are also
encrypted with per-user data keys wrapped by a local server key file. Deleting
either key file makes the affected database data unrecoverable. Treat these
files as server secrets and back them up separately from normal database copies:

```text
~/.llm-tools/assistant/nicegui/hosted/db.key
~/.llm-tools/assistant/nicegui/hosted/user-kek.key
```

### NiceGUI Auth Mode

The default assistant app uses local username/password authentication even on
loopback. The first launch requires creating an admin user. `--auth-mode none`
is an explicit development/test escape hatch and should not be used for normal
local or hosted use. Binding the app to a non-loopback interface still requires
local authentication:

```text
llm-tools-assistant --host 0.0.0.0 --auth-mode local
```

NiceGUI auth uses local admin-created users only; there is no public
self-registration. Users see only their own chat sessions, preferences,
workbench records, temporary sessions, and in-memory credential state. Admins
can create, disable, and reset local users, but v1 does not add a cross-user
chat browser.

Passwords are stored as Argon2 hashes. Browser sessions use random server-side
session tokens recorded as hashes in SQLite and transported through NiceGUI's
signed session storage with HttpOnly SameSite cookie settings. The cookie is
marked HTTPS-only when direct TLS is enabled.

Per-user chat encryption uses server-wrapped user keys. This keeps admin
password resets practical, but it means a server-key compromise can decrypt user
data. It protects against copied database files and accidental cross-user data
exposure, not against a fully compromised web server.

Use a TLS-terminating reverse proxy for normal hosted deployments. Set
`--public-base-url` to the HTTPS URL. Direct `--tls-certfile` and
`--tls-keyfile` support exists for bootstrap and self-signed certificate testing.
If the app is reachable over non-loopback HTTP, secret entry is disabled unless
`--allow-insecure-hosted-secrets` is passed as an explicit risk acceptance.

Future smart-card, OAuth, or OIDC authentication should plug in behind the auth
provider boundary. In an OAuth/OIDC flow, the browser redirects to the identity
provider, the user authenticates there, and this server exchanges the returned
authorization code for tokens. Long-lived refresh tokens must stay server-side,
preferably in a future local secret-store boundary or local secret daemon. The
current NiceGUI hosted mode does not persist provider API keys or tool
credentials.

### Persisted data and caches

Default storage locations:

- LLM Tools Assistant state: `~/.llm-tools/assistant/nicegui/chat.sqlite3`
- LLM Tools Assistant key files: `~/.llm-tools/assistant/nicegui/hosted`
- Harness CLI state: `~/.llm-tools/harness`

Cache locations are split by feature:

- filesystem and document conversion cache: workspace-local
  `.llm_tools/cache/read_file`
- Confluence attachment cache: platform temp dir under
  `tempfile.gettempdir()/llm_tools/confluence_attachment_cache`

If you intentionally redirect any of these into a repository checkout, keep the
paths ignored. This repository ignores `.llm_tools/` in addition to the older
`.llm-tools/` scratch paths.

### Pending approval snapshots

Newly persisted harness approval records store only a scrubbed base context:

- preserved: `invocation_id`, `workspace`, and `metadata`
- cleared before persistence: process environment variables, logs, artifacts,
  and source provenance
- rebuilt on resume: execution context derived from the stored base context plus
  the current process environment

Pending approval turns also keep a minimal approval-audit record so replay and
inspection can show approval status without persisting raw request payloads.

Non-approved approval outcomes are fail-closed: denial, expiration, or operator
cancel records the blocked invocation, but later invocations from that same
paused model turn do not continue running.

Older snapshots created before this hardening change may still contain raw
environment data. Delete those persisted session files if they may have held
sensitive values. The repository does not migrate or scrub old snapshots in
place.

### Harness replay and inspection artifacts

Persisted harness `summary` and `trace` artifacts should be treated as cache-only
derived views. Canonical `HarnessState` remains authoritative for resume,
replay, and inspection, so cached artifacts may be rebuilt or ignored when
absent, stale, inconsistent, or corrupt.

Persisted trace payloads should stay minimal. Keep redacted policy metadata,
status summaries, identifiers, and explicit artifact references when needed, but
do not rely on stored traces to preserve raw request arguments, environment
state, or other unredacted payloads by default.

Harness turns checkpoint an incomplete tail record before provider or tool
execution begins. Resume classifies those partial non-approval turns as
`interrupted` and fails closed by default; callers must opt in before the tail
turn is dropped and replayed.

Protection-triggered purge is broader than final-answer replacement. Persisted
tool-result payloads, logs, artifacts, and execution-record outputs are scrubbed
from stored harness turns so replay, raw inspection payloads, and NiceGUI
deep-task detail views do not re-render the protected material.

### Remote enterprise tools

Confluence reads are split intentionally:

- `read_confluence_page` is a pure remote page read with no filesystem write
- `read_confluence_attachment` downloads attachment bytes into the internal
  cache and therefore requires filesystem permission plus `LOCAL_WRITE`

Jira issue reads no longer expose the full remote `fields` map by default.
Use `requested_fields` when a caller needs additional fields beyond the
allowlisted summary view. GitLab, Jira, Bitbucket, and Confluence collection
reads now apply hard limits and surface truncation metadata instead of returning
unbounded result sets, and the shipped remote tool specs set explicit per-tool
timeouts with retryability only for transient upstream failures.

## Active hardening backlog

Open work:

- `tools`: cross-check every built-in tool spec against actual side effects,
  capability flags, and required secrets
- `workflow_api` and model mediation: identify remaining attack paths where
  model-controlled content could trigger unexpected tool execution, unbounded
  work, or sensitive data disclosure
- `apps`: finish review of shared app config, prompt helpers, `assistant_app`,
  app compatibility surfaces, and `harness_cli`
- project-wide: re-run relevant tests for confirmed security issues and produce
  a final confidence summary

Deferred work:

- bind persisted approval resumes to a reviewed integrity mechanism rather than
  trusting unauthenticated stored approval payloads
- constrain approval resume environment rehydration to a reviewed allowlist or
  stable approved snapshot instead of the full current process environment
- minimize prompt-protection correction state and inspector payload retention
  further once the long-term contract is settled

## Completed review summary

### 2026-04-19: `workflow_api` sequencing, approvals, protection, and replay

- Reviewed the workflow execution path, approval behavior, protection state, and
  replay or trace exposure across `workflow_api`, `tool_api`, `llm_adapters`,
  and `harness_api`.
- Confirmed and then addressed fail-open approval behavior and raw request
  retention in workflow-adjacent replay paths.
- Remaining deferred items are approval-record integrity binding and further
  prompt-protection retention minimization.

### 2026-04-19: `harness_api` lifecycle, control flow, persistence, and replay

- Reviewed durable models, executor control flow, resume semantics, replay,
  summaries, persistence hardening, and corruption handling.
- Addressed `completed` terminalization on blocked work, crash-before-save
  replay risk through incomplete-turn checkpointing, over-budget dispatch, stale
  summary or trace trust, and corrupt file handling.
- Remaining deferred item is narrowing approval-resume environment rehydration.

### 2026-04-19: architecture and security coverage review

- Reviewed architecture tests plus security-relevant tests across tools,
  providers, workflow, harness, and apps.
- Addressed missing regressions for replay, approval, inspection, purge
  propagation, and indirect runtime-bypass shapes.
- Remaining work is project-wide confidence closure rather than the previously
  identified coverage gaps.

### 2026-04-19: GitLab and Atlassian tool-family review

- Reviewed credential handling, request scoping, pagination, remote trust
  assumptions, timeout behavior, and tool contract accuracy.
- Addressed Confluence attachment cache side effects, Jira field exposure,
  bounded collection outputs, and explicit timeout or retry behavior.
- Remaining open work is the broader cross-check of all built-in tool specs.

## Review evidence model

Security review entries in this repository use a consistent evidence bar:

- reviewed code paths are named explicitly
- relevant tests are listed, including whether they were inspected or run
- findings are recorded with severity and affected components
- remediation or deferral status is captured explicitly
- residual risk is summarized even when most findings are addressed

Severity buckets:

- `Critical`: broken core trust boundary or broad compromise path
- `High`: material weakness with realistic exploitation or strong impact
- `Medium`: meaningful weakness with narrower preconditions or partial
  mitigation
- `Low`: defense-in-depth issue or hardening gap
