# Security Hardening

This repository ships multiple execution surfaces: local filesystem tools,
subprocess-backed Git helpers, remote enterprise read integrations, document
conversion helpers, and durable harness-backed session storage. Treat docs,
configs, and examples as part of that security surface.

## Dependency Surface

The base package installs more than the minimal local runtime:

- `openai` and `instructor` back the OpenAI-compatible provider layer and
  structured response parsing.
- `atlassian-python-api` enables Jira, Confluence, and Bitbucket tools when
  those tools are enabled and the corresponding credentials are present.
- `python-gitlab` enables the GitLab read tools when those tools are enabled
  and credentials are present.
- `markitdown` converts supported office and document formats into markdown
  when filesystem reads hit non-text files such as PDF, Office, HTML, EPUB,
  or RTF documents.
- `mpxj` plus a working Java runtime are required for Microsoft Project
  (`.mpp`/`.mpt`) reads.

Installing `.[streamlit]` adds the interactive Streamlit clients and YAML
config loading support.

## Startup Permissions

In `llm_tools.apps.streamlit_assistant`, selecting a workspace root only
chooses the local directory that filesystem and subprocess tools may use. It
does not enable those permissions. The initial default runtime for a fresh app
load starts with:

- network access off
- filesystem access off
- subprocess access off

Enable each permission explicitly in the sidebar for the current session.
This applies even when a config sets `workspace.default_root` or the launch
command passes a directory argument. The "New Chat" action clones the current
session runtime, so branched sessions inherit whatever permissions are already
enabled there.

## Secret Handling

Prefer one of these patterns for credentials:

- environment variables used by the relevant provider or tool integration
- the assistant sidebar's session-only API-key prompt

Do not commit secrets into assistant YAML configs, scripted harness payloads,
or example files. If you need repo-local overrides, keep them in ignored
files outside version control.

## Persisted Data And Caches

Default storage locations:

- Streamlit assistant state: `~/.llm-tools/assistant/streamlit`
- Streamlit assistant research store: `~/.llm-tools/assistant/streamlit/research`
- Harness CLI state: `~/.llm-tools/harness`
- Converted document cache: `${TMPDIR:-/tmp}/llm_tools/read_file_cache` on
  POSIX systems, or the platform temp directory equivalent reported by
  Python's `tempfile.gettempdir()`

If you intentionally override these locations into a repository, keep the
paths ignored. This repository already ignores `.llm-tools/` and
`.llm-tools-harness/` for that reason.

## Pending Approval Snapshots

Newly persisted harness approval records store only a scrubbed base context:
`invocation_id`, `workspace`, and `metadata` are preserved, while process
environment variables, logs, artifacts, and source provenance are cleared
before the snapshot is written. Approval resume rebuilds execution context
from the stored base context plus the current process environment at resume
time. Non-approved approval outcomes are fail-closed: denial or timeout records
the blocked invocation, but later invocations from that same model response do
not continue running.

Older snapshots created before this hardening change may still contain raw
environment data. Delete those persisted session files if they may have held
sensitive values. The repository does not migrate or scrub old snapshots in
place.

## Harness Replay And Inspection Artifacts

Persisted harness `summary` and `trace` artifacts should be treated as
cache-only derived views. Canonical `HarnessState` remains authoritative for
resume, replay, and inspection, so cached artifacts may be rebuilt or ignored
when absent, stale, inconsistent, or corrupt.

Persisted trace payloads should stay minimal. Keep redacted policy metadata,
status summaries, identifiers, and explicit artifact references when needed,
but do not rely on stored traces to preserve raw request arguments,
environment state, or other unredacted payloads by default.

Malformed file-backed harness session records should be isolated as corruption
outcomes so list and load flows can skip or surface them without trusting the
damaged file's cached artifacts.

Harness turns now checkpoint an incomplete tail record before provider or tool
execution begins. Resume classifies those partial non-approval turns as
`interrupted` and fails closed by default; callers must opt in before the tail
turn is dropped and replayed. Empty task selections with remaining
non-terminal work now stop as `no_progress`, and tool-invocation budgets are
checked before dispatch so an over-budget parsed response does not partially
execute.

Protection-triggered purge is broader than final-answer replacement. Persisted
tool-result payloads, logs, artifacts, and execution-record outputs are scrubbed
from stored harness turns so replay, raw inspection payloads, and Streamlit
research detail views do not re-render the protected material.

## Remote Enterprise Tools

Confluence reads are split intentionally:

- `read_confluence_page` is a pure remote page read with no filesystem write.
- `read_confluence_attachment` downloads attachment bytes into the internal
  cache and therefore requires filesystem permission plus `LOCAL_WRITE`.

Jira issue reads no longer expose the full remote `fields` map by default.
Use `requested_fields` when a caller needs additional fields beyond the
allowlisted summary view. GitLab, Jira, Bitbucket, and Confluence collection
reads now apply hard limits and surface truncation metadata instead of
returning unbounded result sets, and the shipped remote tool specs set
explicit per-tool timeouts with retryability only for transient upstream
failures.
