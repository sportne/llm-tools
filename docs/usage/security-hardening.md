# Security Hardening

`llm-tools` now includes multiple execution surfaces: local filesystem tools,
subprocess-backed Git helpers, remote enterprise read integrations, structured
provider calls, a Streamlit assistant client, and durable harness-backed
research sessions. Treat configs, examples, caches, and persisted session state
as part of the security surface.

## Dependency Surface

Runtime and optional dependency exposure includes:

- `openai` and `instructor` for the OpenAI-compatible provider layer and
  structured response parsing
- `atlassian-python-api` for Jira, Bitbucket, and Confluence tools
- `python-gitlab` for GitLab read tools
- `markitdown` for office/document conversion during read-oriented filesystem
  access
- `mpxj` plus Java for Microsoft Project file reads
- `streamlit` and YAML loading when installing `.[streamlit]`

Only enable the integrations you actually need in the current environment.

## Assistant Permissions And Approvals

In `llm_tools.apps.streamlit_assistant`:

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

## Secret Handling

Prefer one of these patterns for credentials:

- environment variables consumed by providers or integrations
- the Streamlit assistant's session-only API-key entry

Do not commit secrets into assistant YAML configs, scripted harness payloads, or
examples. If you need local overrides, keep them in ignored files.

## Persisted Data And Caches

Default storage locations:

- Streamlit assistant state: `~/.llm-tools/assistant/streamlit`
- Streamlit assistant research store: `~/.llm-tools/assistant/streamlit/research`
- Harness CLI state: `~/.llm-tools/harness`

Cache locations are split by feature:

- Filesystem/document conversion cache: workspace-local `.llm_tools/cache/read_file`
- Confluence attachment cache: platform temp dir under
  `tempfile.gettempdir()/llm_tools/confluence_attachment_cache`

If you intentionally redirect any of these into a repository checkout, keep the
paths ignored. This repository ignores `.llm_tools/` in addition to the older
`.llm-tools/` scratch paths.

## Pending Approval Snapshots

Newly persisted harness approval records store only a scrubbed base context:

- preserved: `invocation_id`, `workspace`, and `metadata`
- cleared before persistence: process environment variables, logs, artifacts,
  and source provenance
- rebuilt on resume: execution context derived from the stored base context plus
  the current process environment

Pending approval turns also keep a minimal approval-audit record so replay and
inspection can show approval status without persisting raw request payloads.

Non-approved approval outcomes are fail-closed: denial or timeout records the
blocked invocation, but later invocations from that same model response do not
continue running.

Older snapshots created before this hardening change may still contain raw
environment data. Delete those persisted session files if they may have held
sensitive values. The repository does not migrate or scrub old snapshots in
place.

## Harness Replay And Inspection Artifacts

Persisted harness `summary` and `trace` artifacts should be treated as cache-only
derived views. Canonical `HarnessState` remains authoritative for resume,
replay, and inspection, so cached artifacts may be rebuilt or ignored when
absent, stale, inconsistent, or corrupt.

Persisted trace payloads should stay minimal. Keep redacted policy metadata,
status summaries, identifiers, and explicit artifact references when needed, but
do not rely on stored traces to preserve raw request arguments, environment
state, or other unredacted payloads by default.

Malformed file-backed harness session records should be isolated as corruption
outcomes so list and load flows can skip or surface them without trusting the
damaged file's cached artifacts.
