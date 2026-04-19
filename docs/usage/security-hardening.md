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
paths ignored. This repository now ignores `.llm_tools/` in addition to the
older `.llm-tools/` scratch paths.

## Pending Approval Snapshots

Newly persisted harness approval records store only a scrubbed base context:

- preserved: `invocation_id`, `workspace`, and `metadata`
- cleared before persistence: process env, logs, artifacts, and source
  provenance
- rebuilt on resume: execution context derived from the stored base context plus
  the current process environment

Older snapshots written before this hardening pass may still contain raw
environment data. Delete those persisted files if they may have held secrets.
