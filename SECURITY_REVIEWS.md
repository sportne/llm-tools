# Security Reviews

This file is the cumulative record for completed security assessments in
`llm-tools`.

## Architecture and Security Test Coverage Review

Date: `2026-04-19`

### Reviewed Areas

- `tests/architecture/`
- Security-relevant tests in `tests/tool_api/`, `tests/tools/`,
  `tests/llm_adapters/`, `tests/llm_providers/`, `tests/workflow_api/`,
  `tests/harness_api/`, and `tests/apps/`
- Supporting architecture and security documentation, including
  `docs/design/spec.md`, `docs/design/harness_api.md`,
  `docs/usage/security-hardening.md`, `docs/usage/streamlit-assistant.md`, and
  `SEC_TASKS.md`

### Findings

- `High`: No regression tests prove that research inspection, replay, approval,
  and raw inspection payload views redact secrets before rendering. The app
  exposes approval state and raw inspection payloads directly in the research
  detail surface, while the existing tests only assert that those views are
  present, not that sensitive fields are suppressed. References:
  [docs/usage/streamlit-assistant.md](/mnt/c/Users/steph/.codex/worktrees/4bc0/llm-tools/docs/usage/streamlit-assistant.md:111),
  [app.py](/mnt/c/Users/steph/.codex/worktrees/4bc0/llm-tools/src/llm_tools/apps/streamlit_assistant/app.py:2202),
  [test_streamlit_assistant.py](/mnt/c/Users/steph/.codex/worktrees/4bc0/llm-tools/tests/apps/test_streamlit_assistant.py:1608)
- `High`: No end-to-end regression proves that protection-triggered purge
  survives persistence, replay, inspection, and UI detail views. The purge path
  is implemented when harness state is saved, but existing coverage stops at
  scrub-helper unit tests and generic inspection/replay checks instead of
  validating the persisted and rendered outputs together. References:
  [executor.py](/mnt/c/Users/steph/.codex/worktrees/4bc0/llm-tools/src/llm_tools/harness_api/executor.py:820),
  [test_protection_scrub.py](/mnt/c/Users/steph/.codex/worktrees/4bc0/llm-tools/tests/harness_api/test_protection_scrub.py:20),
  [test_session_api.py](/mnt/c/Users/steph/.codex/worktrees/4bc0/llm-tools/tests/harness_api/test_session_api.py:99)
- `Medium`: Runtime-mediation architecture enforcement only detects direct
  `.invoke()` and `.ainvoke()` call sites. That leaves other bypass shapes
  unguarded, including direct use of implementation methods or internal runtime
  execution helpers outside the approved runtime boundary. References:
  [test_no_direct_tool_invocation.py](/mnt/c/Users/steph/.codex/worktrees/4bc0/llm-tools/tests/architecture/test_no_direct_tool_invocation.py:12),
  [_helpers.py](/mnt/c/Users/steph/.codex/worktrees/4bc0/llm-tools/tests/architecture/_helpers.py:102),
  [docs/design/spec.md](/mnt/c/Users/steph/.codex/worktrees/4bc0/llm-tools/docs/design/spec.md:151)
- `Medium`: Brokered-execution guarantees are well unit-tested in `tool_api`,
  but not asserted end to end across higher layers and shipped tool paths. The
  broker primitives are covered in isolation, while higher-layer workflow and
  app coverage does not prove those guarantees survive sequencing, persistence,
  and shipped tool integration surfaces. References:
  [test_execution.py](/mnt/c/Users/steph/.codex/worktrees/4bc0/llm-tools/tests/tool_api/test_execution.py:141),
  [workflow_api/executor.py](/mnt/c/Users/steph/.codex/worktrees/4bc0/llm-tools/src/llm_tools/workflow_api/executor.py:640),
  [test_runtime_integration.py](/mnt/c/Users/steph/.codex/worktrees/4bc0/llm-tools/tests/tools/test_runtime_integration.py:145)

### Recommended Regression Tests

- Add a research-session rendering regression that seeds secret-bearing approval
  and inspection payloads, then asserts approval state, replay, trace, and raw
  inspection views do not expose raw secret values.
- Add an end-to-end harness persistence regression that sets
  `protection_review.purge_requested=True` and verifies the original final
  response does not survive in stored snapshots, replay output, trace output,
  inspection payloads, or Streamlit research details.
- Expand the architecture guard to fail on additional runtime-bypass shapes,
  including direct use of tool implementation methods and internal runtime
  execution helpers outside `tool_api.runtime` and approved tests.
- Add workflow and harness integration coverage that asserts brokered-execution
  guarantees, including provenance continuity and runtime-mediated access,
  remain intact across sequencing and approval-resume flows.

### Residual Risk

Lower-layer import boundaries and core runtime normalization are covered
reasonably well. The main remaining risk is confidentiality drift in durable
inspection, replay, approval, and research-detail surfaces, where current tests
mostly confirm feature presence rather than end-to-end redaction and purge
invariants.

### Execution Notes

This assessment was a static review of code, tests, and documentation. `pytest`
could not be run in the current environment because `pytest` was not available.

## 2026-04-19: GitLab and Atlassian Built-In Tool Families

### Scope

- Reviewed code paths:
  - `src/llm_tools/tools/gitlab/`
  - `src/llm_tools/tools/atlassian/`
  - shared execution, secret, and policy surfaces in `src/llm_tools/tool_api/`
  - relevant tests in `tests/tools/`, plus policy and contract coverage relevant to these tool families
- Primary focus areas:
  - credential handling and secret scoping
  - request scoping and remote trust boundaries
  - pagination and data-exposure risks
  - unsafe assumptions about remote content
  - network error handling and retry behavior
  - approval and side-effect expectations
  - tool spec vs actual capability flags / required secrets / side effects
  - missing negative tests

### Tests

- Inspected:
  - `tests/tools/test_gitlab.py`
  - `tests/tools/test_atlassian.py`
  - `tests/tools/test_runtime_integration.py`
  - `tests/tool_api/test_policy.py`
  - `tests/architecture/test_tool_contracts.py`
- Attempted targeted execution:
  - `python3 -m pytest -q tests/tools/test_gitlab.py tests/tools/test_atlassian.py tests/tools/test_runtime_integration.py`
- Execution result:
  - blocked in the current environment because `pytest` is unavailable: `No module named pytest`

### Findings

- `High`: `read_confluence_content` writes remote attachment bytes and metadata into a temp cache but declares only `EXTERNAL_READ` and does not set `writes_internal_workspace_cache=True`.
  - Impact: policy that denies internal cache writes will not stop this path, and approval semantics can understate the actual side effects.
  - References:
    - `src/llm_tools/tools/atlassian/tools.py`
    - `src/llm_tools/tool_api/policy.py`
- `Medium`: pagination and output bounding are inconsistent across the tool families.
  - Impact: Jira, Bitbucket, and Confluence searches accept effectively unbounded positive limits, and MR/PR readers materialize full commit and change collections without local truncation metadata, increasing data exposure and availability risk on large remotes.
  - References:
    - `src/llm_tools/tools/atlassian/tools.py`
    - `src/llm_tools/tools/gitlab/tools.py`
- `Medium`: `read_jira_issue` exposes the full Jira `fields` map through `raw_fields` instead of an allowlisted subset.
  - Impact: the effective data exposure is broader than the tool description suggests and can include arbitrary custom fields or sensitive instance-specific metadata.
  - References:
    - `src/llm_tools/tools/atlassian/tools.py`
    - `tests/tools/test_atlassian.py`
- `Medium`: network resilience is weak because the remote tool specs do not set per-tool timeouts and the gateway builders rely on client-library defaults.
  - Impact: slow or stalled remotes can hang invocations longer than intended, and transient upstream failures are surfaced as generic execution failures without explicit retry policy.
  - References:
    - `src/llm_tools/tool_api/execution.py`
    - `src/llm_tools/tool_api/runtime.py`
    - `src/llm_tools/tools/gitlab/tools.py`
    - `src/llm_tools/tools/atlassian/tools.py`

### Recommended Remediation

- Split Confluence page reads from attachment reads, or mark attachment caching with `writes_internal_workspace_cache=True` so policy and approval behavior match the actual side effects.
- Add hard upper bounds and explicit truncation metadata for Jira, Bitbucket, and Confluence searches and for GitLab and Bitbucket MR/PR commit and change collections.
- Replace default `raw_fields` exposure in Jira reads with an allowlisted issue view, and make broader field access explicit and opt-in.
- Define per-tool network timeouts, decide which upstream failures should be retryable, and add negative tests for timeout, oversized result sets, and cache-write policy denial.

### Residual Risk

- Secret scoping in the runtime is generally sound because tools only receive the secrets listed in `required_secrets`, and matching provider gateways are built from that scoped view.
- Residual risk remains at the remote-boundary level: once a provider token is granted, there is no finer local allowlist for hosts, projects, repositories, spaces, or issue scopes.
- Remote HTML, excerpts, issue fields, and similar payloads are mostly treated as untrusted pass-through data rather than normalized content, so downstream consumers still need to treat these outputs as untrusted remote material.
