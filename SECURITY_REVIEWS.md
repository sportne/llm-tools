# SECURITY_REVIEWS.md

## Purpose

This document is the cumulative artifact for completed security assessments in
`llm-tools`.

Use it to record completed review outcomes in one place after the corresponding
backlog work in `SEC_TASKS.md` is done. `SEC_TASKS.md` remains the canonical
backlog; this file is the durable review log.

## Entry Conventions

Each completed review entry should include:

- review date
- component or scope reviewed
- code paths reviewed
- tests inspected and run
- findings summary with severity
- recommended remediation summary
- residual risk summary
- related `SEC_TASKS.md` phase or checklist reference

## Completed Reviews

### 2026-04-19: `harness_api` lifecycle and control-flow review

Related backlog: Phase 4 in `SEC_TASKS.md`

#### Scope

- `src/llm_tools/harness_api/models.py`
- `src/llm_tools/harness_api/executor.py`
- `src/llm_tools/harness_api/tasks.py`
- `src/llm_tools/harness_api/session.py`
- directly related helpers: `resume.py`, `replay.py`, `store.py`, `planning.py`,
  `context.py`, `verification.py`, `protection.py`
- relevant tests under `tests/harness_api/`

#### Code Paths Reviewed

- session, task, turn, and pending-approval state models
- executor run, resume, retry, approval, budget, and persistence-conflict paths
- task lifecycle transition helpers
- public session stop/resume/inspection behavior
- resume-time corruption detection and replay artifact construction

#### Tests

- inspected: `tests/harness_api/test_harness_models.py`, `test_resume.py`,
  `test_harness_executor.py`, `test_task_lifecycle.py`, `test_session_api.py`,
  `test_session_additional_coverage.py`, `test_planning.py`
- run: `~/.venvs/llm-tools/bin/python -m pytest -q tests/harness_api/test_harness_models.py tests/harness_api/test_resume.py tests/harness_api/test_harness_executor.py tests/harness_api/test_task_lifecycle.py tests/harness_api/test_session_api.py tests/harness_api/test_session_additional_coverage.py`
- result: `94 passed`
- environment note: `make install-dev` could not refresh dependencies because
  network access was unavailable, but the existing shared virtual environment
  already contained the required test packages

#### Findings Summary

- `Medium`: executor can persist `stop_reason=completed` when the planner
  returns no actionable tasks even though blocked non-terminal tasks remain
- `Medium`: non-approval turns have at-least-once semantics, so crash-before-save
  can replay non-idempotent side effects
- `Medium`: `max_tool_invocations` is enforced after turn execution, so one turn
  can overshoot the configured cap
- `Medium`: approval resume rehydrates from the full current process
  environment, which can widen privilege across turns
- `Low`: `NO_PROGRESS` is modeled and validated but not enforced by executor
  control flow

#### Recommended Remediation Summary

- prevent `completed` terminalization when non-terminal blocked work remains
- add durable checkpointing or idempotency safeguards for executed side effects
- enforce tool-invocation budget before or during dispatch
- constrain approval resume environment rehydration to a reviewed scope
- add executor-level no-progress detection and negative coverage for repeated
  non-progress loops

#### Residual Risk

Model validation and resume-time corruption checks are strong, but control-flow
integrity still depends on follow-up hardening in stop semantics, recovery
behavior, budget enforcement timing, and approval resume privilege boundaries.

### 2026-04-19: `harness_api` persistence, resume, replay, and summary review

#### Scope

- `src/llm_tools/harness_api/store.py`
- `src/llm_tools/harness_api/resume.py`
- `src/llm_tools/harness_api/replay.py`
- `src/llm_tools/harness_api/session.py`
- `src/llm_tools/harness_api/models.py`
- `src/llm_tools/harness_api/protection.py`
- workflow approval resume path used by harness resume:
  `src/llm_tools/workflow_api/executor.py`

#### Evidence

- reviewed code paths: `src/llm_tools/harness_api/store.py`,
  `src/llm_tools/harness_api/resume.py`, `src/llm_tools/harness_api/replay.py`,
  `src/llm_tools/harness_api/session.py`, `src/llm_tools/harness_api/models.py`,
  `src/llm_tools/harness_api/protection.py`,
  `src/llm_tools/workflow_api/executor.py`
- relevant tests inspected and run: `tests/harness_api/test_store.py`,
  `tests/harness_api/test_resume.py`, `tests/harness_api/test_replay_golden.py`,
  `tests/harness_api/test_protection_scrub.py`,
  `tests/harness_api/test_session_api.py`,
  `tests/harness_api/test_harness_executor.py`
- test execution result: focused harness test set passed via the repo venv:
  `66 passed`
- direct repro evidence:
  denied approval still allowed a later `write_file` invocation from the same
  persisted parsed response to execute
  tampered persisted approval payload could redirect approved execution to a
  different tool invocation

#### Findings

- `High`: denied, canceled, or expired approvals still execute later invocations
  from the same persisted parsed response.
  Preconditions: an approval-gated invocation is followed by additional
  invocations in the same parsed response, and resume is resolved with a
  non-approve outcome.
  Impact: approval denial does not fully stop execution of the persisted action
  sequence, so later side effects can still occur.
- `Medium`: persisted approval records are structurally validated but
  unauthenticated, so tampered stored state can redirect approved execution.
  Preconditions: an attacker or lower-trust principal can modify persisted
  harness session JSON before approval resume.
  Impact: operator approval may be applied to a modified persisted action rather
  than the originally reviewed action.
- `Medium`: stored summaries and traces are trusted for replay and inspection
  without consistency checks against canonical state.
  Preconditions: a persisted artifact can be modified independently of the
  canonical state payload.
  Impact: operator or audit views can be spoofed even when the main stored state
  is unchanged.
- `Medium`: persisted approvals, traces, metadata, logs, and summaries can retain
  sensitive content.
  Preconditions: secrets or sensitive data appear in tool arguments, policy
  metadata, logs, artifact names, or free-form summary fields.
  Impact: file-backed session storage and inspector surfaces may retain data
  beyond the scrubbed pending-approval base context.
- `Low`: corrupt file-backed state is not handled defensively and can cause load
  or list failures.
  Preconditions: malformed or partially written session artifacts exist on disk.
  Impact: corrupted persisted files become an availability issue for inspection
  and session enumeration.

#### Recommended Remediation

- prevent post-denial execution of trailing invocations after approval
  rejection, cancellation, or expiry
- add integrity binding for persisted approval payloads used during resume so
  the approved payload can be authenticated before execution
- validate or recompute persisted trace and summary artifacts before trusting
  them for replay or inspection
- reduce sensitive data retention in persisted approval, trace, and summary
  artifacts by default, including arguments, policy metadata, logs, artifact
  names, and free-form summary content
- handle malformed file-backed session artifacts without breaking load and list
  flows

#### Residual Risk

Persisted harness state should currently be treated as trusted storage. Until
approval replay stops on non-approved outcomes and persisted approval payloads
and observability artifacts gain integrity protection, a principal who can
modify stored session JSON can influence future execution after approval and can
falsify replay or summary views.

### 2026-04-19: architecture and security test coverage review

#### Reviewed Areas

- `tests/architecture/`
- security-relevant tests in `tests/tool_api/`, `tests/tools/`,
  `tests/llm_adapters/`, `tests/llm_providers/`, `tests/workflow_api/`,
  `tests/harness_api/`, and `tests/apps/`
- supporting architecture and security documentation, including
  `docs/design/spec.md`, `docs/design/harness_api.md`,
  `docs/usage/security-hardening.md`, `docs/usage/streamlit-assistant.md`, and
  `SEC_TASKS.md`

#### Findings

- `High`: no regression tests prove that research inspection, replay, approval,
  and raw inspection payload views redact secrets before rendering. The app
  exposes approval state and raw inspection payloads directly in the research
  detail surface, while the existing tests only assert that those views are
  present, not that sensitive fields are suppressed.
- `High`: no end-to-end regression proves that protection-triggered purge
  survives persistence, replay, inspection, and UI detail views. The purge path
  is implemented when harness state is saved, but existing coverage stops at
  scrub-helper unit tests and generic inspection/replay checks instead of
  validating the persisted and rendered outputs together.
- `Medium`: runtime-mediation architecture enforcement only detects direct
  `.invoke()` and `.ainvoke()` call sites. That leaves other bypass shapes
  unguarded, including direct use of implementation methods or internal runtime
  execution helpers outside the approved runtime boundary.
- `Medium`: brokered-execution guarantees are well unit-tested in `tool_api`,
  but not asserted end to end across higher layers and shipped tool paths.

#### Recommended Regression Tests

- add a research-session rendering regression that seeds secret-bearing approval
  and inspection payloads, then asserts approval state, replay, trace, and raw
  inspection views do not expose raw secret values
- add an end-to-end harness persistence regression that sets
  `protection_review.purge_requested=True` and verifies the original final
  response does not survive in stored snapshots, replay output, trace output,
  inspection payloads, or Streamlit research details
- expand the architecture guard to fail on additional runtime-bypass shapes,
  including direct use of tool implementation methods and internal runtime
  execution helpers outside `tool_api.runtime` and approved tests
- add workflow and harness integration coverage that asserts brokered-execution
  guarantees, including provenance continuity and runtime-mediated access,
  remain intact across sequencing and approval-resume flows

#### Residual Risk

Lower-layer import boundaries and core runtime normalization are covered
reasonably well. The main remaining risk is confidentiality drift in durable
inspection, replay, approval, and research-detail surfaces, where current tests
mostly confirm feature presence rather than end-to-end redaction and purge
invariants.

#### Execution Notes

This assessment was a static review of code, tests, and documentation. `pytest`
could not be run in the reviewed environment because `pytest` was unavailable.

### 2026-04-19: GitLab and Atlassian built-in tool families

#### Scope

- reviewed code paths:
  `src/llm_tools/tools/gitlab/`, `src/llm_tools/tools/atlassian/`, shared
  execution, secret, and policy surfaces in `src/llm_tools/tool_api/`, and
  relevant tests in `tests/tools/`, plus policy and contract coverage relevant
  to these tool families
- primary focus areas:
  credential handling and secret scoping, request scoping and remote trust
  boundaries, pagination and data-exposure risks, unsafe assumptions about
  remote content, network error handling and retry behavior, approval and
  side-effect expectations, tool spec vs actual capability flags, required
  secrets, side effects, and missing negative tests

#### Tests

- inspected: `tests/tools/test_gitlab.py`, `tests/tools/test_atlassian.py`,
  `tests/tools/test_runtime_integration.py`, `tests/tool_api/test_policy.py`,
  `tests/architecture/test_tool_contracts.py`
- attempted targeted execution:
  `python3 -m pytest -q tests/tools/test_gitlab.py tests/tools/test_atlassian.py tests/tools/test_runtime_integration.py`
- execution result: blocked in the reviewed environment because `pytest` was
  unavailable: `No module named pytest`

#### Findings

- `High`: `read_confluence_content` writes remote attachment bytes and metadata
  into a temp cache but declares only `EXTERNAL_READ` and does not set
  `writes_internal_workspace_cache=True`.
  Impact: policy that denies internal cache writes will not stop this path, and
  approval semantics can understate the actual side effects.
- `Medium`: pagination and output bounding are inconsistent across the tool
  families.
  Impact: Jira, Bitbucket, and Confluence searches accept effectively unbounded
  positive limits, and MR/PR readers materialize full commit and change
  collections without local truncation metadata, increasing data exposure and
  availability risk on large remotes.
- `Medium`: `read_jira_issue` exposes the full Jira `fields` map through
  `raw_fields` instead of an allowlisted subset.
  Impact: the effective data exposure is broader than the tool description
  suggests and can include arbitrary custom fields or sensitive
  instance-specific metadata.
- `Medium`: network resilience is weak because the remote tool specs do not set
  per-tool timeouts and the gateway builders rely on client-library defaults.
  Impact: slow or stalled remotes can hang invocations longer than intended,
  and transient upstream failures are surfaced as generic execution failures
  without explicit retry policy.

#### Recommended Remediation

- split Confluence page reads from attachment reads, or mark attachment caching
  with `writes_internal_workspace_cache=True` so policy and approval behavior
  match the actual side effects
- add hard upper bounds and explicit truncation metadata for Jira, Bitbucket,
  and Confluence searches and for GitLab and Bitbucket MR/PR commit and change
  collections
- replace default `raw_fields` exposure in Jira reads with an allowlisted issue
  view, and make broader field access explicit and opt-in
- define per-tool network timeouts, decide which upstream failures should be
  retryable, and add negative tests for timeout, oversized result sets, and
  cache-write policy denial

#### Residual Risk

- secret scoping in the runtime is generally sound because tools only receive
  the secrets listed in `required_secrets`, and matching provider gateways are
  built from that scoped view
- residual risk remains at the remote-boundary level: once a provider token is
  granted, there is no finer local allowlist for hosts, projects,
  repositories, spaces, or issue scopes
- remote HTML, excerpts, issue fields, and similar payloads are mostly treated
  as untrusted pass-through data rather than normalized content, so downstream
  consumers still need to treat these outputs as untrusted remote material
