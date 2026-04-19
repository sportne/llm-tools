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

Consolidation note:

- Standalone 2026-04-19 implementation review documents were folded into this
  file so findings live in one cumulative review log.
- Redundant findings were collapsed so each review entry owns the findings most
  specific to its layer.

### 2026-04-19: `workflow_api` sequencing, approvals, protection, and replay review

Related backlog: Phase 3 in `SEC_TASKS.md`

#### Scope

- `src/llm_tools/workflow_api/executor.py`
- `src/llm_tools/workflow_api/chat_session.py`
- `src/llm_tools/workflow_api/protection.py`
- `src/llm_tools/workflow_api/models.py`
- related mediation and persistence touchpoints in `llm_adapters`, `tool_api`,
  and `harness_api`
- relevant tests under `tests/workflow_api/`, `tests/tool_api/`,
  `tests/architecture/`, and selected harness tests

#### Code Paths Reviewed

- workflow execution sequencing and runtime mediation boundaries
- approval request, resume, deny, timeout, and cancel behavior
- prompt-protection pending state, correction capture, and inspector payloads
- replay and trace construction paths that expose workflow request data
- adapter/runtime/harness touchpoints that influence workflow approval and
  observability behavior

#### Tests

- inspected: `tests/workflow_api/test_executor.py`,
  `test_executor_additional.py`, `test_chat_session.py`,
  `test_protection.py`, `tests/tool_api/test_runtime.py`,
  `test_policy.py`, `tests/architecture/test_runtime_mediation.py`,
  `test_no_direct_tool_invocation.py`, `tests/harness_api/test_harness_executor.py`,
  `test_replay_golden.py`, `test_protection_scrub.py`
- run: `"$HOME/.venvs/llm-tools/bin/python" -m pytest -q tests/workflow_api/test_executor.py tests/workflow_api/test_executor_additional.py tests/workflow_api/test_chat_session.py tests/workflow_api/test_protection.py tests/tool_api/test_runtime.py tests/tool_api/test_policy.py tests/architecture/test_runtime_mediation.py tests/architecture/test_no_direct_tool_invocation.py tests/harness_api/test_harness_executor.py tests/harness_api/test_replay_golden.py tests/harness_api/test_protection_scrub.py`
- result: `137 passed`

#### Findings Summary

- `High`: approval denial and timeout are not fail-closed in raw
  `WorkflowExecutor`, so later model-selected invocations can still run after a
  non-approved outcome.
- `High`: persisted approval resume is not cryptographically or structurally
  bound to the originally approved request, allowing tampered approval records
  to redirect execution.
- `Medium`: prompt protection retains raw blocked prompt material in session
  state, correction storage, and inspector-adjacent payloads.
- `Medium`: replay and trace artifacts preserve raw request arguments and
  approval metadata outside the runtime redaction path.

#### Recommended Remediation Summary

- make non-approved approval outcomes terminal by default in `WorkflowExecutor`
- bind approval resume to a stable digest over the approved invocation and
  approval-relevant context
- minimize or redact blocked prompt snapshots, correction metadata, and
  inspector payloads
- build replay and trace request views from redacted execution data rather than
  raw request payloads

#### Residual Risk Summary

Runtime mediation remains the strongest lower-layer control, but the main
remaining `workflow_api` risks are approval fail-open behavior, approval-resume
integrity, and observability paths that preserve more raw model or tool data
than the runtime result surface.

#### Status Update (2026-04-19 follow-up)

- `Addressed`: non-approved approval outcomes now fail closed in
  `WorkflowExecutor`, including sync and async persisted-resume paths.
- `Addressed`: workflow-adjacent replay and trace views no longer persist raw
  request arguments by default; the harness rebuild path uses redacted argument
  projections and canonical state, and follow-up regressions cover approval
  history preservation after resume.
- `Deferred`: approval-record binding remains open because the repository still
  lacks a trust anchor and stable signed-digest format for persisted approval
  records.
- `Deferred`: prompt-protection retention minimization remains open because the
  correction-state and inspector-payload contract is not yet settled.

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
- public session stop, resume, and inspection behavior
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
  returns no actionable tasks even though blocked non-terminal tasks remain.
- `Medium`: non-approval turns have at-least-once semantics, so crash-before-save
  can replay non-idempotent side effects.
- `Medium`: `max_tool_invocations` is enforced after turn execution, so one turn
  can overshoot the configured cap.
- `Medium`: approval resume rehydrates from the full current process
  environment, which can widen privilege across turns.
- `Low`: `NO_PROGRESS` is modeled and validated but not enforced by executor
  control flow.

#### Recommended Remediation Summary

- prevent `completed` terminalization when non-terminal blocked work remains
- add durable checkpointing or idempotency safeguards for executed side effects
- enforce tool-invocation budget before or during dispatch
- constrain approval resume environment rehydration to a reviewed scope
- add executor-level no-progress detection and negative coverage for repeated
  non-progress loops

#### Residual Risk Summary

Model validation and resume-time corruption checks are strong, but control-flow
integrity still depends on follow-up hardening in stop semantics, recovery
behavior, budget enforcement timing, and approval resume privilege boundaries.

#### Status Update (2026-04-19 follow-up)

- `Open`: `stop_reason=completed` can still be persisted while blocked or other
  non-terminal tasks remain. This needs planner/applier contract changes rather
  than a replay or persistence-only patch.
- `Open`: non-approval turns still have at-least-once semantics across
  crash-before-save recovery. Addressing this requires durable checkpointing or
  explicit idempotency controls for side effects.
- `Open`: `max_tool_invocations` is still enforced after a turn completes, so a
  single turn can overshoot the configured limit.
- `Deferred`: approval-resume environment narrowing is still pending because the
  repository does not yet define a reviewed allowlist or stable approved-env
  snapshot format.
- `Open`: executor-level no-progress detection is still not implemented.

### 2026-04-19: `harness_api` persistence, resume, replay, and summary review

Related backlog: Phase 4 in `SEC_TASKS.md`

#### Scope

- `src/llm_tools/harness_api/store.py`
- `src/llm_tools/harness_api/resume.py`
- `src/llm_tools/harness_api/replay.py`
- `src/llm_tools/harness_api/session.py`
- `src/llm_tools/harness_api/models.py`
- `src/llm_tools/harness_api/protection.py`
- supporting persistence and resume touchpoints in `harness_api`,
  `workflow_api`, and `tool_api`

#### Evidence

- inspected and run: `tests/harness_api/test_store.py`, `test_resume.py`,
  `test_replay_golden.py`, `test_protection_scrub.py`, `test_session_api.py`,
  `test_harness_executor.py`
- run: `$HOME/.venvs/llm-tools/bin/python -m pytest -q tests/harness_api/test_store.py tests/harness_api/test_resume.py tests/harness_api/test_replay_golden.py tests/harness_api/test_protection_scrub.py tests/harness_api/test_session_api.py tests/harness_api/test_harness_executor.py`
- result: `66 passed`
- note: approval fail-open sequencing and approval-record binding are tracked in
  the `workflow_api` review above to avoid duplicating the same cross-layer
  findings here

#### Findings Summary

- `High`: persisted harness state and derived artifacts can retain sensitive
  data, including tool arguments, outputs, metadata, traces, and summaries.
- `Medium`: stored summaries are trusted without recomputation or consistency
  checks against canonical state.
- `Medium`: replay trusts stored trace artifacts as authoritative when present,
  rather than validating against canonical turns.
- `Low`: corrupted on-disk session files are not isolated into typed corruption
  handling and can surface as raw load or list failures.

#### Recommended Remediation Summary

- add a harness-level persistence scrub for completed turns and derived
  artifacts, defaulting to minimization or hashing for sensitive fields
- treat summary and trace artifacts as cache rather than authority by
  recomputing them or validating them against canonical state
- harden file-store writes and reads with atomic save behavior and typed
  corruption outcomes
- add negative regression tests for tampered summaries, tampered traces,
  corrupted JSON session files, and secret-bearing persisted artifacts

#### Residual Risk Summary

Stored harness sessions should currently be treated as confidentiality-sensitive
and attacker-modifiable. The largest remaining persistence risks are data
retention and operator trust in derived artifacts rather than baseline model
validation.

#### Status Update (2026-04-19 follow-up)

- `Addressed`: stored summaries and traces are now treated as cache-only
  artifacts and rebuilt from canonical `snapshot.state` during inspect, list,
  and replay flows.
- `Addressed`: persisted trace payloads no longer retain raw request arguments
  by default; they use minimized redacted argument projections instead.
- `Addressed`: file-backed session storage now uses atomic save behavior and
  typed corruption handling, and corrupt files are skipped during listing.
- `Addressed`: negative regressions now cover tampered summaries, tampered
  traces, corrupt session files, and purge-propagation rendering paths.
- `Addressed`: a follow-up fix restored approval-request audit history,
  preserved per-turn verification snapshots in canonical trace rebuilds, and
  replaced turn-level raw approval payload retention with minimized approval
  audit metadata so normalized turn history does not rewrite history or retain
  blocked invocation arguments after resume.

### 2026-04-19: architecture and security test coverage review

Related backlog: Phase 6 in `SEC_TASKS.md`

#### Reviewed Areas

- `tests/architecture/`
- security-relevant tests in `tests/tool_api/`, `tests/tools/`,
  `tests/llm_adapters/`, `tests/llm_providers/`, `tests/workflow_api/`,
  `tests/harness_api/`, and `tests/apps/`
- supporting architecture and security documentation, including
  `docs/design/spec.md`, `docs/design/harness_api.md`,
  `docs/usage/security-hardening.md`, `docs/usage/streamlit-assistant.md`, and
  `SEC_TASKS.md`

#### Findings Summary

- `High`: no regression tests prove that research inspection, replay, approval,
  and raw inspection payload views redact secrets before rendering.
- `High`: no end-to-end regression proves that protection-triggered purge
  survives persistence, replay, inspection, and UI detail views.
- `Medium`: runtime-mediation architecture enforcement only detects direct
  `.invoke()` and `.ainvoke()` call sites, leaving other bypass shapes
  unguarded.
- `Medium`: brokered-execution guarantees are well unit-tested in `tool_api`,
  but not asserted end to end across higher layers and shipped tool paths.

#### Recommended Remediation Summary

- add rendering regressions for secret-bearing approval and inspection payloads
- add an end-to-end persistence regression for purge propagation across stored
  and rendered outputs
- expand architecture guards to catch additional runtime-bypass shapes
- add workflow and harness integration coverage for brokered-execution
  guarantees, including provenance continuity and runtime-mediated access

#### Residual Risk Summary

Lower-layer import boundaries and core runtime normalization are covered
reasonably well. The main remaining risk is confidentiality drift in durable
inspection, replay, approval, and research-detail surfaces, where current tests
mostly confirm feature presence rather than end-to-end redaction and purge
invariants.

#### Status Update (2026-04-19 follow-up)

- `Addressed`: research inspection, replay, approval, and raw inspection payload
  rendering now have regressions that assert scrubbed secret content does not
  reappear in user-facing views.
- `Addressed`: end-to-end purge propagation is now covered across persistence,
  replay, inspection, and Streamlit research detail rendering.
- `Addressed`: architecture guards now detect additional runtime-bypass shapes,
  including indirect `.invoke()` and `.ainvoke()` access patterns.
- `Open`: higher-layer end-to-end brokered-execution coverage is still
  incomplete; lower-layer guarantees remain well unit-tested, but the broader
  provenance-continuity integration checks were not completed in this pass.

#### Execution Notes

This assessment was a static review of code, tests, and documentation. `pytest`
could not be run in the reviewed environment because `pytest` was unavailable.

### 2026-04-19: GitLab and Atlassian built-in tool families

Related backlog: Phase 2 in `SEC_TASKS.md`

#### Scope

- reviewed code paths: `src/llm_tools/tools/gitlab/`,
  `src/llm_tools/tools/atlassian/`, shared execution, secret, and policy
  surfaces in `src/llm_tools/tool_api/`, and relevant tests in `tests/tools/`,
  plus policy and contract coverage relevant to these tool families
- primary focus areas: credential handling and secret scoping, request scoping
  and remote trust boundaries, pagination and data-exposure risks, unsafe
  assumptions about remote content, network error handling and retry behavior,
  approval and side-effect expectations, tool spec vs actual capability flags,
  required secrets, side effects, and missing negative tests

#### Tests

- inspected: `tests/tools/test_gitlab.py`, `tests/tools/test_atlassian.py`,
  `tests/tools/test_runtime_integration.py`, `tests/tool_api/test_policy.py`,
  `tests/architecture/test_tool_contracts.py`
- attempted targeted execution: `python3 -m pytest -q tests/tools/test_gitlab.py tests/tools/test_atlassian.py tests/tools/test_runtime_integration.py`
- execution result: blocked in the reviewed environment because `pytest` was
  unavailable: `No module named pytest`

#### Findings Summary

- `High`: `read_confluence_content` writes remote attachment bytes and metadata
  into a temp cache but declares only `EXTERNAL_READ` and does not set
  `writes_internal_workspace_cache=True`.
- `Medium`: pagination and output bounding are inconsistent across the tool
  families.
- `Medium`: `read_jira_issue` exposes the full Jira `fields` map through
  `raw_fields` instead of an allowlisted subset.
- `Medium`: network resilience is weak because the remote tool specs do not set
  per-tool timeouts and the gateway builders rely on client-library defaults.

#### Recommended Remediation Summary

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

#### Residual Risk Summary

- secret scoping in the runtime is generally sound because tools only receive
  the secrets listed in `required_secrets`, and matching provider gateways are
  built from that scoped view
- residual risk remains at the remote-boundary level: once a provider token is
  granted, there is no finer local allowlist for hosts, projects,
  repositories, spaces, or issue scopes
- remote HTML, excerpts, issue fields, and similar payloads are mostly treated
  as untrusted pass-through data rather than normalized content, so downstream
  consumers still need to treat these outputs as untrusted remote material

#### Status Update (2026-04-19 follow-up)

- `Open`: this tool-family review was not remediated in the harness/workflow
  hardening pass.
- `Open`: the Confluence attachment-cache side-effect declaration mismatch still
  needs either a spec change or a behavioral split between page reads and
  attachment reads.
- `Open`: pagination and output-bound inconsistencies still require per-tool
  response-shaping changes and matching negative coverage.
- `Open`: Jira `raw_fields` exposure still needs an allowlisted default view and
  explicit opt-in for broader field access.
- `Open`: per-tool network timeout and retry policy decisions still need to be
  implemented at the remote tool layer.
- `Why not addressed here`: these fixes change shipped remote tool contracts,
  default payload shapes, and connector behavior, so they need a dedicated tool
  remediation pass rather than a review-log-only update.
