# SEC_TASKS.md

## Purpose

This document tracks the repository-wide security assessment backlog for
`llm-tools`.

The goal is to evaluate the codebase module by module and as an integrated
system so repository security status can be stated with high confidence.

The root `SEC_TASKS.md` remains the canonical backlog for security review
execution and for tracking security hardening follow-up in this repository.

## Status conventions

- `[ ]` Not started
- `[~]` In progress
- `[x]` Done
- `[-]` Deferred

## Current state

- The codebase has explicit architectural layers enforced by tests.
- Central runtime mediation, policy, and redaction exist in `tool_api`, and
  recent hardening has landed in runtime safety and output-retention handling.
- Security-sensitive shipped surfaces include filesystem, network, and
  subprocess-capable tools, OpenAI-compatible provider integration, persisted
  harness sessions, and assistant and CLI entrypoints.
- `streamlit_chat` has been removed from shipped surfaces; remaining app review
  scope centers on `streamlit_assistant`, `harness_cli`, and shared app
  compatibility code that still affects those entrypoints.
- This document is the dedicated security backlog, and recent hardening work
  has already landed across `tool_api`, built-in tools, adapters and
  providers, `harness_api`, and assistant-facing app defaults.
- Initial Batch 1 review work is complete for `tool_api`, the filesystem, text,
  and git tool families, `llm_adapters`, `llm_providers`, and the packaging,
  example, config, and usage-doc surfaces.
- Recent harness persistence hardening makes non-approved approvals fail
  closed, treats persisted summary and trace artifacts as cache-only derived
  views, minimizes stored trace payloads, and isolates malformed file-backed
  session records as corruption outcomes.

## Phased backlog

### [x] Phase 0: Security review framework and evidence model

Outcome: establish how the deep dive will be run and what “high confidence”
means.

- [x] Define the review lane template used for every component: trust
  boundaries, inputs and outputs, privilege and approval model, secret
  handling, persistence and logging, external calls, abuse cases, failure
  modes, and test gaps.
- [x] Define evidence requirements for closing a component review: code paths
  inspected, tests reviewed and run, assumptions recorded, findings logged, and
  residual risk stated.
- [x] Define severity buckets and follow-up handling inside this file so
  findings can be tracked without creating a second backlog immediately.

Phase 0 decisions:

- Review lane template for every component review:
  1. Trust boundary and assets: identify what the component is trusted to do,
     what data or capabilities it can reach, and which lower and higher layers
     depend on it.
  2. Entry points and data flow: enumerate public APIs, model payloads, CLI or
     app entrypoints, persisted artifacts, environment variables, and any other
     untrusted inputs or outputs.
  3. Privilege and approval model: verify side-effect class declarations,
     capability flags, approval requirements, and any runtime mediation or
     bypass opportunities.
  4. Secret and sensitive data handling: inspect credentials, tokens, prompt
     content, local file contents, logs, artifacts, and replay or summary data
     for retention, redaction, and accidental disclosure risks.
  5. Persistence and observability: review what gets stored, logged, replayed,
     or surfaced to users; note tamper assumptions, retention defaults, and
     cross-session leakage risks.
  6. External interactions and dependency trust: inspect filesystem, network,
     subprocess, provider, parser, and third-party library usage with attention
     to scoping, argument construction, retries, and trust of returned data.
  7. Abuse and failure modes: evaluate denial of service, malformed input,
     oversized payload, state confusion, approval bypass, privilege escalation,
     unsafe recovery, and replay or resume edge cases.
  8. Test coverage and blind spots: map existing tests to the threat surface,
     identify missing negative tests, and note any assumptions not enforced by
     automated checks.
- Closure evidence required for marking a component review done:
  - reviewed code paths are listed by file or module area;
  - relevant existing tests are listed, with whether they were only inspected
    or actually run;
  - key security assumptions and non-goals are written down explicitly;
  - findings are recorded with severity, affected component, exploit
    preconditions, and recommended remediation;
  - residual risk is summarized even when no actionable findings are opened.
- Severity buckets for findings:
  - `Critical`: plausible confidentiality, integrity, or execution compromise
    with low friction, broad blast radius, or a broken core trust boundary.
  - `High`: material security weakness with realistic exploitation or strong
    impact, but narrower preconditions or containment than `Critical`.
  - `Medium`: meaningful weakness that requires more setup, has limited blast
    radius, or is partially mitigated by existing controls.
  - `Low`: defense-in-depth issue, hardening gap, or low-likelihood weakness
    that should be fixed but does not currently threaten a core boundary alone.
- Follow-up handling rules:
  - record findings under the affected phase once discovered, and keep the
    canonical remediation task under that same phase rather than creating a
    separate backlog file by default;
  - use Phase 7 for cross-cutting summaries, reopened issues, and final
    confidence statements, not as the only place findings exist;
  - add new remediation checklist items using the format `- [ ] <severity>:
    <component> - <issue>` so outstanding work stays sortable and scannable;
  - a phase can be marked done only when its review is complete, all accepted
    findings are logged, and any deferred fixes are called out explicitly.

### [~] Phase 1: Core execution substrate (`tool_api`)

Outcome: validate that the central execution and policy layer enforces the
intended safety invariants.

- [x] Review model validation boundaries in `models.py`, `tool.py`,
  `registry.py`, `runtime.py`, and `execution.py`.
- [x] Audit policy enforcement for allow, deny, and approval behavior,
  capability flags, secret requirements, and error normalization.
- [x] Audit redaction behavior for inputs, outputs, logs, artifacts, and
  metadata retention defaults.
- [x] Review runtime observability for leakage, inconsistent states, bypass
  paths, and denial-of-service exposure.
- [x] Identify missing negative tests for policy bypass, unsafe defaults,
  oversized payloads, and malformed tool results.
- [x] Landed hardening: tighten runtime safety, brokered-execution controls,
  policy enforcement coverage, secret-view isolation, and raw output-retention
  behavior touched by recent filesystem, text, and git security fixes.
- [x] Landed hardening: add `tool_api` execution-broker coverage in
  `tests/tool_api/test_execution.py` and extend adjacent runtime, policy, and
  mediation tests.

### [~] Phase 2: Tool implementations (`tools`)

Outcome: assess every built-in tool family against its real side effects and
trust boundaries.

- [x] `filesystem`: path traversal, symlink handling, workspace escape,
  overwrite semantics, size and resource limits, and content extraction safety.
- [x] `text`: propagation of filesystem constraints, search and read
  amplification risks, parsing edge cases, and content-based denial of service.
- [x] `git`: subprocess construction, cwd and root control, argument injection
  resistance, output handling, and approval expectations.
- [x] `gitlab` and `atlassian`: credential handling, request scoping,
  pagination and data exposure, network error handling, and unsafe remote
  content assumptions.
- [ ] Cross-check every built-in tool spec against actual side effects,
  capability flags, and required secrets.
- [x] Landed hardening: filesystem and text path, cache, and runtime safety
  updates are merged.
- [x] Landed hardening: git tool subprocess, output-retention, and integration
  safety updates are merged.
- [x] Review log: GitLab and Atlassian tool-family security review documented
  in `SECURITY_REVIEWS.md` on 2026-04-19.

### [~] Phase 3: Model mediation path (`llm_adapters`, `llm_providers`, `workflow_api`)

Outcome: confirm model output cannot bypass the typed execution boundary or
silently widen privileges.

- [x] Review adapter parsing and normalization for malformed or adversarial
  model payloads.
- [x] Review provider request and response handling for secret leakage, unsafe
  retries, schema mismatch behavior, and endpoint trust assumptions.
- [x] Review `workflow_api` execution sequencing, protection hooks, and
  one-turn control flow for approval, replay, and partial-failure safety,
  including persisted approval/resume and replay touchpoints in
  `harness_api`.
- [ ] Identify attack paths where model-controlled content could trigger
  unexpected tool execution, unbounded work, or sensitive data disclosure.
- [x] Landed hardening: adapter parsing and OpenAI-compatible provider fallback
  coverage updates are merged.
- [x] Completed 2026-04-19: `workflow_api` security posture review executed
  against the workflow, runtime, adapter, and harness approval/replay paths;
  findings and residual risk captured in
  `SECURITY_REVIEWS.md`.

### [x] Phase 4: Durable orchestration (`harness_api`)

Outcome: assess multi-turn session orchestration, persistence, replay, and
verification as a security-critical control plane.

- [x] Review session, task, and turn lifecycle models for state confusion,
  replay inconsistency, and approval durability issues.
- [x] Audit persisted storage, resume, replay, and summaries for secret
  retention, tamper exposure, and unsafe trust in stored artifacts.
- Completed review artifact: `SECURITY_REVIEWS.md` (includes lifecycle and
  persistence review summaries).
- [x] Review planning, context construction, verification, and protection
  scrubbing for privilege escalation or leakage across turns.
- [x] Assess stop conditions, no-progress handling, retries, and recovery
  logic for abuse, infinite work, or unsafe resumptions.
- [x] Landed hardening: harness approval persistence and session-safety updates
  are merged.
- [x] High: harness_api/workflow_api - make approval denial and timeout
  fail-closed so later invocations from the same parsed response do not run.
- [-] Medium: harness_api - add integrity binding for persisted approval
  payloads used during resume.
- [x] Medium: harness_api - treat persisted trace and summary artifacts as
  cache-only derived views and rebuild or ignore them when they are missing,
  stale, or inconsistent.
- [x] Medium: harness_api - minimize persisted trace payloads and adjacent
  derived observability artifacts so raw request arguments are not retained by
  default.
- [ ] Medium: harness_api - prevent `completed` terminalization when the planner
  returns no actionable tasks but blocked or otherwise non-terminal tasks still
  exist.
- [ ] Medium: harness_api - add durable checkpointing or idempotency safeguards
  so side-effectful work is not replayed after crash-before-save recovery.
- [ ] Medium: harness_api - enforce `max_tool_invocations` before or during
  dispatch so a single turn cannot overshoot the configured tool budget.
- [-] Medium: harness_api - constrain approval resume environment rehydration to
  a reviewed allowlist or stable approved snapshot instead of the full current
  process environment.
- [x] Low: harness_api - isolate malformed file-backed session artifacts into
  typed corruption handling so list/load flows skip or surface them cleanly.
- [ ] Low: harness_api - implement executor-level no-progress detection and add
  negative tests for repeated non-progress loops and blocked-only sessions.

### [~] Phase 5: User-facing surfaces (`apps`)

Outcome: validate that interactive clients and CLI entrypoints preserve
lower-layer security guarantees.

- [ ] Review shared app config, runtime, and prompt helpers for dangerous
  defaults, secret persistence, approval toggles, and tool enablement drift.
- [ ] Review `streamlit_assistant` as the primary supported client for session
  isolation, persisted research controls, and prompt and config trust
  boundaries.
- [ ] Review remaining app compatibility surfaces and shared repository-chat
  helpers that still influence assistant or CLI behavior after
  `streamlit_chat` removal.
- [ ] Review `harness_cli` and related app entrypoints for unsafe argument
  handling, persistence assumptions, and bypass of normal protections.
- [x] Landed hardening: assistant defaults, config examples, and related app
  entrypoint protections are merged.

### [~] Phase 6: System-of-systems and supply chain review

Outcome: evaluate the repository as an integrated deliverable rather than only
as isolated modules.

Review artifact:
- `SECURITY_REVIEWS.md` is the cumulative record for completed security review
  reports.

- [x] Review `pyproject.toml`, optional dependencies, console scripts, and
  packaging metadata for unnecessary exposure and dependency risk
  concentration.
- [x] Review examples, assistant configs, and usage docs for insecure guidance,
  secret-handling mistakes, and unsafe copy-paste defaults.
- [x] Cross-check architecture tests and existing security-relevant tests
  against the actual threat model to identify blind spots.
- [ ] Assess cross-layer invariants: lower layers must not import higher
  layers, tools must stay runtime-mediated, approval semantics must stay
  consistent, and redaction and protection expectations must remain aligned end
  to end.

### [ ] Phase 7: Findings triage and confidence statement

Outcome: convert review output into an actionable security status for the
project.

- [x] Record findings in this file or a clearly linked companion artifact with
  severity, affected components, exploit preconditions, and recommended fixes.
- Companion artifact in use: `SECURITY_REVIEWS.md`.
- [x] Track remediation tasks separately within this backlog under the affected
  phase or in a dedicated follow-up section.
- [x] Mark explicitly deferred findings with `[-]` so accepted deferrals are
  visible instead of looking like active remediation work.
- [ ] Re-run relevant tests and add targeted regression tests for each
  confirmed issue.
- [ ] Produce a final project-wide confidence summary: reviewed areas,
  unresolved risks, deferred work, and what would still block a stronger
  security claim.
