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

- [ ] Review model validation boundaries in `models.py`, `tool.py`,
  `registry.py`, `runtime.py`, and `execution.py`.
- [ ] Audit policy enforcement for allow, deny, and approval behavior,
  capability flags, secret requirements, and error normalization.
- [ ] Audit redaction behavior for inputs, outputs, logs, artifacts, and
  metadata retention defaults.
- [ ] Review runtime observability for leakage, inconsistent states, bypass
  paths, and denial-of-service exposure.
- [ ] Identify missing negative tests for policy bypass, unsafe defaults,
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

- [ ] `filesystem`: path traversal, symlink handling, workspace escape,
  overwrite semantics, size and resource limits, and content extraction safety.
- [ ] `text`: propagation of filesystem constraints, search and read
  amplification risks, parsing edge cases, and content-based denial of service.
- [ ] `git`: subprocess construction, cwd and root control, argument injection
  resistance, output handling, and approval expectations.
- [ ] `gitlab` and `atlassian`: credential handling, request scoping,
  pagination and data exposure, network error handling, and unsafe remote
  content assumptions.
- [ ] Cross-check every built-in tool spec against actual side effects,
  capability flags, and required secrets.
- [x] Landed hardening: filesystem and text path, cache, and runtime safety
  updates are merged.
- [x] Landed hardening: git tool subprocess, output-retention, and integration
  safety updates are merged.

### [~] Phase 3: Model mediation path (`llm_adapters`, `llm_providers`, `workflow_api`)

Outcome: confirm model output cannot bypass the typed execution boundary or
silently widen privileges.

- [ ] Review adapter parsing and normalization for malformed or adversarial
  model payloads.
- [ ] Review provider request and response handling for secret leakage, unsafe
  retries, schema mismatch behavior, and endpoint trust assumptions.
- [ ] Review `workflow_api` execution sequencing, protection hooks, and
  one-turn control flow for approval, replay, and partial-failure safety.
- [ ] Identify attack paths where model-controlled content could trigger
  unexpected tool execution, unbounded work, or sensitive data disclosure.
- [x] Landed hardening: adapter parsing and OpenAI-compatible provider fallback
  coverage updates are merged.

### [~] Phase 4: Durable orchestration (`harness_api`)

Outcome: assess multi-turn session orchestration, persistence, replay, and
verification as a security-critical control plane.

- [ ] Review session, task, and turn lifecycle models for state confusion,
  replay inconsistency, and approval durability issues.
- [x] Audit persisted storage, resume, replay, and summaries for secret
  retention, tamper exposure, and unsafe trust in stored artifacts.
- Completed review: `harness_api` persistence, resume, replay, and summary
  behavior is documented in `SECURITY_REVIEWS.md` (2026-04-19 entry).
- [ ] Review planning, context construction, verification, and protection
  scrubbing for privilege escalation or leakage across turns.
- [ ] Assess stop conditions, no-progress handling, retries, and recovery
  logic for abuse, infinite work, or unsafe resumptions.
- [x] Landed hardening: harness approval persistence and session-safety updates
  are merged.
- [ ] High: harness_api/workflow_api - prevent post-denial execution of
  trailing invocations after approval rejection.
- [ ] Medium: harness_api - add integrity binding for persisted approval
  payloads used during resume.
- [ ] Medium: harness_api - validate or recompute persisted trace and summary
  artifacts before trusting them for replay or inspection.
- [ ] Medium: harness_api - reduce sensitive data retention in persisted
  approval, trace, and summary artifacts.
- [ ] Low: harness_api - handle malformed file-backed session artifacts without
  breaking list/load flows.

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

### [ ] Phase 6: System-of-systems and supply chain review

Outcome: evaluate the repository as an integrated deliverable rather than only
as isolated modules.

- [ ] Review `pyproject.toml`, optional dependencies, console scripts, and
  packaging metadata for unnecessary exposure and dependency risk
  concentration.
- [ ] Review examples, assistant configs, and usage docs for insecure guidance,
  secret-handling mistakes, and unsafe copy-paste defaults.
- [ ] Cross-check architecture tests and existing security-relevant tests
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
- [ ] Track remediation tasks separately within this backlog under the affected
  phase or in a dedicated follow-up section.
- [ ] Re-run relevant tests and add targeted regression tests for each
  confirmed issue.
- [ ] Produce a final project-wide confidence summary: reviewed areas,
  unresolved risks, deferred work, and what would still block a stronger
  security claim.
