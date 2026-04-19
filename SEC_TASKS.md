# SEC_TASKS.md

## Purpose

This document tracks the repository-wide security assessment backlog for
`llm-tools`.

The goal is to evaluate the codebase module by module and as an integrated
system so repository security status can be stated with high confidence.

The root `SEC_TASKS.md` remains the canonical backlog for security review
execution and post-review hardening follow-up in this repository.

## Status conventions

- `[ ]` Not started
- `[~]` In progress
- `[x]` Done
- `[-]` Deferred

## Current state

- The codebase has explicit architectural layers enforced by tests.
- Central runtime mediation, policy, and redaction exist in `tool_api`.
- Security-sensitive shipped surfaces include filesystem, network, and
  subprocess-capable tools, OpenAI-compatible provider integration, persisted
  harness sessions, and interactive app and CLI entrypoints.
- `streamlit_chat` is deprecated but still shipped, so it remains in scope
  until removed.
- The repository already includes security-relevant tests, but it does not yet
  have a dedicated security backlog document.

## Phased backlog

### [ ] Phase 0: Security review framework and evidence model

Outcome: establish how the deep dive will be run and what “high confidence”
means.

- [ ] Define the review lane template used for every component: trust
  boundaries, inputs and outputs, privilege and approval model, secret
  handling, persistence and logging, external calls, abuse cases, failure
  modes, and test gaps.
- [ ] Define evidence requirements for closing a component review: code paths
  inspected, tests reviewed and run, assumptions recorded, findings logged, and
  residual risk stated.
- [ ] Define severity buckets and follow-up handling inside this file so
  findings can be tracked without creating a second backlog immediately.

### [ ] Phase 1: Core execution substrate (`tool_api`)

Outcome: validate that the central execution and policy layer enforces the
intended safety invariants.

- [ ] Review model validation boundaries in `models.py`, `tool.py`,
  `registry.py`, and `runtime.py`.
- [ ] Audit policy enforcement for allow, deny, and approval behavior,
  capability flags, secret requirements, and error normalization.
- [ ] Audit redaction behavior for inputs, outputs, logs, artifacts, and
  metadata retention defaults.
- [ ] Review runtime observability for leakage, inconsistent states, bypass
  paths, and denial-of-service exposure.
- [ ] Identify missing negative tests for policy bypass, unsafe defaults,
  oversized payloads, and malformed tool results.

### [ ] Phase 2: Tool implementations (`tools`)

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

### [ ] Phase 3: Model mediation path (`llm_adapters`, `llm_providers`, `workflow_api`)

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

### [ ] Phase 4: Durable orchestration (`harness_api`)

Outcome: assess multi-turn session orchestration, persistence, replay, and
verification as a security-critical control plane.

- [ ] Review session, task, and turn lifecycle models for state confusion,
  replay inconsistency, and approval durability issues.
- [ ] Audit persisted storage, resume, replay, and summaries for secret
  retention, tamper exposure, and unsafe trust in stored artifacts.
- [ ] Review planning, context construction, verification, and protection
  scrubbing for privilege escalation or leakage across turns.
- [ ] Assess stop conditions, no-progress handling, retries, and recovery
  logic for abuse, infinite work, or unsafe resumptions.

### [ ] Phase 5: User-facing surfaces (`apps`)

Outcome: validate that interactive clients and CLI entrypoints preserve
lower-layer security guarantees.

- [ ] Review shared app config, runtime, and prompt helpers for dangerous
  defaults, secret persistence, approval toggles, and tool enablement drift.
- [ ] Review `streamlit_assistant` as the primary supported client for session
  isolation, persisted research controls, and prompt and config trust
  boundaries.
- [ ] Review deprecated `streamlit_chat` because it still ships: session
  persistence, export paths, approval UX, and deprecated-surface risk.
- [ ] Review `harness_cli` and related app entrypoints for unsafe argument
  handling, persistence assumptions, and bypass of normal protections.

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

- [ ] Record findings in this file or a clearly linked companion artifact with
  severity, affected components, exploit preconditions, and recommended fixes.
- [ ] Track remediation tasks separately within this backlog under the affected
  phase or in a dedicated follow-up section.
- [ ] Re-run relevant tests and add targeted regression tests for each
  confirmed issue.
- [ ] Produce a final project-wide confidence summary: reviewed areas,
  unresolved risks, deferred work, and what would still block a stronger
  security claim.
