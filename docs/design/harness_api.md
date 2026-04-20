# harness_api

## Purpose

`harness_api` is the repository's durable orchestration layer above the
one-turn workflow primitive in `workflow_api`.

It exists to support longer-running research and workflow execution that must
be:

- persisted
- resumable
- inspectable
- replayable
- verifiable

This layer is part of the supported repository direction for prescribed
workflows and future definable-agent capabilities.

## Truth resolution

This document describes the implemented `harness_api` surface. If it diverges
from code or tests, update the docs to match the actual public API and current
behavior.

## What `harness_api` owns

`harness_api` owns session-level concerns that should not be pushed into the
base tool runtime or into the one-turn workflow primitive.

That includes:

- persisted `HarnessState`, `HarnessTurn`, and related durable models
- state stores and schema-version handling
- resume classification for stored sessions
- replay traces and operator summaries
- task lifecycle state transitions
- deterministic planning and replanning triggers
- verification contracts and no-progress signals
- a public session service for create/run/resume/inspect/list/stop flows

## Public surface

The supported public `harness_api` surface is centered on:

- durable models in `models.py`
- replay, resume, store, task, and verification contracts
- `HarnessExecutor`
- `HarnessSessionService`
- request/result models for session operations
- default driver, applier, and provider protocols used by the built-in service
  path

- `TaskLifecycleStatus`: task lifecycle enum for pending, in-progress, blocked,
  completed, failed, canceled, and superseded work.
- `TaskOrigin`: durable provenance enum distinguishing user-requested tasks
  from derived subtasks.
- `VerificationStatus`: task verification enum for not-run, passed, failed, and
  inconclusive checks.
- `HarnessStopReason`: canonical session stop reasons such as completed,
  budget-exhausted, verification-failed, no-progress, canceled, and error.
- `TurnDecisionAction`: post-turn action enum for continue, select-tasks, and
  stop decisions.
- `BudgetPolicy`: configured session budget limits. It holds only durable limit
  settings and requires at least one positive configured bound.
- `VerificationExpectation`: lightweight durable expectations attached to tasks
  so verification requirements survive replay and resume.
- `VerificationOutcome`: persisted verification result attached to a task,
  including status, checked timestamp, summary, and evidence references.
- `TaskRecord`: canonical durable unit-of-work record, with task identity,
  title, intent, origin, lifecycle status, parent/dependency links,
  verification expectations, verification state, artifact refs, supersession
  metadata, and durable status timestamps.
- `TurnDecision`: harness-owned decision emitted after a turn completes,
  including action, selected task ids, optional stop reason, and summary text.
- `PendingApprovalRecord`: durable approval-resume record that preserves the
  blocked `ApprovalRequest`, the parsed model response, the base `ToolContext`,
  and the pending invocation index needed to continue the interrupted turn.
- `HarnessTurn`: one persisted harness turn, including the turn index, durable
  timestamps, optional `WorkflowTurnResult`, optional `TurnDecision`, any
  no-progress signals detected for the turn, turn-local verification snapshots,
  and minimized approval-audit metadata for resumed approval history.
- `HarnessSession`: session-level metadata including session id, root task id,
  budget policy, current turn index, start time, and terminal stop metadata.
- `HarnessState`: top-level persisted envelope containing `schema_version`,
  `HarnessSession`, all `TaskRecord` entries, all persisted verification
  evidence records, all `HarnessTurn` entries, and any durable pending
  approvals required for resume.

The live public session API is service-based, not free functions. The key entry
point is `HarnessSessionService`, which exposes typed create, run, resume,
inspect, list, and stop operations.

The package keeps its public package boundaries stable even though the
implementation is now split into narrower modules:

- `executor.py` is a thin public facade over execution internals
- `session.py` is a thin public facade over session-service and default-driver
  internals
- lower-level implementation details live in modules such as
  `executor_loop.py`, `executor_approvals.py`, `session_service.py`, and
  `defaults.py`

## Canonical models

`harness_api.models` defines the canonical persisted contracts for durable
session execution, including:

- `BudgetPolicy`
- `HarnessSession`
- `HarnessState`
- `HarnessTurn`
- `PendingApprovalRecord`
- `TurnApprovalAuditRecord`
- `TaskRecord`
- `TurnDecision`
- task, stop-reason, and verification enums

These contracts are intentionally not UI-shaped. They do not store chat
transcripts, prompt text, provider payloads, or presentation-specific state.
They are the durable source of truth from which projections, summaries, and
replay views are derived.

`HarnessTurn` may embed `workflow_api.WorkflowTurnResult` directly. This keeps
one-turn execution semantics canonical in `workflow_api` while allowing the
harness to persist exact per-turn execution results without reshaping them.

## Persistence and versioning

Harness persistence treats `HarnessState` as the canonical serialized record for
durable session storage, replay, and resume.

The persistence contract is:

- persist `HarnessState` as structured data using Pydantic serialization
- keep store metadata such as revisions outside `HarnessState`
- require explicit `schema_version` on persisted records
- gate unsupported versions through the store/resume layer
- preserve embedded `WorkflowTurnResult` payloads as part of canonical turn
  history
- derive prompt context, summaries, and operator views from canonical state
  instead of storing them as the primary record

`harness_api.store` provides:

- `HarnessStateStore`
- `StoredHarnessState`
- `InMemoryHarnessStateStore`
- `FileHarnessStateStore`
- schema-version constants and validation helpers
- typed corruption and optimistic-concurrency errors

Malformed file-backed session records are treated as corruption rather than
partially trusted state.

## Resume and approval durability

`harness_api.resume` classifies persisted snapshots before the multi-turn
executor continues work.

`ResumeDisposition` includes:

- `runnable`
- `waiting_for_approval`
- `approval_expired`
- `terminal`
- `incompatible_schema`
- `corrupt`

Pending approval snapshots persist exactly one incomplete tail turn plus one
matching `PendingApprovalRecord`.

Newly written pending-approval records preserve only a scrubbed base context:

- preserved: `invocation_id`, `workspace`, and `metadata`
- cleared before persistence: process env, logs, artifacts, and source
  provenance
- rebuilt on resume: execution context derived from the stored base context plus
  the current process environment

Pending approval turns also persist `TurnApprovalAuditRecord`, which keeps only
minimal approval metadata needed for replay and inspection. This avoids storing
raw request payloads on the durable turn history surface.

Approval denial, expiration, and operator cancel are fail-closed: the blocked
invocation is recorded, but later invocations from the same paused model turn do
not continue running.

## Durable executor contract

`HarnessExecutor` owns the durable multi-turn control loop above
`workflow_api`.

It is responsible for:

- storing the starting snapshot
- loading and classifying persisted sessions on resume
- selecting task ids through the configured driver
- building turn context through the configured driver
- running one parsed model response through `WorkflowExecutor`
- persisting approval waits as durable incomplete turns
- applying completed turns through the configured applier
- enforcing retry budgets and stop conditions
- persisting the next canonical snapshot plus derived artifacts

`workflow_api` remains the one-turn bridge. `harness_api` does not push durable
session state, task lifecycle logic, or resume semantics down into the workflow
layer.

## Default service path

The built-in service path composes:

- `HarnessSessionService`
- `DefaultHarnessTurnDriver`
- `MinimalHarnessTurnApplier`
- `ScriptedParsedResponseProvider` for deterministic tests and scripts

These defaults are intentionally narrow. They support root-task sessions,
basic planning/context projection, durable approval handling, and replayable
session traces without forcing applications to adopt a larger agent runtime.

## Replay and summaries

`harness_api.replay` defines the derived observability and replay surface:

- `HarnessInvocationTrace`
- `HarnessTurnTrace`
- `HarnessSessionTrace`
- `HarnessSessionSummary`
- `HarnessReplayResult`
- `StoredHarnessArtifacts`

Stored artifacts are cache-only derived views. Canonical `HarnessState` remains
authoritative for inspect, list, replay, and resume.

Public inspect/list flows rebuild trusted artifacts from canonical state through
`build_canonical_artifacts(...)` rather than trusting stale or tampered stored
summary or trace payloads.

Per-turn trace data may include:

- workflow outcome statuses
- redacted policy metadata
- approval ids and minimal approval metadata
- decision summaries and stop reasons
- per-turn verification-status snapshots for selected tasks

- canonical persisted fields stay on `HarnessState`, `TaskRecord`, and
  `HarnessTurn`
- only approved canonical fields are copied into projections
- copied task fields are limited to task identity, title, intent, lifecycle
  state, parent/dependency links, status summary, retry count, verification
  status, and artifact refs
- copied session fields are limited to session identity, root task id, current
  turn index, session start time, retry count, and configured `BudgetPolicy`
- copied turn fields are limited to turn index, selected task ids, decision
  action, decision stop reason, decision summary, and workflow outcome statuses
- derived fields such as selected/actionable flags, omission counts,
  truncation flags, and remaining budget are kept separate from copied
  canonical data
- prompt text, provider payloads, token estimates, and formatted instructions
  are not stored in canonical state and are not emitted by this builder

The default budget policy is explicit and provider-neutral:

- selected tasks are projected first
- additional actionable tasks are projected second in durable task order
- completed turns are projected third in newest-first order
- each copied text field is capped by `max_chars_per_text_field`
- total copied text across projected fields is capped by `max_total_chars`
- once the total text budget is exhausted, lower-priority optional projections
  are omitted and omission metadata is recorded
- the projection is serialized into `ToolContext.metadata["harness_turn_context"]`
  without writing anything back into canonical harness state

## Turn sequencing and commit points

The durable control loop is intentionally explicit and now uses an incomplete
turn checkpoint plus one persisted commit point per durable outcome:

1. Load or save the current `HarnessState` snapshot.
2. Classify it through `resume_session(...)`.
3. If the session is waiting on approval, require an explicit approval
   resolution before work can continue.
4. Before starting a new turn, enforce `BudgetPolicy.max_turns` and
   `BudgetPolicy.max_elapsed_seconds`.
5. Select task ids deterministically, build a `ToolContext`, and persist an
   incomplete tail `HarnessTurn` before provider or tool execution begins.
6. Obtain one `ParsedModelResponse` from the driver.
7. Reject over-budget parsed responses before dispatch when they would exceed
   `BudgetPolicy.max_tool_invocations`.
8. Execute the parsed response through `WorkflowExecutor`.
9. If the turn ends in `approval_requested`, persist exactly one incomplete tail
   `HarnessTurn` plus one matching `PendingApprovalRecord`, then stop.
10. Otherwise, apply the completed turn, stamp `ended_at`, persist any
    no-progress signals, and save the completed turn.
11. Continue only for `TurnDecisionAction.CONTINUE` and
   `TurnDecisionAction.SELECT_TASKS`; terminal stops stamp
   `session.ended_at` and `session.stop_reason`.

If the process crashes after the incomplete-turn checkpoint but before the
completed-turn save, resume classifies the tail turn as `interrupted`. The
default behavior is fail-closed: operators must explicitly opt in before the
interrupted tail turn is dropped and replayed.

## Retry and approval durability rules

Retry and recovery are explicit rather than inferred from UI state:

- provider or driver exceptions retry up to `max_provider_retries`
- completed workflow turns with any executed `ToolResult.error.retryable=True`
  retry up to `max_retryable_tool_retries`
- approval waits, approval denials, approval expirations, validation failures,
  and non-retryable tool errors do not auto-retry
- approval denial, expiration, and operator cancel are fail-closed: the
  denied invocation is recorded, but later invocations from that same parsed
  response do not run
- interrupted non-approval tail turns are not replayed automatically; they must
  be explicitly acknowledged and dropped before rerun
- every actual retry attempt increments `HarnessSession.retry_count` and the
  selected tasks' `TaskRecord.retry_count`
- optimistic-concurrency save conflicts reload the latest snapshot and retry
  only when the canonical pre-turn state is unchanged

Unresolved approvals remain durable through the combination of an incomplete
tail `HarnessTurn` and a `PendingApprovalRecord`. Resolved approvals reuse the
persisted `WorkflowTurnResult` outcome list as the durable history surface, so
approval approval, denial, timeout, and operator cancel can be replayed without
adding a second approval-history model in Phase 7. Session-level stop reasons
now distinguish `approval_denied`, `approval_expired`, and
`approval_canceled`.

The implemented evidence shape is intentionally minimal and additive:

- `VerificationEvidenceRecord.evidence_id`: stable record id
- `task_id`: optional owning task reference
- `recorded_at`: optional durable timestamp
- `summary`: optional human-readable description
- `artifact_ref`: optional pointer to an external artifact, file, or report
- `verifier_name`: optional verifier identifier

Persisted trace payloads should stay minimal and must not be treated as the
canonical store of raw request arguments, environment state, or other
unredacted execution payloads.

## Dependency direction

`harness_api` sits above the lower layers and may depend on:

- `tool_api`
- `llm_adapters`
- `llm_providers`
- `workflow_api`

The reverse direction is forbidden. In particular:

- `tool_api` must not depend on `harness_api`
- `llm_adapters` must not depend on `harness_api`
- `llm_providers` must not depend on `harness_api`
- `tools` must not depend on `harness_api`
- `workflow_api` must not depend on `harness_api`

Applications may compose `harness_api` with other layers, but the harness must
remain a library layer rather than an application dependency sink.
