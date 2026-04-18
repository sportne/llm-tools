# harness_api

## Purpose

`harness_api` is the future home for durable, multi-turn orchestration above
`workflow_api`.

`workflow_api` stays the one-turn bridge: it prepares a model-facing contract,
accepts parsed model output, and executes the returned tool invocations
sequentially. `harness_api` will own session-level control flow that persists
across turns.

## Responsibilities

`harness_api` is responsible for session-oriented concerns that do not belong
in the one-turn workflow bridge:

- durable session execution
- turn sequencing and resumption
- task and subtask tracking
- planning and replanning hooks
- persistence and replay
- verification orchestration
- budget and stop-condition evaluation
- session-level observability and summaries

It should consume the lower-layer typed contracts rather than redefining them.

## Proposed module map

The package is intentionally scaffolded first as a boundary, then expanded with
explicit modules as the harness implementation lands.

Proposed modules:

- `models` for canonical session, turn, task, state, and decision models
- `tasks` for explicit task lifecycle transitions and state-machine helpers
- `executor` for multi-turn harness control flow
- `store` for durable state persistence and rehydration
- `resume` for persisted-session classification and startup validation
- `verification` for verifier contracts and evidence capture
- `planning` for task decomposition and replanning hooks
- `context` for derived turn-context construction
- `replay` for trace reconstruction and debugging support

The package boundary is now scaffolded with `harness_api.models` as the first
concrete harness submodule. The remaining modules above stay planned and can be
added incrementally as the implementation lands.

## Canonical model inventory

`harness_api.models` now defines the canonical persisted contracts for durable
session execution:

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
  timestamps, optional `WorkflowTurnResult`, and optional `TurnDecision`.
- `HarnessSession`: session-level metadata including session id, root task id,
  budget policy, current turn index, start time, and terminal stop metadata.
- `HarnessState`: top-level persisted envelope containing `schema_version`,
  `HarnessSession`, all `TaskRecord` entries, all `HarnessTurn` entries, and
  any durable pending approvals required for resume.

These contracts are intentionally not UI-shaped. They do not store chat
transcripts, prompt text, provider payloads, or presentation-specific state.
They are the durable source of truth from which future context projections,
debug views, or summaries can be derived.

`HarnessTurn` may embed `workflow_api.WorkflowTurnResult` directly. This keeps
one-turn execution semantics canonical in `workflow_api` while allowing the
harness to persist exact per-turn execution results without copying or
reshaping them.

## Model invariants

The canonical models enforce the following invariants:

- `BudgetPolicy` requires at least one configured limit, and every configured
  limit must be positive.
- `VerificationExpectation` requires non-empty durable expectation text.
- `VerificationOutcome` requires `checked_at` once verification has run, and
  `evidence_refs` must be unique.
- `TaskRecord` requires non-empty `task_id`, `title`, and `intent`; tracks
  whether the task is user-requested or derived; forbids self dependency; and
  requires unique dependency ids and artifact refs.
- `TaskRecord` uses `superseded_by_task_id` only for `SUPERSEDED` tasks.
- `TurnDecision` requires `stop_reason` only for stop decisions and requires
  unique `selected_task_ids`.
- `PendingApprovalRecord` requires its `pending_index` to match the persisted
  approval request and reference an invocation in the stored parsed response.
- `HarnessTurn` requires `ended_at` once a decision has been recorded.
- `HarnessSession` requires `current_turn_index >= 0`, requires non-empty
  `session_id`, `root_task_id`, and `started_at`, and requires `stop_reason`
  whenever `ended_at` is set.
- `HarnessState` requires unique task ids, root-task consistency, acyclic
  dependency/parent/supersession graphs, contiguous ascending turn indices
  starting at `1`, selected task ids that resolve to known tasks, at most one
  incomplete tail turn, unique pending approval ids, and
  `session.current_turn_index == len(turns)`.

The invariant split is deliberate. Per-record rules live on the individual
models, while cross-record integrity checks live on `HarnessState`.

## Task lifecycle operations

`harness_api.tasks` owns the deterministic task state machine above the durable
models. The current lifecycle helpers are:

- `create_root_task`
- `create_derived_task`
- `start_task`
- `block_task`
- `unblock_task`
- `complete_task`
- `fail_task`
- `cancel_task`
- `supersede_task`

The lifecycle rules are intentionally explicit:

- root tasks are created as `user_requested`; derived tasks must point at a
  known parent task
- tasks do not reopen after terminal states; retries and replans create new
  derived tasks instead
- `start_task` requires a `pending` task whose dependencies are already
  `completed`
- `block_task` is allowed from non-terminal active states
- `unblock_task` returns a blocked task to `pending`
- `complete_task` requires `in_progress`
- `fail_task` and `cancel_task` move work to terminal states without mutating
  unrelated tasks
- `supersede_task` requires a replacement task id and records that durable
  replacement link instead of reusing failure/cancelation semantics

Invalid transitions raise `InvalidTaskLifecycleError`, so replay and future
planning logic get stable failure modes instead of silent coercions.

## Persistence and versioning expectations

Harness persistence should treat `HarnessState` as the canonical serialized
record for durable session storage, replay, and resume. The persistence
contract is:

- persist `HarnessState` as structured data using Pydantic serialization rather
  than ad hoc prompt or UI projections
- keep timestamps as durable string fields so stores are not forced into a
  datetime parsing policy too early
- require `schema_version` on input for every persisted harness state record;
  it is not defaulted by the model
- bump `schema_version` on incompatible persisted-state changes and gate
  rehydration on explicit migration logic
- preserve embedded `WorkflowTurnResult` payloads as part of the canonical
  turn record so replay can reconstruct exact workflow outcomes
- round-trip via `model_dump(mode="json")` and `model_validate_json(...)`
- derive prompt context, summaries, and operator views from canonical state
  instead of persisting those projections as the primary record

`harness_api.store` provides the initial durable storage contract:

- `HarnessStateStore` defines `load_session`, `save_session`, and
  `delete_session`
- `StoredHarnessState` wraps canonical `HarnessState` plus store-managed
  revision metadata
- `InMemoryHarnessStateStore` proves the contract and optimistic concurrency
  behavior in tests
- `CURRENT_HARNESS_STATE_SCHEMA_VERSION` and
  `SUPPORTED_HARNESS_STATE_SCHEMA_VERSIONS` gate what can be rehydrated without
  migration logic

Store metadata such as revision ids stays in the store layer rather than inside
`HarnessState`. Summaries and indexes are optional derived projections and are
not required to resume a session.

## Resume semantics

`harness_api.resume` classifies persisted snapshots before any future
multi-turn executor continues work.

`ResumeDisposition` currently distinguishes:

- `runnable`
- `waiting_for_approval`
- `approval_expired`
- `terminal`
- `incompatible_schema`
- `corrupt`

Resume validation checks include:

- schema version must be supported
- canonical `HarnessState` validation must still pass when the snapshot is
  rehydrated
- terminal sessions do not resume execution
- incomplete turns must appear only at the tail
- incomplete turns require exactly one durable `PendingApprovalRecord`
- the persisted pending approval must match the incomplete turn’s final
  `approval_requested` workflow outcome
- expired approvals are surfaced explicitly rather than silently discarded

This keeps resume semantics grounded in persisted typed state instead of hidden
in-memory executor state.

## Dependency direction

`harness_api` sits above the lower layers and may depend on:

- `tool_api` for canonical tool contracts
- `llm_adapters` for parsing model output into canonical invocation payloads
- `llm_providers` for model transport and typed response execution
- `workflow_api` for one-turn execution primitives

The reverse direction is forbidden:

- `tool_api` must not depend on `harness_api`
- `llm_adapters` must not depend on `harness_api`
- `llm_providers` must not depend on `harness_api`
- `tools` must not depend on `harness_api`
- `workflow_api` must not depend on `harness_api`

Applications may compose `harness_api` with other layers, but the harness must
remain a library layer rather than an application dependency sink.

Persisted session and replay data should classify and redact sensitive fields
by default before storage or presentation.

## Boundary with `workflow_api`

`workflow_api` remains the one-turn bridge between parsed model output and
tool execution.

`harness_api` will own:

- durable session state
- multi-turn progress decisions
- resume and recovery semantics
- persisted history and replay artifacts

`workflow_api` will continue to own:

- model interaction preparation for a single turn
- execution of parsed invocations from that turn
- one-turn result normalization

The architectural boundary is deliberate: `workflow_api` is a reusable
composition primitive, while `harness_api` is the durable session orchestrator
that will build on it.
