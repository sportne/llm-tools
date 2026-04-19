# harness_api

## Purpose

`harness_api` is the durable, multi-turn orchestration layer above
`workflow_api`.

`workflow_api` stays the one-turn bridge: it prepares a model-facing contract,
accepts parsed model output, and executes the returned tool invocations
sequentially. `harness_api` owns the session-level control flow that persists
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

## Module map

The package started as a boundary scaffold and now has concrete modules for the
implemented harness surface, plus room for future extension.

Implemented modules:

- `models` for canonical session, turn, task, state, and decision models
- `tasks` for explicit task lifecycle transitions and state-machine helpers
- `executor` for multi-turn harness control flow
- `store` for durable state persistence and rehydration
- `resume` for persisted-session classification and startup validation
- `verification` for verifier contracts and evidence capture
- `planning` for task decomposition and replanning hooks
- `context` for derived turn-context construction
- `replay` for trace reconstruction and debugging support
- `session` for the public Python session service and the minimal built-in
  runner surfaces

Future extensions can still add richer planning, decomposition, verification,
and app-facing helpers without changing the role of the implemented modules
above.

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
  timestamps, optional `WorkflowTurnResult`, optional `TurnDecision`, and any
  no-progress signals detected for the turn.
- `HarnessSession`: session-level metadata including session id, root task id,
  budget policy, current turn index, start time, and terminal stop metadata.
- `HarnessState`: top-level persisted envelope containing `schema_version`,
  `HarnessSession`, all `TaskRecord` entries, all persisted verification
  evidence records, all `HarnessTurn` entries, and any durable pending
  approvals required for resume.

`harness_api.verification` now defines the canonical verification contracts:

- `Verifier`: protocol for verification implementations that inspect durable
  task and session state and return a structured `VerificationResult`.
- `VerificationExpectation`: task-level declaration of what must be verified,
  whether it blocks completion, and when it should run.
- `VerificationResult`: first-class verifier output with task id, status,
  checked timestamp, expectation coverage, evidence, and optional failure mode.
- `VerificationEvidenceRecord`: persisted evidence record stored once at the
  `HarnessState` level and referenced from `VerificationOutcome.evidence_refs`.
- `VerificationFailureMode`: stable taxonomy for failed or inconclusive
  verification outcomes.
- `NoProgressSignal`: persisted structured signal describing stalled or
  repetitive session behavior.
- Supporting enums: `VerificationTrigger`, `VerificationTiming`, and
  `NoProgressSignalKind`.

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
  requires unique dependency ids, verification expectation ids, and artifact
  refs.
- `TaskRecord` requires `verification.status == passed` when the task is
  `completed` and any attached expectation has
  `required_for_completion=True`.
- `TaskRecord` uses `superseded_by_task_id` only for `SUPERSEDED` tasks.
- `TurnDecision` requires `stop_reason` only for stop decisions and requires
  unique `selected_task_ids`.
- `PendingApprovalRecord` requires its `pending_index` to match the persisted
  approval request and reference an invocation in the stored parsed response.
- `HarnessTurn` requires `ended_at` once a decision has been recorded, and a
  turn that stops with `HarnessStopReason.NO_PROGRESS` must carry at least one
  `NoProgressSignal`.
- `HarnessSession` requires `current_turn_index >= 0`, requires non-empty
  `session_id`, `root_task_id`, and `started_at`, and requires `stop_reason`
  whenever `ended_at` is set.
- `HarnessState` requires unique task ids, root-task consistency, acyclic
  dependency/parent/supersession graphs, verification evidence ids, evidence
  refs that resolve without borrowing evidence owned by another task,
  no-progress signal task ids that resolve to known tasks, contiguous
  ascending turn indices starting at `1`, selected task ids that resolve to
  known tasks, at most one incomplete tail turn, unique pending approval ids,
  and
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
- persist verification evidence records at the `HarnessState` level and keep
  task-level `VerificationOutcome.evidence_refs` as stable references into that
  shared evidence list, with optional task ownership on each evidence record
- persist no-progress signals on `HarnessTurn` records so a no-progress stop
  can carry canonical signal data rather than relying on free-form summaries
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

`harness_api.resume` classifies persisted snapshots before the multi-turn
executor continues work.

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

## Durable executor contract

`harness_api.executor` now provides the concrete multi-turn control loop above
`workflow_api`:

- `HarnessExecutor`: durable sync and async session runner over a
  `HarnessStateStore`
- `HarnessTurnDriver`: task-selection, context-building, and parsed-response
  production for one turn
- `HarnessTurnApplier`: post-turn state application that returns canonical
  `TurnDecision` values
- `HarnessRetryPolicy`: explicit retry budgets for provider, retryable-tool,
  and optimistic-concurrency recovery
- `HarnessExecutionResult`: final stored snapshot plus the post-run resume
  classification

`HarnessExecutor` keeps `workflow_api` as the one-turn layer. It never asks the
workflow layer to own persisted session state, task lifecycles, approval stop
semantics, or retry accounting.

## Planning and replanning

`harness_api.planning` adds the first minimal planner abstraction without
changing canonical persisted state:

- `HarnessPlanner`: protocol for deterministic task selection and
  replanning-trigger detection
- `TaskSelection`: provider-neutral selected task ids plus stable diagnostic
  blocked reasons
- `ReplanningTrigger`: explicit derived reasons to revisit task selection
- `DeterministicHarnessPlanner`: the minimal concrete planner

The implemented deterministic selection rules are:

- select exactly one task per turn
- a task is actionable only when its status is `pending` or `in_progress` and
  all dependencies are already `completed`
- blocked and terminal tasks are never selected
- actionable `in_progress` tasks win before actionable `pending` tasks
- durable `HarnessState.tasks` order is the only tie-breaker
- no actionable task returns an empty selection rather than synthesizing new
  work

The implemented replanning triggers are derived, pure signals:

- `select_tasks_requested` when a turn decision asks to select tasks again
- `selected_task_blocked` when previously selected work is now blocked
- `selected_task_terminal` when previously selected work reached a terminal
  lifecycle state
- `new_derived_task_created` when new derived tasks appear in canonical state
- `no_actionable_tasks_remaining` when non-terminal work remains but nothing is
  currently actionable

These triggers do not mutate `HarnessState`, synthesize prompt text, or create
replacement tasks. They are a reusable policy surface for future harness
drivers and appliers.

## Turn-context construction

`harness_api.context` adds a provider-neutral derived-context builder that
projects canonical session state into one bounded turn-context payload:

- `TurnContextBudget`: explicit count and character limits
- `TaskContextProjection`, `TurnContextProjection`, and
  `BudgetContextProjection`: structured derived views
- `HarnessTurnContext`: the composed provider-neutral turn context
- `TurnContextBundle`: the built `ToolContext` plus its structured projection
- `HarnessContextBuilder` and `DefaultHarnessContextBuilder`: the public
  context-builder contract and the minimal implementation

Canonical vs derived rules are explicit:

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

The durable control loop is intentionally explicit and uses one persisted
commit point per durable outcome:

1. Load or save the current `HarnessState` snapshot.
2. Classify it through `resume_session(...)`.
3. If the session is waiting on approval, require an explicit approval
   resolution before work can continue.
4. Before starting a new turn, enforce `BudgetPolicy.max_turns` and
   `BudgetPolicy.max_elapsed_seconds`.
5. Select task ids deterministically, build a `ToolContext`, and obtain one
   `ParsedModelResponse` from the driver.
6. Execute that parsed response through `WorkflowExecutor`.
7. If the turn ends in `approval_requested`, persist exactly one incomplete tail
   `HarnessTurn` plus one matching `PendingApprovalRecord`, then stop.
8. Otherwise, apply the completed turn, evaluate the post-turn budget check for
   `max_tool_invocations`, stamp `ended_at`, and persist the completed turn.
9. Continue only for `TurnDecisionAction.CONTINUE` and
   `TurnDecisionAction.SELECT_TASKS`; terminal stops stamp
   `session.ended_at` and `session.stop_reason`.

If the process crashes before the post-turn save, the last durable snapshot
remains authoritative and the turn is recomputed from that snapshot on resume.
Approval waits are the only intentionally persisted incomplete turns.

## Retry and approval durability rules

Retry and recovery are explicit rather than inferred from UI state:

- provider or driver exceptions retry up to `max_provider_retries`
- completed workflow turns with any executed `ToolResult.error.retryable=True`
  retry up to `max_retryable_tool_retries`
- approval waits, approval denials, approval expirations, validation failures,
  and non-retryable tool errors do not auto-retry
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

`harness_api` owns:

- durable session state
- multi-turn progress decisions
- verification expectations, structured verifier results, persisted evidence,
  and no-progress signals
- resume and recovery semantics
- persisted history and replay artifacts

`workflow_api` will continue to own:

- model interaction preparation for a single turn
- execution of parsed invocations from that turn
- one-turn result normalization

Verification is deliberately separate from both tool invocation and planning.
Tools may produce artifacts that verifiers later inspect, and planners may
propose or reorder tasks, but verifier contracts and persisted evidence live in
`harness_api` so completion can be earned through explicit checks rather than
free-form summaries. The current model layer records no-progress signals and
stop-shape invariants, but not the detection heuristics or policy thresholds
that would produce those signals.

The architectural boundary is deliberate: `workflow_api` is a reusable
composition primitive, while `harness_api` is the durable session orchestrator
that will build on it.


## Persisted traces, summaries, and replay

`harness_api.replay` now owns the stable derived artifact contracts that sit
alongside canonical `HarnessState` in storage rather than inside it:

- `HarnessPolicySnapshot` for redacted policy decisions per invocation
- `HarnessInvocationTrace` and `HarnessTurnTrace` for durable turn observability
- `HarnessSessionTrace` and `HarnessSessionSummary` for aggregated inspection
- `HarnessReplayStep` and `HarnessReplayResult` for deterministic replay

These artifacts are persisted through `StoredHarnessState.artifacts` and
`StoredHarnessState.saved_at`. Canonical state serialization remains unchanged:
`serialize_harness_state(...)` still owns only the durable harness state model.
The stored artifacts intentionally persist redacted policy metadata and
execution-record summaries, not raw environment state or unredacted payloads.

## Public session surface

`harness_api.session` now exposes the public session-level contracts for:

- `create_session(...)`
- `run_session(...)`
- `resume_session(...)`
- `inspect_session(...)`
- `list_sessions(...)`
- `stop_session(...)`

The public surface stays injectable around `HarnessExecutor`, but the package
now also ships a narrow built-in runner for simple root-task sessions:

- `DefaultHarnessTurnDriver` composes the deterministic planner, context
  builder, and model-turn provider callback
- `MinimalHarnessTurnApplier` provides the smallest approval-aware completion
  policy suitable for tests, replay, and the CLI
- richer task decomposition and app-specific progression remain outside this
  built-in path

## Minimal CLI and app integration plan

The `llm_tools.apps.harness_cli` entrypoint is a thin client over the public
session API. It exists to keep persisted session inspection, replay, and manual
approval handling available without binding those flows to the existing apps.

The current repository scope stops at shared seams plus the documented
integration plan in `docs/implementation/harness-app-integration-plan.md`.
The remaining Streamlit apps should consume the public session service and stored artifacts rather than reimplementing harness orchestration or trace models in app code.
