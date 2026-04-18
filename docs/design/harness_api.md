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
- `executor` for multi-turn harness control flow
- `store` for durable state persistence and rehydration
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
  completed, failed, and canceled work.
- `VerificationStatus`: task verification enum for not-run, passed, failed, and
  inconclusive checks.
- `HarnessStopReason`: canonical session stop reasons such as completed,
  budget-exhausted, verification-failed, no-progress, canceled, and error.
- `TurnDecisionAction`: post-turn action enum for continue, select-tasks, and
  stop decisions.
- `BudgetPolicy`: configured session budget limits. It holds only durable limit
  settings and requires at least one positive configured bound.
- `VerificationOutcome`: persisted verification result attached to a task,
  including status, checked timestamp, summary, and evidence references.
- `TaskRecord`: canonical durable unit-of-work record, with task identity,
  title, lifecycle status, dependency ids, verification state, and durable
  timestamps.
- `TurnDecision`: harness-owned decision emitted after a turn completes,
  including action, selected task ids, optional stop reason, and summary text.
- `HarnessTurn`: one persisted harness turn, including the turn index, durable
  timestamps, optional `WorkflowTurnResult`, and optional `TurnDecision`.
- `HarnessSession`: session-level metadata including session id, root task id,
  budget policy, current turn index, start time, and terminal stop metadata.
- `HarnessState`: top-level persisted envelope containing `schema_version`,
  `HarnessSession`, all `TaskRecord` entries, and all `HarnessTurn` entries.

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
- `VerificationOutcome` requires `checked_at` once verification has run, and
  `evidence_refs` must be unique.
- `TaskRecord` requires non-empty `task_id` and `title`, forbids self
  dependency, and requires unique dependency ids.
- `TurnDecision` requires `stop_reason` only for stop decisions and requires
  unique `selected_task_ids`.
- `HarnessTurn` requires `ended_at` once a decision has been recorded.
- `HarnessSession` requires `current_turn_index >= 0`, requires non-empty
  `root_task_id`, and requires `stop_reason` whenever `ended_at` is set.
- `HarnessState` requires unique task ids, task dependency ids that resolve to
  known tasks, contiguous ascending turn indices starting at `1`, selected task
  ids that resolve to known tasks, a `root_task_id` that resolves to a known task, and
  `session.current_turn_index == len(turns)`.

The invariant split is deliberate. Per-record rules live on the individual
models, while cross-record integrity checks live on `HarnessState`.

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
