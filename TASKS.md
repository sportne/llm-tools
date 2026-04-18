# TASKS.md

## Purpose

This document defines the concrete engineering backlog required to evolve
`llm-tools` from its current typed tool and one-turn workflow foundation into a
true harness.

The repository is not starting from zero. The existing `tool_api`,
`llm_adapters`, `llm_providers`, built-in `tools`, policy/runtime,
observability, and `workflow_api` layers are the foundation. The work below
describes what must be added above or beside that foundation so the project can
own durable session execution rather than only one-turn parsing and tool
execution.

This plan is intentionally implementation-oriented. It focuses on typed
contracts, layering, persistence, verification, replay, and session usability.
It does not propose self-optimizing or meta-harness behavior.

## Status conventions

- `[ ]` Not started
- `[~]` In progress
- `[x]` Done
- `[-]` Deferred

## Definition of Done

`llm-tools` qualifies as a harness when all of the following are true:

- A first-class `harness_api` layer exists with clearly documented boundaries
  relative to `workflow_api`, `tool_api`, `llm_adapters`, `llm_providers`, and
  built-in `tools`.
- The harness owns canonical typed models for session execution, including
  session, turn, harness state, task record, task lifecycle status,
  verification outcome, stop reason, budget policy, and turn decision.
- A durable state store abstraction exists, harness state can be persisted and
  reloaded, and a session can resume from persisted state without losing its
  canonical task or verification history.
- A multi-turn control loop exists above the current one-turn
  `WorkflowExecutor`, with explicit continue, stop, retry, recovery, and resume
  semantics.
- Verification is a first-class subsystem with typed verifier contracts,
  verification expectations attached to tasks, persisted verification evidence,
  and no-progress detection.
- Planning and execution are separated by contract, even if the initial planner
  is intentionally minimal and deterministic.
- Turn context construction is explicit, budget-aware, and separate from the
  canonical persisted state.
- Approval and policy state are durable and replayable, including approval
  stops, approval resume behavior, and policy snapshots recorded in traces.
- Harness-level observability exists through structured turn traces, session
  summaries, and practical replay/debug support.
- The harness is covered by state-machine tests, end-to-end session tests,
  architecture boundary tests, and golden trace tests.
- A maintainer can start, inspect, stop, and resume harness sessions through a
  supported Python API and a minimal CLI.

## Guiding Principles

- Build on the current typed substrate. New harness behavior should extend the
  existing runtime, policy, provider, adapter, and workflow contracts rather
  than bypass them.
- Keep canonical state typed and durable. Prompt text, message history, and
  context projections are derived artifacts, not the source of truth.
- Preserve clear layer boundaries. `workflow_api` remains the one-turn
  composition bridge; `harness_api` should own multi-turn session orchestration.
- Favor deterministic control flow where possible. Stop reasons, retry rules,
  task selection, and replanning triggers should be explicit and testable.
- Make verification part of execution, not an afterthought. The harness should
  know what must be checked, what evidence was gathered, and why a task is or
  is not complete.
- Treat persistence and replay as core requirements. If a session cannot be
  resumed or explained after interruption, the harness is incomplete.
- Reuse repository terminology. Prefer names and concepts already established by
  `tool_api`, `workflow_api`, `ToolRuntime`, `ToolPolicy`, `PolicyDecision`,
  `ExecutionRecord`, and related architecture docs.
- Keep scope practical. Start with a minimal trustworthy harness before
  expanding into broader agent features.

## Phased Task Plan

### [~] Phase 1: Harness Contract and Architecture

Outcome: define the new top-level harness contract, the canonical harness
models, and the architectural boundary between one-turn workflow execution and
multi-turn session orchestration.

#### [x] 1.1 Introduce the `harness_api` layer

Description: Propose and document a first-class `src/llm_tools/harness_api/`
package as the home for session-level harness contracts and orchestration.

Why it matters: The current architecture has a strong typed substrate and a
thin `workflow_api`, but no stable home for durable session behavior. A named
layer is required before the harness can grow without leaking concepts across
existing packages.

Suggested deliverables:
- A documented package responsibility statement for `harness_api`
- A proposed module map for models, executor, persistence, verification,
  planning, context, and replay concerns
- An architecture note describing how `harness_api` depends on lower layers

Dependencies: None

Status: Done.

#### [ ] 1.2 Define canonical harness models

Description: Specify the canonical Pydantic models that the harness will own,
including session, turn, harness state, task record, verification outcome, stop
reason, budget policy, and turn decision.

Why it matters: Without a canonical typed model set, persistence, replay,
testing, CLI inspection, and cross-layer integration will drift or depend on UI
or prompt-shaped artifacts.

Suggested deliverables:
- A harness model inventory with field-level responsibilities
- Initial invariants for each model
- Serialization and versioning expectations for persisted records

Dependencies: `1.1`

#### [x] 1.3 Define harness boundaries relative to lower layers

Description: Document which responsibilities remain in `tool_api`,
`llm_adapters`, `llm_providers`, `tools`, and `workflow_api`, and which move
into `harness_api`.

Why it matters: The repository already has working lower layers. The harness
must compose them cleanly rather than duplicating runtime logic or putting
session state into provider- or UI-specific code.

Suggested deliverables:
- A boundary table for each subsystem
- A dependency-direction rule set for architecture tests
- A list of extension points where the harness consumes lower-layer artifacts

Dependencies: `1.1`, `1.2`

#### [x] 1.4 Define harness acceptance criteria

Description: Translate the high-level goal of becoming a harness into concrete
acceptance criteria aligned with the repository’s architecture style and test
strategy.

Why it matters: The repository already has a completed implementation backlog
for the typed substrate. The harness effort needs equally concrete completion
criteria to prevent partial session work from being mistaken for a finished
harness.

Suggested deliverables:
- A harness acceptance checklist
- Milestone entry and exit criteria
- Cross-references to required tests and trace artifacts

Dependencies: `1.2`, `1.3`

### [ ] Phase 2: State and Task Representation

Outcome: make work explicit and durable by introducing structured task records,
task lifecycle operations, and persisted session state that can be resumed.

#### [ ] 2.1 Define structured task records

Description: Specify a canonical `TaskRecord` or equivalent harness model for
unit-of-work tracking, including task identity, intent, status, dependencies,
verification expectations, and relevant artifacts.

Why it matters: A harness cannot reason about progress, retries, or completion
if work only exists as transient prompt text or UI messages.

Suggested deliverables:
- A typed task model with required and optional fields
- Rules for representing user-requested tasks versus derived subtasks
- A compatibility note describing how task records relate to current
  `workflow_api` turn results

Dependencies: `1.2`

#### [ ] 2.2 Define task lifecycle operations

Description: Specify the allowed task state transitions and the operations that
move tasks through their lifecycle, such as create, start, block, complete,
fail, cancel, and supersede.

Why it matters: Trustworthy session behavior depends on deterministic task
transitions. Informal task handling will make replay, recovery, and planner
behavior ambiguous.

Suggested deliverables:
- A task lifecycle state machine
- Transition preconditions and postconditions
- Error handling rules for invalid transitions

Dependencies: `2.1`

#### [ ] 2.3 Add durable harness state storage abstractions

Description: Define storage interfaces for persisting sessions, turns, task
records, approvals, verification evidence, and summarized projections needed
for resume and replay.

Why it matters: Durable state is the line between an interactive loop and a
harness. Storage must be designed explicitly before the multi-turn executor can
be trustworthy.

Suggested deliverables:
- A storage abstraction surface and persistence contract
- A canonical persisted session envelope or equivalent top-level record
- Versioning and migration expectations for stored state

Dependencies: `1.2`, `2.1`, `2.2`

#### [ ] 2.4 Support session resume from persisted state

Description: Define how a harness session is rehydrated from storage, including
how incomplete turns, pending approvals, active tasks, and verification status
are restored.

Why it matters: Resume semantics are essential for interruption handling,
operator trust, and future CLI or app integration.

Suggested deliverables:
- A rehydration contract for session startup and resume
- A list of resume-time validation checks
- Explicit handling for corrupt, partial, or incompatible persisted state

Dependencies: `2.3`

### [ ] Phase 3: Multi-turn Control Loop

Outcome: add a session-level executor that repeatedly builds turn context,
calls the lower workflow layer, updates durable state, and decides whether to
continue or stop.

#### [ ] 3.1 Add `HarnessExecutor` above `workflow_api`

Description: Define a `HarnessExecutor` or equivalent session-level executor
that owns the durable control loop while delegating one-turn model/tool
execution to the existing `WorkflowExecutor`.

Why it matters: The current `workflow_api` is explicitly one-turn. A harness
needs a higher executor that can own session state, turn sequencing, and stop
conditions without overloading the lower layer.

Suggested deliverables:
- A session executor contract
- An execution diagram showing how `HarnessExecutor` composes
  `WorkflowExecutor`
- Clear sync and async entry-point expectations

Dependencies: `1.3`, `2.3`, `2.4`

#### [ ] 3.2 Define continue and stop semantics

Description: Specify the canonical stop reasons and turn decisions that govern
whether the harness continues, stops successfully, pauses for approval, stops
for budget limits, or yields for operator action.

Why it matters: A multi-turn loop is not trustworthy if continuation is driven
by ad hoc booleans or UI-specific checks.

Suggested deliverables:
- Typed `StopReason` and `TurnDecision` definitions
- A continuation decision table
- Rules for successful completion versus partial completion versus paused state

Dependencies: `1.2`, `3.1`

#### [ ] 3.3 Define retry and recovery behavior

Description: Specify how the harness retries transient failures, recovers from
recoverable state mismatches, and records retry attempts at the session and task
level.

Why it matters: Long-running sessions will encounter provider failures, tool
errors, approval interruptions, and partial persistence failures. Recovery must
be explicit to avoid duplicated or unsafe work.

Suggested deliverables:
- A retry classification matrix
- Retry counters and persisted retry metadata
- Recovery rules for interrupted turns and partially applied state updates

Dependencies: `2.2`, `2.4`, `3.1`, `3.2`

#### [ ] 3.4 Define per-turn state update sequencing

Description: Document the exact sequence for reading session state, selecting
work, building context, executing a turn, applying results, persisting updates,
and emitting traces.

Why it matters: The control loop becomes testable and replayable only when turn
sequencing is explicit.

Suggested deliverables:
- A per-turn state transition flow
- Commit-point rules for persisted state updates
- Failure-handling guidance for each turn phase

Dependencies: `2.3`, `3.1`, `3.2`, `3.3`

### [ ] Phase 4: Verification Subsystem

Outcome: make verification a first-class harness concern with durable evidence,
task-level expectations, and detection of sessions that are not making progress.

#### [ ] 4.1 Add first-class verifier abstractions

Description: Define typed verifier contracts that can evaluate task outcomes,
inspect artifacts, and emit structured verification results separate from tool
execution and planning.

Why it matters: A harness should not treat verification as a final free-form
message. Verification needs its own subsystem so session completion can be
earned and explained.

Suggested deliverables:
- A verifier interface and verification result model
- A taxonomy of verification statuses and failure modes
- A boundary note separating verification from tool invocation and planning

Dependencies: `1.2`, `1.3`

#### [ ] 4.2 Attach verification expectations to tasks

Description: Define how tasks declare what must be verified, when verification
is required, and whether verification is blocking for completion.

Why it matters: Verification only becomes actionable when it is tied to the
task model and not inferred after the fact from generic session output.

Suggested deliverables:
- Task-level verification expectation fields or linked records
- Rules for mandatory versus optional verification
- A mapping from task lifecycle states to verification requirements

Dependencies: `2.1`, `2.2`, `4.1`

#### [ ] 4.3 Persist verification evidence and outcomes

Description: Specify how verification evidence, verifier inputs, verifier
outputs, and final verification outcomes are stored in durable session state.

Why it matters: Persisted evidence supports replay, auditability, resume, and
operator trust when the harness claims a task is done.

Suggested deliverables:
- A verification evidence record format
- Retention and redaction expectations for evidence artifacts
- Resume-time rules for pending or stale verification records

Dependencies: `2.3`, `4.1`, `4.2`

#### [ ] 4.4 Add no-progress detection

Description: Define heuristics and explicit signals for detecting stalled
sessions, repeated ineffective turns, or repeated retries that are not moving
task state forward.

Why it matters: A harness should stop or escalate when it is spinning, rather
than quietly consuming budget while repeating the same behavior.

Suggested deliverables:
- A no-progress signal model
- Threshold and policy rules for stalled sessions
- Stop or escalation semantics when no-progress is detected

Dependencies: `3.2`, `3.3`, `4.2`, `4.3`

### [ ] Phase 5: Planning and Decision Policy

Outcome: separate planning from execution with a minimal planner and
deterministic rules for task selection and replanning.

#### [ ] 5.1 Add a minimal planner abstraction

Description: Define a planner contract that can inspect canonical harness state
and propose task additions, ordering, decomposition, or completion updates.

Why it matters: The harness needs a place for structured planning decisions
that is separate from the execution loop and separate from provider-specific
prompting concerns.

Suggested deliverables:
- A planner interface
- Planner input and output models
- A minimal planner scope statement describing what v1 planning does and does
  not do

Dependencies: `2.1`, `2.2`, `3.1`

#### [ ] 5.2 Add deterministic task selection rules

Description: Specify how the harness picks the next actionable task from
canonical state, including tie-breaking, blocked-task handling, and budget-aware
selection.

Why it matters: Task selection should be explainable and replayable even before
the planner becomes sophisticated.

Suggested deliverables:
- A task selection policy
- Deterministic ordering and tie-break rules
- Interaction rules between planner output and executor task selection

Dependencies: `2.2`, `3.2`, `5.1`

#### [ ] 5.3 Define replanning triggers

Description: Specify when the harness must re-enter planning, such as after
task completion, verification failure, approval denial, no-progress detection,
or budget exhaustion.

Why it matters: Without explicit replanning triggers, planning becomes an
implicit side effect of prompt construction and is hard to test or reason
about.

Suggested deliverables:
- A replanning trigger table
- Rules for partial-plan invalidation
- Persisted planner decision records or equivalent audit entries

Dependencies: `3.2`, `4.4`, `5.1`, `5.2`

### [ ] Phase 6: Context Construction

Outcome: create a turn-context subsystem that builds prompt-ready projections
from canonical state while enforcing explicit budget and projection rules.

#### [ ] 6.1 Add a turn-context builder

Description: Define a component that projects canonical harness state into the
turn-specific context needed by providers, adapters, and the one-turn workflow
layer.

Why it matters: Context construction is currently scattered across UI- or
session-specific logic. A true harness needs a reusable, typed context builder.

Suggested deliverables:
- A turn-context builder contract
- A typed turn-context projection model
- A mapping from canonical state fields to projected context fields

Dependencies: `2.3`, `3.1`, `5.2`

#### [ ] 6.2 Define context-budget policies

Description: Specify how the harness budgets prompt context across task state,
turn history, verification evidence, tool results, summaries, and policy
artifacts.

Why it matters: Context budget handling should be deterministic and not hidden
inside ad hoc prompt assembly logic.

Suggested deliverables:
- A `BudgetPolicy` or equivalent model for turn-context limits
- Trimming, summarization, and priority rules for context inclusion
- A set of failure modes when required context cannot fit

Dependencies: `1.2`, `6.1`

#### [ ] 6.3 Keep canonical state separate from prompt projections

Description: Define projection rules that make it explicit which data is
canonical persisted state and which data is prompt-only text or model-facing
schema.

Why it matters: This separation is necessary for replay, deterministic resume,
and future provider portability.

Suggested deliverables:
- A canonical-versus-derived data classification
- Rules for when summaries become persisted state versus ephemeral projection
- Architecture constraints preventing prompt text from becoming the source of
  truth

Dependencies: `2.3`, `6.1`, `6.2`

### [ ] Phase 7: Approval and Policy Integration

Outcome: make approvals and policy decisions durable session concerns rather
than transient workflow events.

#### [ ] 7.1 Persist approval state

Description: Define how pending approvals, approval decisions, expiration, and
related policy metadata are stored in harness state.

Why it matters: The current workflow layer can hold pending approvals in memory,
but a harness must survive interruption and resume approval-aware execution.

Suggested deliverables:
- A persisted approval record model
- Approval lifecycle rules, including timeout and cancellation
- A mapping from workflow approval events into harness state

Dependencies: `2.3`, `2.4`, `3.1`

#### [ ] 7.2 Support approval-aware stop and resume semantics

Description: Specify how the harness pauses for approval, exposes that stop
reason, and resumes safely after approval, denial, timeout, or operator cancel.

Why it matters: Approval handling is a core trust boundary between policy and
session execution.

Suggested deliverables:
- Approval-related `StopReason` variants
- Resume rules for each approval outcome
- Session UX requirements for surfaced approval state

Dependencies: `3.2`, `7.1`

#### [ ] 7.3 Snapshot policy context into traces

Description: Define how policy inputs, `PolicyDecision` summaries, approval
requirements, and related runtime context are recorded in harness traces.

Why it matters: Session replay and operator trust depend on being able to
understand why execution was allowed, denied, or paused.

Suggested deliverables:
- A policy snapshot record for each turn or invocation group
- Redaction rules for sensitive policy context
- Replay expectations for policy-aware traces

Dependencies: `3.4`, `7.1`, `7.2`

### [ ] Phase 8: Observability and Replay

Outcome: provide harness-level tracing, summaries, and replay support that make
multi-turn execution explainable after the fact.

#### [ ] 8.1 Add structured harness turn traces

Description: Define a harness trace model that records each turn’s selected
task, projected context summary, planner input/output, workflow result,
verification result, policy state, and stop/continue decision.

Why it matters: Existing execution records are necessary but not sufficient.
Harness traces need to explain how session-level decisions were made across
turns.

Suggested deliverables:
- A turn trace schema
- A session trace aggregation format
- Correlation rules linking harness traces to lower-layer `ExecutionRecord`
  entries

Dependencies: `3.4`, `4.3`, `5.3`, `6.1`, `7.3`

#### [ ] 8.2 Add session summary artifacts

Description: Define durable summary artifacts that describe session status,
completed work, open tasks, pending approvals, verification status, and final
outcomes.

Why it matters: Maintainers and future coding agents need a quick, typed view
of session state without replaying every turn manually.

Suggested deliverables:
- A session summary model
- Rules for incremental summary updates
- A clear distinction between operator-facing summaries and canonical session
  records

Dependencies: `2.3`, `4.3`, `8.1`

#### [ ] 8.3 Add replay and debug support

Description: Define a practical replay path that can reconstruct session
progress from persisted state and traces for debugging, audit, and regression
testing.

Why it matters: A durable harness should be debuggable when sessions fail,
stall, or produce surprising outcomes.

Suggested deliverables:
- A replay contract and supported replay modes
- Debug expectations for step-by-step turn inspection
- Limitations and non-goals for replay fidelity

Dependencies: `2.4`, `8.1`, `8.2`

### [ ] Phase 9: Testing and Architectural Enforcement

Outcome: make the harness trustworthy through explicit state-machine coverage,
end-to-end tests, trace fixtures, and architecture constraints.

#### [ ] 9.1 Add harness state-machine tests

Description: Design tests that validate session, task, approval, and
verification state transitions independently of any UI.

Why it matters: The harness is fundamentally a state machine. State-machine
tests provide stronger assurance than only exercising happy-path examples.

Suggested deliverables:
- State transition test matrices
- Invalid-transition coverage expectations
- Failure and recovery scenario coverage

Dependencies: `2.2`, `3.2`, `4.4`, `7.2`

#### [ ] 9.2 Add end-to-end harness tests

Description: Define end-to-end tests that run representative multi-turn
sessions through planning, context building, workflow execution, verification,
stop/resume, and final completion.

Why it matters: The harness must prove that its subsystems work together, not
just in isolation.

Suggested deliverables:
- A catalog of canonical end-to-end session scenarios
- Resume, retry, approval, and verification coverage expectations
- Guidance for deterministic provider and tool stubbing

Dependencies: `3.4`, `4.3`, `5.3`, `6.2`, `7.2`, `8.1`

#### [ ] 9.3 Add architecture tests for layering boundaries

Description: Extend architecture enforcement so `harness_api` depends only on
approved lower layers and lower layers do not import harness concerns.

Why it matters: Boundary drift is a high risk once session features are added.
Architecture tests keep the repository aligned with its stated subsystem model.

Suggested deliverables:
- Import boundary rules for `harness_api`
- Tests that protect `workflow_api` from accumulating durable harness concerns
- Documentation of allowed dependency directions

Dependencies: `1.3`

#### [ ] 9.4 Add golden trace and replay tests

Description: Define golden artifacts that capture representative harness traces
and replay outcomes for regression testing.

Why it matters: Golden traces make observability contracts concrete and protect
against accidental changes to stop reasons, planner records, or replayable
session state.

Suggested deliverables:
- Golden trace fixture strategy
- Trace stability rules and update process
- Replay assertions for representative session histories

Dependencies: `8.1`, `8.2`, `8.3`, `9.2`

### [ ] Phase 10: User-facing Session Interfaces

Outcome: expose the harness through stable session-level interfaces that make
starting, inspecting, and resuming sessions practical for maintainers and other
applications.

#### [ ] 10.1 Add a Python session API

Description: Define a Python API for creating, running, stopping, inspecting,
and resuming harness sessions without requiring the Textual apps.

Why it matters: The harness should be usable as a library, not only through a
single application surface.

Suggested deliverables:
- A public Python session API surface
- Typed request and response models for common session operations
- Usage documentation for embedding the harness in Python applications

Dependencies: `3.1`, `2.4`, `8.2`, `9.2`

#### [ ] 10.2 Add a minimal harness CLI

Description: Define a minimal CLI for starting a session, resuming a session,
inspecting current state, and reviewing recent session summaries or pending
approval state.

Why it matters: A CLI makes the harness operational for maintainers and testable
outside of a full UI integration.

Suggested deliverables:
- A CLI command set and argument contract
- Output expectations for session status and inspection commands
- Resume and approval handling behavior at the CLI boundary

Dependencies: `10.1`

#### [ ] 10.3 Integrate session state into existing apps where useful

Description: Evaluate how the existing Textual chat app or workbench can expose
harness-backed sessions without forcing the harness to depend on those apps.

Why it matters: The repository already has developer-facing and chat-facing app
surfaces. They should be able to consume the harness once it exists, but they
should not define its core contracts.

Suggested deliverables:
- An integration plan for current apps
- A list of app changes needed to consume persisted session state
- Explicit non-goals to avoid UI-led architecture drift

Dependencies: `10.1`, `10.2`

## Milestones

### [ ] Milestone A: The Loop Exists

Goal: the repository has a real harness loop rather than only a one-turn
workflow bridge.

Included phases:
- Phase 1: Harness Contract and Architecture
- Phase 2: State and Task Representation
- Phase 3: Multi-turn Control Loop

Exit criteria:
- `harness_api` is defined as a first-class layer
- canonical harness models exist
- durable session state can be persisted and reloaded
- `HarnessExecutor` can resume a session and produce explicit stop reasons

### [ ] Milestone B: The Harness Is Trustworthy

Goal: the harness can justify its decisions, preserve evidence, and be tested as
a durable state machine.

Included phases:
- Phase 4: Verification Subsystem
- Phase 5: Planning and Decision Policy
- Phase 6: Context Construction
- Phase 7: Approval and Policy Integration
- Phase 8: Observability and Replay
- Phase 9: Testing and Architectural Enforcement

Exit criteria:
- verification is first-class and persisted
- no-progress detection exists
- planning and task selection are explicit
- context construction is budget-aware and separate from canonical state
- approval and policy decisions are durable and traceable
- structured replay and golden trace coverage exist
- state-machine, end-to-end, and architecture tests cover the harness

### [ ] Milestone C: The Harness Is Usable

Goal: maintainers and downstream applications can operate the harness through
stable session-level interfaces.

Included phases:
- Phase 10: User-facing Session Interfaces

Exit criteria:
- a public Python session API exists
- a minimal CLI exists for start, inspect, and resume flows
- existing apps have a clear path to consume harness sessions without owning the
  core contracts

## Out of Scope (for now)

The following items are intentionally excluded from this plan:

- self-optimizing or meta-harness behavior
- autonomous model self-improvement loops
- multi-agent orchestration or agent-to-agent protocols
- long-term memory systems beyond the durable session state required for harness
  execution
- retrieval systems, vector databases, or knowledge indexing as a prerequisite
- distributed execution infrastructure
- manifest-based plugin discovery or plugin marketplaces
- UI-polish work beyond the minimum session interfaces needed to operate the
  harness
- broad prompt-management systems unrelated to harness execution contracts

## Suggested Implementation Order

The safest implementation order is:

1. Finish Phase 1 before creating any harness package or public session surface.
2. Complete Phase 2 before implementing a durable multi-turn executor.
3. Land Phase 3 before expanding the current chat-oriented session runner.
4. Add verification, planning, and context construction before claiming the
   harness is trustworthy.
5. Add replay and test fixtures before stabilizing the public Python API or CLI.
6. Integrate existing apps only after the harness contracts are stable enough to
   consume from above.
