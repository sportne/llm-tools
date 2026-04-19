# harness_api

## Purpose

`harness_api` is the repository's durable orchestration layer above the
one-turn workflow primitive.

It exists to support longer-running research/workflow execution that must be:

- persisted
- resumable
- inspectable
- replayable
- verifiable

This layer is part of the supported repository direction for prescribed
workflows and future definable-agent capabilities.

## Truth resolution

This document describes the implemented `harness_api` surface. If it diverges
from code or tests, the docs should be updated to match the actual public API.

## What `harness_api` owns

`harness_api` owns durable, session-level concerns that should not be pushed
into the base tool runtime or into the one-turn workflow primitive.

That includes:

- persisted `HarnessState`, `HarnessTurn`, and related durable models
- state stores and schema-version handling
- resume classification for stored sessions
- replay traces and session summaries
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
- default driver/applier/provider protocols used by the built-in service path

The live public session API is service-based, not free functions. The key entry
point is `HarnessSessionService`, which exposes operations such as:

- `create_session(...)`
- `run_session(...)`
- `run_session_async(...)`
- `resume_session(...)`
- `resume_session_async(...)`
- `inspect_session(...)`
- `list_sessions(...)`
- `stop_session(...)`

The harness CLI is a thin product surface over this service.

## Module map

### `models.py`

Canonical durable state for sessions, turns, tasks, approvals, budgets, and
verification outcomes.

### `store.py`

Serialization, schema-version checks, stored snapshot contracts, and concrete
store implementations such as `FileHarnessStateStore`.

### `resume.py`

Classification of stored snapshots into resumable, waiting-for-approval,
terminal, corrupt, or incompatible states.

### `replay.py`

Replay artifacts, traces, summaries, and session-inspection helpers.

### `tasks.py`

Deterministic task lifecycle transitions.

### `planning.py`

Minimal planning/replanning abstractions used by the default harness driver.

### `verification.py`

Verification expectations, evidence, outcomes, timing, and no-progress signals.

### `context.py`

Projection of durable harness state into lower-level `ToolContext` metadata.

### `executor.py`

Durable run loop coordinating the store, the driver, one-turn workflow
execution, approval persistence/resumption, and stop conditions.

### `session.py`

Public session service plus built-in default driver, scripted provider, and
turn-applier helpers.

## Boundary with `workflow_api`

The cleanest current seam is the handoff through `WorkflowTurnResult`.

- `workflow_api` owns one parsed model turn and returns `WorkflowTurnResult`
- `harness_api` persists that result inside `HarnessTurn` / `HarnessState`
- `harness_api` decides whether the session should continue, stop, wait for
  approval, or be resumed later

This dependency direction is correct and is enforced by the architecture tests:
`workflow_api` must not depend on `harness_api`.

### What remains in `workflow_api`

The repo still keeps some assistant-oriented in-memory session behavior in
`workflow_api`, including chat-session and protection helpers. That means the
semantic boundary is blurrier than the import graph alone suggests.

For current purposes, the rule is:

- use `workflow_api` for one-turn execution and assistant chat/protection
  helpers
- use `harness_api` for durable orchestration and persisted research/workflow
  state

## Boundary with `apps`

Applications may compose `harness_api`, but `harness_api` itself must not depend
on `apps`.

Current app consumers are:

- `llm_tools.apps.harness_cli`
- `llm_tools.apps.streamlit_assistant` research-session features

App presentation state, Streamlit widgets, and assistant transcript UX belong in
`apps`, not in `harness_api`.

## Current strengths

The strongest aspects of the current harness design are:

- explicit durable contracts
- clear persistence and resume modeling
- replay and summary support
- no dependency from lower layers back into harness
- a real public session API used by both tests and product entrypoints

## Current cleanup targets

The main issues in the current harness layer are concentration, not direction.

Primary cleanup targets are:

- `executor.py`, which combines too many run-loop concerns in one file
- `session.py`, which bundles public service APIs together with default driver
  and applier implementations
- clearer documentation of how approval waiting is divided between workflow and
  harness layers

These are refactor targets. They are not evidence that the durable harness
surface should be removed.
