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

The repository does not add harness submodules yet, but the package boundary
is already scaffolded. This note defines the intended shape before
implementation starts.

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
