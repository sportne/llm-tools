# ADR 0018: Separate One-Turn Workflow from Durable Harness Orchestration

## Status

Accepted

## Context

`llm-tools` supports both single model-turn execution and longer-running durable
work. Those flows share model-output parsing and runtime-mediated tool
execution, but durable sessions also need persistence, resume classification,
approval durability, task lifecycle, replay, verification, and no-progress
signals. Pushing those concerns into the one-turn layer would make the reusable
workflow primitive depend on harness-shaped state.

## Decision

Keep `workflow_api` responsible for one parsed model turn and its mediated tool
execution, and keep `harness_api` responsible for durable session
orchestration.

The boundary between them is the handoff from `WorkflowTurnResult` into
persisted `HarnessTurn` and `HarnessState`. `harness_api` may depend on
`workflow_api`; `workflow_api` must not depend on `harness_api`.

## Consequences

One-turn execution stays reusable for assistant chat and other callers that do
not need durable orchestration, while the harness can build resume, replay,
approval, task, and verification semantics on top.

The cost is an explicit translation and persistence step between workflow
results and harness state. That duplication is preferable to letting durable
session concepts leak downward into every workflow caller.
