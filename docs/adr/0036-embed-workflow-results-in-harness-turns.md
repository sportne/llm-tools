# ADR 0036: Embed Workflow Results in Harness Turns

## Status

Accepted

## Context

`harness_api` persists durable multi-turn sessions on top of one-turn workflow
execution. It could flatten workflow outcomes into harness-specific records, but
that would duplicate workflow semantics and risk losing details needed for
inspection, replay, approval history, protection scrubbing, and debugging.

## Decision

Allow `HarnessTurn` to embed the exact `WorkflowTurnResult` produced by
`workflow_api`.

`workflow_api` remains the canonical owner of one-turn execution semantics, tool
invocation outcomes, final responses, and approval-request results. The harness
persists those results as part of durable turn history and adds durable session,
task, replay, resume, and verification state around them.

## Consequences

Harness replay and inspection can preserve the workflow result as produced
without lossy reshaping, while workflow semantics stay canonical in
`workflow_api`.

The cost is tighter durable harness schema coupling to workflow models. That is
accepted because flattening would create a second representation of the same
execution facts and make the workflow-harness boundary harder to reason about.
