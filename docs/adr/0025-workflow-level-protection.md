# ADR 0025: Treat Protection as Workflow-Level Behavior

## Status

Accepted

## Context

Assistant Chat and Deep Task both need to handle proprietary or otherwise
sensitive information before, during, and after model turns. If protection were
implemented only as app UI moderation, each product surface would need its own
prompt checks, response checks, provenance handling, redaction, persistence
scrubbing, and replay behavior.

## Decision

Treat protection as workflow-level behavior that app and harness surfaces can
compose.

Protection checks, sensitivity categories, provenance, redacted model-turn
events, final-answer replacement, and purge behavior should be shared through
workflow and runtime surfaces rather than owned only by NiceGUI presentation
code. App layers remain responsible for user interaction, notices, challenge
dialogs, and product-specific display.

## Consequences

Assistant Chat and Deep Task can share protection behavior, and persisted
workflow or harness records can be scrubbed consistently when protected material
must not be replayed or inspected later.

The cost is that lower layers carry more security and policy concepts than a
pure tool-execution library would. That is accepted because protection affects
the correctness of model turns, tool-result retention, and durable replay, not
just the UI.
