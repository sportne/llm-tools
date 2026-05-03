# ADR 0010: Mediate Tool Execution Through ToolRuntime

## Status

Accepted

## Context

`llm-tools` exposes typed tool implementations, but tool execution is also where
validation, policy, approval, execution services, redaction, timeout handling,
source provenance, and result normalization meet. If workflow, harness,
assistant, or integration code calls tool implementations directly, those
cross-cutting guarantees become optional and inconsistent.

## Decision

All normal tool execution must go through `ToolRuntime`.

Tool implementations may define their direct `invoke` and `ainvoke` methods,
but callers outside approved runtime internals and tests should dispatch through
`ToolRuntime.execute(...)` or the corresponding mediated workflow or harness
surface. Architecture tests should continue to flag direct invocation bypasses.

## Consequences

The runtime remains the single place that resolves tools, validates inputs and
outputs, applies policy, provides execution services, redacts sensitive values,
normalizes errors, records execution metadata, and returns canonical
`ToolResult` envelopes.

This adds some indirection for simple callers and tests, but it keeps tool
behavior consistent across direct library use, assistant chat, durable harness
sessions, and future agent flows.
