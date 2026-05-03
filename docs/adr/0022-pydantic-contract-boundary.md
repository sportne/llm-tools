# ADR 0022: Use Pydantic as the Contract Boundary

## Status

Accepted

## Context

`llm-tools` relies on typed contracts for tool definitions, model-output
normalization, workflow results, durable harness state, provider configuration,
assistant persistence, and app-facing settings. Those contracts need consistent
runtime validation, JSON serialization, schema generation, and corruption
detection.

## Decision

Use Pydantic models as the canonical contract boundary for public typed data,
model-visible schemas, persisted durable state, and validated app storage
payloads.

Runtime services may use ordinary Python objects internally, but data crossing
tool, workflow, harness, provider, or app persistence seams should use explicit
Pydantic contracts when it is part of the supported surface.

## Consequences

The project gets one validation and serialization model for schemas, tool
inputs and outputs, durable session records, replayable workflow results, and
stored assistant data.

The cost is tighter coupling to Pydantic conventions and migration behavior.
That is acceptable because schema clarity and validated boundaries are central
to the library's purpose.
