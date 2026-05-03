# ADR 0031: Treat Malformed Persistence as Corruption

## Status

Accepted

## Context

The harness and Assistant both persist structured state that later affects
resume, replay, inspection, session ownership, preferences, workflow state, and
runtime configuration. Best-effort partial recovery from malformed JSON or
invalid stored payloads can silently drop fields, trust incomplete state, or
hide data integrity problems.

## Decision

Treat malformed persisted records as corruption at durable trust boundaries.

Harness session files, serialized harness state, assistant stored JSON payloads,
and encrypted assistant field envelopes should be fully validated before use.
When validation, decoding, schema-version checks, or authenticated decryption
fails, callers should receive an explicit corruption or unsupported-version
error instead of silently repairing, dropping, or partially trusting the record.

## Consequences

Durable resume, inspect, replay, and app-load flows do not continue from
partially trusted state.

The cost is less forgiving recovery for damaged files or incompatible stored
payloads. That is accepted because explicit corruption is safer than quietly
changing session truth or losing user-owned data without a visible failure.
