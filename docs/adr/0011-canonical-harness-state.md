# ADR 0011: Treat HarnessState as the Canonical Durable Record

## Status

Accepted

## Context

Durable harness sessions need to support resume, inspect, list, replay, and
operator summaries. Those flows can be served from convenient summary and trace
artifacts, but trusting those artifacts as primary state would let stale,
partial, or tampered derived views redefine what actually happened in a
session.

## Decision

Persist `HarnessState` as the canonical durable record for harness sessions.

Stored summaries, traces, and replay artifacts are cache-only derived views.
Trusted inspect, list, and replay flows should rebuild artifacts from
`HarnessState` when needed instead of treating stored artifacts as authoritative.
Store metadata such as revisions may live outside `HarnessState`, but durable
session truth stays in the validated canonical state model.

## Consequences

Resume and inspection logic have one trusted source of truth, and malformed or
stale derived artifacts cannot change session state.

This limits how much extra information summaries and traces can carry by
default. Richer observability must either be derivable from canonical state or
added deliberately to the durable model, rather than smuggled into cache
artifacts that later become de facto state.
