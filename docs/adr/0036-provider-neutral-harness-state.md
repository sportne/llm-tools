# ADR 0036: Keep Harness State Provider-Neutral

## Status

Accepted

## Context

The harness builds model-facing context from durable session and task state, but
provider prompts, serialized messages, token accounting, and transport payloads
are shaped by a particular model service and app flow. Persisting those payloads
as canonical harness state would couple durable orchestration to provider
details and increase retention of sensitive prompt material.

## Decision

Keep canonical harness state provider-neutral.

Harness context projections should be derived from `HarnessState` and may be
attached to execution metadata for inspection, but canonical harness models
should not store provider prompts, provider messages, token fields, or transport
payloads as primary durable state. Exact workflow execution results may still be
embedded in `HarnessTurn` according to ADR 0035.

## Consequences

Harness resume, replay, task lifecycle, and verification remain less coupled to
specific provider transports or prompt layouts.

The cost is that exact provider prompt replay is not available from canonical
harness state by default. That is accepted because durable orchestration should
preserve task/session truth and workflow outcomes without making provider
payload retention the default.
