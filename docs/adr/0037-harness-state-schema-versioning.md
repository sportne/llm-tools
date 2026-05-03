# ADR 0037: Version Persisted Harness State Explicitly

## Status

Accepted

## Context

Harness sessions are durable records used for resume, replay, inspection, and
task lifecycle state. As the harness models evolve, old or future session files
could otherwise be interpreted under the wrong model shape or partially loaded
without a reviewed migration path.

## Decision

Persist an explicit `schema_version` on canonical harness state and gate loads
or saves through supported-version checks.

Unsupported harness state versions should fail with an explicit
unsupported-version error rather than best-effort deserialization. Store
metadata such as revisions may remain outside canonical state, but schema
compatibility belongs to the durable state contract.

## Consequences

Resume and replay do not silently reinterpret session files across incompatible
model changes.

The cost is migration friction when harness persistence changes. That is
accepted because durable orchestration state must fail visibly when the current
code cannot safely understand a stored session.
