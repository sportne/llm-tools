# ADR 0008: Mechanically Enforce Layer Boundaries

## Status

Accepted

## Context

`llm-tools` has a deliberately layered architecture: low-level tool and model
substrates sit below skills, workflow, harness, and assistant app surfaces. As
the project grows toward an agent framework, convenient cross-layer imports
would make it easy for reusable library surfaces to accumulate app-specific
knowledge or durable orchestration concerns by accident.

## Decision

Enforce architectural dependency direction with static tests, including import
boundary checks, private lower-layer surface checks, model-file hygiene checks,
and runtime-mediation bypass checks.

The tests should encode the current approved dependency graph and fail when a
change imports upward, reaches into private lower-layer modules, hides runtime
behavior in model files, or bypasses `ToolRuntime` mediation outside approved
locations.

## Consequences

Architecture becomes an executable contract rather than guidance maintained
only in prose.

This makes some local changes less convenient: callers may need to introduce a
public lower-layer API, move code to the correct layer, or split a shared helper
instead of importing whatever is nearby. That friction is intentional because it
keeps `tool_api`, `skills_api`, `workflow_api`, and `harness_api` reusable as
the assistant and agent surfaces grow.
