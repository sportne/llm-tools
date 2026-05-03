# ADR 0030: Keep Harness Planning Deterministic by Default

## Status

Accepted

## Context

`harness_api` supports durable longer-running work with persisted sessions,
task lifecycle state, replay, resume, verification, and no-progress signals.
Those capabilities could be paired with an autonomous planner that freely
creates and reprioritizes tasks, but doing so in the core harness would make
durable execution harder to inspect, replay, and test.

## Decision

Keep the default harness planning and task lifecycle primitives deterministic
and explicit.

The harness owns task records, task selection, replanning triggers,
verification state, no-progress signals, and durable application of completed
turns. Richer autonomous planning or agent-specific task decomposition should be
layered on deliberately through drivers, callers, or future agent surfaces
rather than assumed by the core harness service path.

## Consequences

Harness sessions remain inspectable, replayable, and testable because task
state changes happen through explicit models and deterministic lifecycle rules.

The cost is that the default harness does not feel like a fully autonomous
planner out of the box. That is accepted because durable orchestration should
provide a trustworthy substrate before higher-level agent planning policies are
added.
