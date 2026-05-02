# ADR 0002: Extract Assistant Runtime Assembly Bundle

## Status

Accepted

## Context

`apps/assistant_app/controller_core.py` currently assembles the same app-layer
runtime concerns in both Assistant Chat and Deep Task paths:

- effective assistant configuration;
- provider construction;
- visible tool specs under admin feature flags;
- enabled and exposed tool names;
- policy and workflow executor construction;
- session-scoped environment overrides;
- prompt construction;
- protection environment and controller construction.

This weakens locality. Permission, tool exposure, prompt, and protection changes
must be kept aligned across `_build_runner(...)` and
`_build_harness_service(...)`, even though those flows differ only after shared
runtime assembly is complete.

## Decision

Create an app-layer Assistant Runtime Assembly module that returns an
AssistantRuntimeBundle.

The bundle should own the shared session assembly decisions:

- effective configuration;
- provider;
- registry and workflow executor;
- root/workspace capability gating;
- enabled and exposed tool names;
- policy;
- session environment overrides;
- chat and Deep Task system prompts;
- protection setup for Assistant Chat and Deep Task.

The bundle may expose thin mode-specific constructors for caller convenience,
such as building an Assistant Chat runner or Deep Task harness service, but the
primary interface should not be shaped around controller event flow,
transcript persistence, or queued work handling.

The controller remains responsible for product flow control: session selection,
transcript updates, queued events, approval UI state, persistence timing, and
mapping execution results back to UI state.

## Consequences

Assistant Runtime Assembly gains locality in `apps` without pushing product
concerns into `workflow_api` or `harness_api`.

Assistant Chat and Deep Task can share one source of truth for what tools,
policy, provider, prompts, and protection are active in a session.

Tests should move toward the AssistantRuntimeBundle interface for permission,
tool exposure, prompt, and protection assembly. Controller tests can then focus
on user-flow behavior instead of reconstructing assembly details.
