# Extract Assistant Runtime Assembly Bundle

## Status

[ ] Not started

## Task

Extract an app-layer Assistant Runtime Assembly module that returns an
AssistantRuntimeBundle shared by Assistant Chat and Deep Task.

The bundle should:

- own effective assistant configuration for a runtime session;
- create or carry the provider;
- build the assistant registry and workflow executor;
- apply admin feature flags to enabled tools;
- calculate exposed tool names;
- build the session policy;
- carry session-scoped environment overrides;
- provide chat and Deep Task system prompts;
- build protection setup for Assistant Chat and Deep Task;
- offer thin mode-specific constructors without taking over controller flow.

The controller should continue to own session selection, transcript updates,
queued events, approval UI state, persistence timing, and result-to-UI mapping.

## References

- [ADR 0002](../docs/adr/0002-assistant-runtime-assembly-bundle.md)
- [Architecture](../docs/design/architecture.md)
- [Context](../docs/CONTEXT.md)
