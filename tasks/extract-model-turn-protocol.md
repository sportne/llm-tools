# Extract a Workflow-Level Model-Turn Protocol Module

## Status

[ ] Not started

## Task

Extract a workflow-level Model-Turn Protocol module shared by Assistant Chat and
Deep Task.

The module should:

- accept caller-built provider messages;
- emit only redacted Model-Turn Events;
- own protocol fallback and repair retries;
- keep optional protection inside the module implementation;
- support sync and async execution;
- accept an optional event observer;
- return one parsed Model Turn.

## References

- [ADR 0001](../docs/adr/0001-model-turn-protocol-module.md)
- [Architecture](../docs/design/architecture.md)
- [Context](../docs/CONTEXT.md)
