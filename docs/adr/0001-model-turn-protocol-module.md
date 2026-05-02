# ADR 0001: Extract a Workflow-Level Model-Turn Protocol Module

## Status

Accepted

## Context

`workflow_api/chat_runner.py` and `apps/assistant_research_provider.py` both
encode model-facing turn protocol behavior: native structured responses,
staged schemas, prompt-emitted tool calls, protocol fallback, repair retries,
provider capability detection, and final-response parsing.

This duplication weakens locality. Protocol changes have to be made in both
Assistant Chat and Deep Task paths, and tests exercise similar fake-provider
cases through multiple caller-shaped surfaces.

## Decision

Create a workflow-level Model-Turn Protocol module that produces one parsed
Model Turn from already-serialized provider messages.

The module interface should:

- accept caller-built provider messages, prepared model interaction metadata,
  a final-response model, provenance, and optional decision context;
- return one parsed model turn;
- support sync and async execution;
- accept an optional event observer;
- emit only redacted Model-Turn Events;
- own protocol-stage observability, protocol fallback, repair retries, provider
  capability detection, and response parsing;
- keep optional prompt/response protection inside the module implementation,
  without making protection the primary interface.

Callers remain responsible for caller-specific context projection. Assistant
Chat builds chat/session messages. Deep Task builds harness-task messages. The
Model-Turn Protocol module may append protocol-stage messages, repair messages,
protection guard messages, and forced-final instructions.

The module should not emit raw provider messages or raw provider responses
across its seam. Redaction is an invariant of Model-Turn Events.

## Consequences

Protocol behavior gains locality in `workflow_api`, shared by Assistant Chat and
Deep Task.

Assistant Chat can adapt Model-Turn Events into chat status and inspector
events. Deep Task can later adapt the same events into durable progress or UI
feedback without reimplementing protocol internals.

The module does not become a chat or harness prompt builder. That keeps the
seam focused on the Model-Turn Protocol rather than on caller-specific state.

Tests should move toward the Model-Turn Protocol interface as the primary test
surface. Once covered there, duplicated protocol tests in app- and chat-shaped
callers can be reduced.
