# ADR 0035: Let ToolRuntime Wire Execution Services

## Status

Accepted

## Context

`ToolRuntime` mediates execution for local filesystem, subprocess, secret, and
remote-service interactions. Some bundled remote integrations need gateway
objects for services such as GitLab, Jira, Confluence, and Bitbucket. A purer
modular design would move all vendor gateway construction out of the runtime,
but then execution services, credentials, policy context, and redaction behavior
would be assembled in multiple caller-specific places.

## Decision

Let `ToolRuntime` own construction of execution services and bundled remote
gateway wiring at the mediated execution boundary.

Tool implementations should receive mediated service objects through
`ToolExecutionContext` rather than constructing their own host, subprocess,
secret, filesystem, or remote-service access paths. Bundled gateway wiring in
the runtime is accepted when it supports that single execution-services
boundary.

## Consequences

Tool execution has one place to assemble policy-aware host services, credentials,
gateways, redaction, and inspection metadata.

The cost is that `tool_api` is not completely vendor-agnostic in its execution
service wiring. That tension is accepted because centralized mediation is more
important than a purist plugin boundary for the currently bundled integrations.
Future modularization should preserve the single mediated service boundary
rather than moving gateway construction into ad hoc app or tool code.
