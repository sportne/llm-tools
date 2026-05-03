# ADR 0014: Keep Assistant Credentials Session-Memory-Only

## Status

Accepted

## Context

The Assistant can run as a local loopback app or as a private-network hosted
server with multiple browser users. If provider API keys or tool credentials
were loaded implicitly from the Assistant server process environment, one
server-side secret could become available to unrelated app sessions or users
without an explicit user action.

## Decision

Assistant-entered provider API keys and tool credentials are held in server
memory for the current browser or app session only.

The Assistant does not read process environment variables as implicit provider
or tool credentials, and it does not persist submitted credentials in SQLite,
config files, browser storage, or provider connection presets. Credential input
fields should clear after submission. Missing provider credentials block model
turn submission for credential-required provider connections; missing tool
credentials block the affected tool exposure rather than the whole model turn.
In-memory credentials expire after a bounded session TTL, initially two hours,
and must be re-entered through the app the next time the expired credential is
needed.
When the Assistant is reachable over non-loopback HTTP, secret entry should be
blocked unless the operator passes an explicit insecure-hosted-secrets override;
normal hosted use should terminate TLS before accepting credentials.

Lower-level library and CLI provider calls may still accept explicit credentials
or conventional environment-variable credentials outside the Assistant app
boundary.

## Consequences

Credential authority is explicit per app session and does not leak from the
server process into hosted users by default.

This is less convenient for local users who expect environment variables to be
picked up automatically, and credentials must be re-entered after restart,
logout, browser session reset, or TTL expiry. The trade-off is intentional
because the same Assistant code must support local and hosted use without
treating server environment secrets as user-scoped app secrets.
