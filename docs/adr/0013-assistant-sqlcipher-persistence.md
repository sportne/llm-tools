# ADR 0013: Use Encrypted SQLite for Assistant Persistence

## Status

Accepted

## Context

The LLM Tools Assistant needs durable chat sessions, preferences, workbench
records, local users, browser sessions, and hosted-mode isolation. Earlier
file-per-session storage and browser-managed state would not provide a strong
enough foundation for authenticated local or private-network hosted use.

## Decision

Use SQLCipher-backed SQLite as the durable persistence layer for the Assistant,
with user-owned chat fields encrypted using per-user data keys wrapped by a
local server key.

The Assistant stores provider and tool credentials only in server memory for the
current browser or app session. It does not persist those secrets in SQLite,
config files, browser storage, or provider presets. The SQLite database is
intended to be local to one Assistant server process; multi-user deployments
should centralize access through that server rather than sharing a live database
file across machines.

Legacy JSON-file migration is not part of the v1 persistence boundary.

## Consequences

The app gains durable multi-user session state, local authentication support,
per-user chat isolation, and a clearer security posture for copied database
files.

The cost is a heavier runtime dependency set, operational key-file management,
and an explicit constraint that SQLite is not a cross-machine concurrency
boundary. Losing the database key or user key-wrapping key makes the affected
encrypted data unrecoverable.
