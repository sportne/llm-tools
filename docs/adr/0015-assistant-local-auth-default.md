# ADR 0015: Require Local Assistant Authentication by Default

## Status

Accepted

## Context

The Assistant supports both local loopback use and private-network hosted use.
It stores durable user-owned chat state, accepts provider and tool credentials,
and can expose filesystem, subprocess, and remote read capabilities depending on
session policy. Defaulting to an unauthenticated local app would make the local
and hosted security models diverge and could expose a capable assistant if a
binding or deployment choice changes.

## Decision

Use local username/password authentication as the normal Assistant auth mode,
including loopback use.

The first launch creates an admin user, and later users are admin-created. There
is no public self-registration in v1. `--auth-mode none` remains available only
as an explicit development and test escape hatch, not as the normal local mode.

## Consequences

Local and hosted Assistant deployments share one identity boundary, per-user
session ownership model, and per-user encrypted chat state model.

The cost is extra first-run setup and local user management even for single-user
loopback use. That friction is accepted because the Assistant handles secrets
and side-effect-capable tool exposure, and because hosted behavior should not be
an afterthought bolted onto an unauthenticated local app.
