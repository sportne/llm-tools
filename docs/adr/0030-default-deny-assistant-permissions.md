# ADR 0030: Default Assistant Sessions to Denied Tool Permissions

## Status

Accepted

## Context

`llm-tools` intentionally bundles local, subprocess, and remote enterprise tool
families so common assistant deployments need fewer external extensions. If
every bundled capability were active by default, an inclusive distribution would
also become a permissive runtime posture.

## Decision

Start Assistant sessions with filesystem, subprocess, and network permissions
off, and require explicit session policy, tool exposure, credentials, and
approvals before side-effect-capable work can run.

Tools may be present in the registry without being exposed to the model for the
current session. Missing permissions or credentials should block the affected
tool exposure or invocation through the normal runtime and policy path rather
than relying on UI hiding alone.

## Consequences

The project can ship a broad set of built-in capabilities without granting them
to every chat session by default.

The cost is first-run friction: users must deliberately enable the capabilities
they want and approve sensitive actions. That friction is accepted because the
Assistant can operate on local files, subprocesses, remote services, and hosted
browser sessions.
