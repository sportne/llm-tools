# ADR 0025: Use Workspace-Relative POSIX Tool Paths

## Status

Accepted

## Context

Filesystem and search tools need to work across operating systems while keeping
model-visible paths scoped to the selected workspace. Native absolute paths,
Windows separators, drive letters, and user-home details are useful for app
settings and runtime configuration, but they make model-facing tool calls less
portable and harder to validate consistently.

## Decision

Use workspace-relative POSIX-style paths for model-visible filesystem tool
arguments and outputs.

Native absolute paths may be accepted in app settings, workspace selection, and
runtime context, but tool calls should refer to files relative to the configured
workspace using `/` separators. Path validation and policy enforcement should
resolve those paths through the runtime context before touching the host
filesystem.

## Consequences

Tool calls stay portable, easier to inspect, easier to validate, and less likely
to leak host-specific path details.

The cost is explicit translation at app and runtime boundaries, especially on
Windows. That cost is accepted because workspace-relative paths are the safer
and clearer contract for model-generated tool invocations.
