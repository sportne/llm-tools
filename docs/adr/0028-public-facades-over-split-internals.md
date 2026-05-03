# ADR 0028: Preserve Public Facades Over Split Internals

## Status

Accepted

## Context

Several supported surfaces began as larger modules and have since been split
into narrower implementation modules. Examples include workflow chat session and
protection helpers, plus harness executor and session-service internals. If
every implementation split also changed public imports, cleanup refactors would
create avoidable churn for downstream callers.

## Decision

Preserve thin public facade modules for supported surfaces while moving
concentrated implementation behavior into focused internal modules.

Facade modules may re-export public models, services, helpers, and constructors
from split implementation modules. Refactors should treat a facade as a stable
import surface unless the public API is intentionally deprecated or replaced.

## Consequences

Implementation modules can get smaller and more focused without forcing
unnecessary public import changes.

The cost is some indirection when navigating the codebase. That is accepted
because preserving supported import surfaces is more valuable than making every
module boundary map directly to implementation ownership.
