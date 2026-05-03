# ADR 0033: Treat Tested Implementation as Documentation Truth

## Status

Accepted

## Context

`llm-tools` has design docs, architecture notes, security notes, task files,
tests, and implemented product surfaces. As the repository evolves, older docs
can lag behind code or describe an earlier scaffold. Future contributors and
agents need a consistent way to resolve those disagreements.

## Decision

When documentation and implementation disagree, treat tested implemented
behavior as the primary source of truth.

Product entrypoints and public exports break ties when docs lag. Documented
intent wins only when the code is clearly transitional, duplicated, or
inconsistent. When drift is found, update the docs or ADRs to match the
implemented and tested behavior rather than preserving stale aspirational
diagrams.

## Consequences

Architecture documentation stays anchored to the working system rather than to
obsolete plans.

The cost is that aspirational docs have less authority unless they are clearly
marked as future intent. That is accepted because the repository is being
maintained through tests, supported entrypoints, and iterative refactors, and
the docs should help contributors understand the system they actually have.
