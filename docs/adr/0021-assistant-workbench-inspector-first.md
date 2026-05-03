# ADR 0021: Make the Assistant Workbench an Inspector First

## Status

Accepted

## Context

The Assistant app includes a right-side workbench region and persistent
`workbench_items`. That surface could become a Canvas or Artifacts-style product
area, but v1 also needs practical observability for provider messages, parsed
responses, tool events, approvals, timing metadata, protection behavior, and
deep-task details.

## Decision

Use the Assistant workbench as an inspector and debug shell first, while
reserving the UI region and persistence shape for future artifact-style outputs.

V1 workbench records should focus on inspectable runtime and execution details.
Full artifact editing, export, version browsing, and Canvas-like workflows are
deferred until their product contract is explicit.

## Consequences

The Assistant gets useful operator and developer observability without waiting
for a larger artifact product design.

The cost is that the workbench may look like a future artifact surface before it
can edit or manage artifacts as first-class user content. That ambiguity is
accepted as long as v1 documentation and UI behavior keep inspector/debug use
clear.
