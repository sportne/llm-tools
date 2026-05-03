# ADR 0033: Use NiceGUI for the Assistant App

## Status

Accepted

## Context

The Assistant is a supported product surface inside `llm-tools`, not only an
example client. It needs local and private-network browser access, background
turn coordination, session persistence, authentication screens, admin controls,
provider settings, approval UI, and inspector/workbench views while staying
close to the Python runtime layers it composes.

## Decision

Use NiceGUI as the supported Assistant app UI and runtime framework.

The Assistant remains a Python-first app surface that can directly compose the
tool, workflow, harness, provider, protection, and persistence layers without a
separate frontend/backend product architecture. Other consumers may still build
different interfaces on the lower library layers, but the packaged Assistant app
targets NiceGUI.

## Consequences

The repository can ship a usable local and hosted browser Assistant without
maintaining a separate JavaScript frontend stack or remote API boundary first.

The cost is a heavier runtime dependency and a less UI-framework-neutral app
surface. That is accepted because the Assistant is intended to be an integrated
Python product surface for the framework, not just a sample UI.
