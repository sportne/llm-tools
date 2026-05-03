# ADR 0006: Split Available and Loaded Skill Context

## Status

Accepted

## Context

Codex-style skills use progressive disclosure. The model can first see a compact
catalog of available skills, then receive the full `SKILL.md` body only for a
skill that was invoked or selected for the current turn.

`llm-tools` already keeps provider message assembly outside the low-level
parsing and validation layers. If `skills_api` emitted provider-specific message
roles directly, it would couple portable skill mechanics to app and workflow
prompt construction.

## Decision

Represent skill prompt contributions as two structured context shapes:

- **Available Skills Context**: budgeted metadata for discovered and enabled
  skills, rendered by default as a Codex-style `## Skills` block.
- **Loaded Skill Context**: the selected skill's name, path, and full original
  `SKILL.md` contents, rendered by default in a Codex-compatible `<skill>`
  envelope.

`skills_api` emits these context values and renderer defaults, but workflow,
harness, and app consumers decide how to place them into provider messages.

## Consequences

Skill support preserves progressive disclosure while keeping message-role and
prompt-assembly choices with the model-turn and assistant-runtime layers.

Available skill metadata can be budgeted, warned about, and inspected without
loading full instruction bodies.

Loaded skill bodies remain turn-scoped by default, and durable records can store
usage metadata or content hashes without copying full `SKILL.md` text into
transcripts unless a later replay feature explicitly requires that.
