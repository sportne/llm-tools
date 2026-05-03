# ADR 0004: Add a Reusable Skills API Layer

## Status

Accepted

## Context

`llm-tools` is adding Codex-style **Skills** as reusable agent instruction
packages. A skill is not an executable **Tool**, an app-only prompt snippet, or
a plugin. It is a local `SKILL.md` package that can teach an agent a repeatable
workflow and may reference supporting files.

Without a dedicated layer, skill parsing and validation would likely be owned
by the assistant app or harness code first. That would make the canonical skill
format harder to reuse from future workflow, harness, and agent-framework
surfaces, and could blur skill concerns with tool execution or app UI.

## Decision

Create `llm_tools.skills_api` as the canonical reusable library surface for
local `SKILL.md` packages.

The layer owns:

- discovery from caller-provided local roots;
- portable `SKILL.md` metadata parsing and validation;
- skill enablement-aware resolution by name or path;
- loading selected skill instructions;
- structured available-skill and loaded-skill context contributions.

`skills_api` may reference tool names or specs as metadata, but it does not
execute tools, run scripts, install dependencies, own app UI, persist
enablement choices, or perform heuristic skill selection.

## Consequences

Workflow, harness, and app layers can consume one shared skill model without
owning the canonical package format.

The tool runtime stays focused on typed executable capabilities, while skills
remain instruction packages that are still subject to existing tool, filesystem,
shell, network, and credential policy when they guide agent behavior.

Assistant app support can add management and invocation UI later by composing
`skills_api` rather than reimplementing discovery, validation, and resolution.
