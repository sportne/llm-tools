# Implement Core Skills API

Status: completed

## Goal

Add the reusable `llm_tools.skills_api` surface for local `SKILL.md` packages.

## Scope

- Add `src/llm_tools/skills_api` as a supported library surface.
- Model skill discovery, validation, resolution, enablement, loaded-skill
  context, available-skills context, usage metadata, and errors.
- Discover local files named exactly `SKILL.md` under caller-provided roots with
  bounded recursion, hidden-directory skipping, canonical-path dedupe, and no
  symlink following by default.
- Parse required YAML frontmatter fields `name` and `description`.
- Return valid skill metadata and path-specific validation errors without
  failing the whole discovery result.
- Resolve skills by explicit path or plain name with enablement, scope
  precedence, and same-scope ambiguity errors.
- Load selected `SKILL.md` bodies into structured loaded-skill context.
- Render budgeted available-skills context from metadata.
- Add architecture tests so `skills_api` depends only downward and is not tied
  to `workflow_api`, `harness_api`, `apps`, or bundled `tools`.
- Include one or two opt-in, lowest-precedence bundled example skills.

## Out Of Scope

- Automatic skill selection.
- Assistant app UI.
- Codex `agents/openai.yaml` sidecar metadata.
- Plugin-provided skills.
- Remote skill catalogs, installers, or marketplaces.
- Dependency installation or prompting.
- Script execution.

## Acceptance Criteria

- Consumers can discover, validate, resolve, load, and render local skills
  through public `skills_api` imports.
- Invalid skills produce path-specific errors while other skills still load.
- Disabled skills are excluded from resolution and available-skills context
  unless a caller explicitly opts into an administrative override.
- Same-name skills at the same scope produce an ambiguity error for plain-name
  invocation.
- Tests cover discovery, validation, resolution, enablement, context rendering,
  scan limits, and architecture boundaries.

## References

- [Feature PRD](../PRD.md)
- [ADR 0004](../../../docs/adr/0004-skills-api-layer.md)
- [ADR 0005](../../../docs/adr/0005-skill-context-contributions.md)
- [Context](../../../docs/CONTEXT.md)
