# Add Assistant Skills UI

Status: ready-for-agent

## Goal

Add assistant app management and invocation UI for skills after the core
`skills_api` exists.

## Scope

- Compose session-specific skill roots, skill enablement, available-skill
  context, and loaded-skill context through Assistant Runtime Assembly.
- Persist user-visible skill enablement choices in the assistant app or
  embedding product, while passing the resolved enablement state into
  `skills_api`.
- Add a Skills management surface that shows name, description, scope/source,
  enabled state, path, and validation errors.
- Support `$skill-name` or picker-based skill invocation from the composer.
- Inject available-skills context and loaded-skill context into Assistant Chat
  and Deep Task using the context contribution shape from `skills_api`.
- Record skill usage metadata for turns without persisting full `SKILL.md`
  bodies by default.
- Expose turn-visible available skills, loaded skills, usage records, and skill
  validation or enablement warnings in the inspector/debug surface.

## Out Of Scope

- Automatic skill selection.
- Pinned skills across a harness session.
- Codex sidecar metadata.
- Remote catalogs or installation flows.
- Dependency installation or prompting.

## Acceptance Criteria

- Users can view available skills and enable or disable individual skills.
- Disabled skills are not available to manual invocation or future automatic
  selection through the normal resolver path.
- Users can invoke an enabled skill from the composer.
- Skill usage metadata is durable enough for audit/debug without duplicating the
  full skill body in transcripts by default.
- Inspector/debug UI shows which skill metadata was visible and which skill
  bodies were loaded for a turn.

## References

- [Feature PRD](../PRD.md)
- [ADR 0004](../../../docs/adr/0004-skills-api-layer.md)
- [ADR 0005](../../../docs/adr/0005-skill-context-contributions.md)
- [Assistant app design](../../../docs/design/assistant_app.md)
- [Context](../../../docs/CONTEXT.md)
