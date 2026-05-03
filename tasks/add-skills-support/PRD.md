# Add Skills Support

## Status

[x] First slice completed
[ ] Second slice planned

## Task

Add Codex-style **Skill** support to `llm-tools` as a reusable agent-instruction
surface.

The first implementation slice implemented:

- add `src/llm_tools/skills_api` as the canonical library surface;
- model `SkillMetadata`, `SkillDiscoveryResult`, `SkillError`, `SkillScope`,
  `SkillRoot`, `SkillEnablement`, and loaded/context contribution types;
- scan configured roots for portable `SKILL.md` packages;
- parse required YAML frontmatter fields `name` and `description`;
- return valid metadata plus path-specific validation errors;
- resolve skills by name or path with enablement, scope precedence, and
  ambiguity rules;
- load selected `SKILL.md` files as structured loaded-skill context;
- render budgeted available-skills context from metadata;
- add architecture tests for `skills_api` dependency direction;
- include one or two opt-in, lowest-precedence bundled example skills.

The first slice should not include automatic skill selection, Codex sidecar
metadata, plugin-provided skills, app UI, or dependency installation/prompting.

The second implementation slice should add assistant UI and management flows for
viewing, enabling, disabling, and invoking skills. Persisted enablement choices
belong to the app or embedding product, while `skills_api` enforces the
caller-supplied enablement state.

## References

- [ADR 0005](../../docs/adr/0005-skills-api-layer.md)
- [ADR 0006](../../docs/adr/0006-skill-context-contributions.md)
- [Architecture](../../docs/design/architecture.md)
- [Context](../../docs/CONTEXT.md)
