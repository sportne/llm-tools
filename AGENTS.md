# AGENTS.md

## Purpose

This repository builds `llm-tools`, a typed Python library for tool definition,
validation, registration, execution, and exposure, with workflow, harness, and
assistant layers built on top of that substrate.

The project now aims to grow into an agent framework over time. Keep the
foundational tool/runtime abstractions clean and reusable, but do not treat
planning, memory, workflow orchestration, or prompt-management as categorically
out of scope for the repository.

## Package Structure

Use the `src` layout rooted at `src/llm_tools/`.

Primary subsystems live under:

- `src/llm_tools/tool_api`
- `src/llm_tools/llm_adapters`
- `src/llm_tools/llm_providers`
- `src/llm_tools/tools`
- `src/llm_tools/workflow_api`
- `src/llm_tools/harness_api`
- `src/llm_tools/apps`

These subsystem names should stay consistent with the specification and
architecture documents.

## Tooling Conventions

- Use `ruff` for formatting and linting.
- Use `mypy` for static type checking.
- Use `pytest` and `pytest-cov` for testing and coverage reporting.
- Prefer `make` targets for common local workflows.
- Treat Vulture findings as review leads, not ground truth. Before deleting
  code or adding a whitelist entry, check whether the symbol is part of a
  public API, protocol/interface surface, framework callback, generated/runtime
  lookup path, or test-only coverage of an intentional extension point.

Primary commands:

```bash
make setup-venv
make install-dev
make format
make lint
make typecheck
make dead-code
make reachability
make test
make coverage
make package
```

## Dependency Guidance

- Keep runtime dependencies minimal where practical.
- Prefer simplicity and explicitness over convenience abstractions.
- Treat bundled remote integrations and assistant-facing app code as supported
  repository scope, but avoid pushing their concerns down into the lowest layers
  by accident.

## Documentation Guidance

- `docs/system-design.md` is the canonical system design, including subsystem
  boundaries, dependency direction, architectural constraints, security design,
  LLM interaction design, persistence design, built-in tool behavior, and
  dependency inventory.
- `docs/assistant-user-guide.md` is the canonical end-user guide for operating
  the Assistant app.
- `docs/CONTEXT.md` defines the project domain language and should be updated
  when new durable concepts or documentation conventions are introduced.
- `docs/adr/` records architectural decisions and rationale. ADRs may be
  referenced from design work, but they should not replace updates to the
  canonical system design when implemented behavior changes.
- `TASKS.md` is the active implementation checklist and should reflect actual
  repository status.

If those documents diverge, resolve conservatively in favor of implemented and
tested behavior, then update the docs.

## Agent skills

### Issue tracker

Issues and PRDs are tracked as local markdown files under `tasks/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Triage uses the default canonical labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, and `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

This repo uses a single-context domain doc layout with `docs/CONTEXT.md` and `docs/adr/`. See `docs/agents/domain.md`.
