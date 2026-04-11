# AGENTS.md

## Purpose

This repository builds `llm-tools`, a low-level Python library for typed tool
definition, validation, registration, execution, and exposure.

The project is intentionally not an agent framework. Avoid introducing planning,
memory, workflow orchestration, or prompt-management concepts into the base tool
abstraction.

## Step 0 Boundaries

Step 0 is limited to repository foundation work:

- package layout and scaffolding
- packaging metadata
- formatter, linter, type checker, test runner, and coverage setup
- CI workflow setup
- documentation alignment

Do not implement canonical models, runtime logic, registry behavior, adapters,
or built-in tools in Step 0 unless a tiny placeholder is required for package
importability.

## Package Structure

Use the `src` layout rooted at `src/llm_tools/`.

Subsystems live under:

- `src/llm_tools/tool_api`
- `src/llm_tools/llm_adapters`
- `src/llm_tools/tools`
- `src/llm_tools/workflow_api`

These subsystem names should stay consistent with the specification and
architecture documents.

## Tooling Conventions

- Use `ruff` for formatting and linting.
- Use `mypy` for static type checking.
- Use `pytest` and `pytest-cov` for testing and coverage reporting.
- Prefer `make` targets for common local workflows.

Primary commands:

```bash
make setup-venv
make install-dev
make format
make lint
make typecheck
make test
make coverage
make package
```

## Dependency Guidance

- Keep runtime dependencies minimal.
- Prefer simplicity and explicitness over convenience abstractions.
- Do not add LLM provider packages during Step 0 unless the docs and task scope
  explicitly require them.

## Documentation Guidance

- `docs/SPEC.md` describes intended behavior and acceptance criteria.
- `docs/ARCHITECTURE.md` describes subsystem boundaries and dependency
  direction.
- `docs/TASKS.md` is the implementation checklist and should reflect actual
  repository status.

If those documents diverge, resolve conservatively in favor of the existing Step
scope and note the adjustment in docs.
