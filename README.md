# llm-tools

`llm-tools` is a low-level Python library for defining, validating, registering,
executing, and exposing typed tools for LLM and non-LLM applications.

The project is intentionally not an agent framework. It focuses on a strict,
class-based tool substrate that higher-level systems can build on later.

## Status

The repository is in foundation setup. Step 0 establishes packaging, tooling,
CI, and the initial package layout. Canonical models, runtime behavior, and
adapters begin in later steps.

## Package Layout

The library uses a `src` layout rooted at `src/llm_tools/`.

```text
src/llm_tools/
  tool_api/
  llm_adapters/
  tools/
  workflow_api/
```

These packages are scaffolded in Step 0 only. No runtime APIs are implemented
yet.

## Development

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

## Documentation

- [Specification](docs/SPEC.md)
- [Architecture](docs/ARCHITECTURE.md)
- [Tasks](docs/TASKS.md)
- [Agent Conventions](AGENTS.md)
