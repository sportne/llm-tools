# llm-tools

`llm-tools` is a typed Python toolkit for defining, validating, exposing, and
executing tools, with higher-level workflow, harness, and assistant surfaces
built on top of that core.

The repository is broader than a minimal tool substrate. The current product
floor includes:

- typed tool/runtime foundations in `tool_api`
- one-turn execution primitives in `workflow_api`
- durable research/workflow orchestration in `harness_api`
- a Streamlit assistant client with normal chat and durable research sessions
- bundled built-in tools for local files, Git, text search, GitLab, and
  Atlassian products

When repository docs drift, current code and tests are the primary source of
truth. The current scope audit and cleanup backlog live in
[docs/implementation/scope-audit.md](docs/implementation/scope-audit.md).

## Current Status

The implemented codebase currently spans these main layers:

- `tool_api`: canonical typed models, runtime, policy, registry, and execution
  services
- `llm_adapters`: structured model-output normalization
- `llm_providers`: OpenAI-compatible typed provider transport
- `tools`: bundled built-in tool implementations
- `workflow_api`: one-turn execution plus assistant-oriented chat/protection
  helpers
- `harness_api`: durable multi-turn orchestration, replay, resume, and
  verification
- `apps`: product entrypoints and app-local glue

The main supported product entrypoints are:

- `llm_tools.apps.streamlit_assistant`
- `llm_tools.apps.harness_cli`

`apps/*` are supported product surfaces, but they should not be treated as the
default extension API for downstream library consumers.

## Package Layout

```text
src/llm_tools/
  apps/
  harness_api/
  llm_adapters/
  llm_providers/
  tool_api/
  tools/
  workflow_api/
```

## Quick Start

```bash
make setup-venv
make install-dev
```

The default development environment is shared across the main checkout and any
git worktrees at `~/.venvs/llm-tools`. Re-run `make install-dev` from the
checkout you are actively using so the shared environment's editable install
points at that tree.

## Assistant Surfaces

Launch the Streamlit assistant with either:

```bash
python -m llm_tools.apps.streamlit_assistant <directory> --config <path>
```

or:

```bash
llm-tools-streamlit-assistant <directory> --config <path>
```

Launch the persisted harness CLI with either:

```bash
python -m llm_tools.apps.harness_cli start --title "Task" --intent "Do work"
```

or:

```bash
llm-tools-harness start --title "Task" --intent "Do work"
```

The Streamlit assistant is the main interactive client. The harness CLI is a
minimal operational surface over the public `harness_api` session service.

## Dependencies

The base package currently bundles:

- OpenAI-compatible provider dependencies: `openai`, `instructor`
- enterprise read integrations: `atlassian-python-api`, `python-gitlab`
- document-conversion backends: `markitdown`, `mpxj`

These integrations are mostly loaded lazily at runtime, but they are still part
of the packaged dependency surface today. The current rationale and cleanup
follow-ups are tracked in the scope audit.

To install the optional Streamlit runtime:

```bash
~/.venvs/llm-tools/bin/python -m pip install -e .[streamlit]
```

## Development

```bash
make format
make lint
make typecheck
make test
make coverage
make package
```

## Documentation

- [Scope Audit](docs/implementation/scope-audit.md)
- [Design Docs](docs/design/README.md)
- [Usage Docs](docs/usage/README.md)
- [Implementation Docs](docs/implementation/README.md)
- [Examples](examples/README.md)
- [Agent Conventions](AGENTS.md)
