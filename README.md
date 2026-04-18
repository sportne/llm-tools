# llm-tools

`llm-tools` is a low-level Python library for defining, validating, registering,
executing, and exposing typed tools for LLM and non-LLM applications.

The project started as a strict tool substrate and is now evolving toward a
full agent framework. Today, the strongest implemented layers are still the
typed tool/runtime foundations plus a thin one-turn integration bridge, but the
longer-term direction now includes richer agent capabilities on top of that
core.

## Status

The core v0.1 foundation is implemented:

- canonical tool and runtime models
- registry, runtime, policy, and observability
- built-in filesystem, git, GitLab, Atlassian, and text tools
- built-in read-only repository chat tools
- one canonical structured-action adapter (`ActionEnvelopeAdapter`)
- an Instructor-backed OpenAI-compatible provider layer
- a current `workflow_api` bridge for one parsed model turn plus an interactive
  repository-chat session runner
- dual sync/async execution paths across runtime, provider, and workflow layers
- optional app layers with Textual and Streamlit repository chat clients plus a
  developer-facing Textual workbench
- `harness_api` durable session orchestration with persisted traces, replay,
  summaries, a public Python session API, and a minimal persisted-session CLI

## Core Concepts

- tools are Python classes
- structured data uses Pydantic v2
- `ToolSpec` plus `input_model` and `output_model` are canonical
- `ToolRuntime` executes one invocation with validation and normalization
- adapters expose tools and parse model output into canonical turn results
- providers handle model requests through OpenAI-compatible endpoints
- `WorkflowExecutor` currently bridges one parsed model turn into sequential
  tool execution
- the optional workbench app helps inspect those layers interactively

## Package Layout

The library uses a `src` layout rooted at `src/llm_tools/`.

```text
src/llm_tools/
  apps/
  tool_api/
  llm_adapters/
  llm_providers/
  tools/
  workflow_api/
  harness_api/
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

To install the optional Textual apps:

```bash
~/.venvs/llm-tools/bin/python -m pip install -e .[apps]
```

To install the optional Streamlit chat app:

```bash
.venv/bin/python -m pip install -e .[streamlit]
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

- [Design Docs](docs/design/README.md)
- [Usage Docs](docs/usage/README.md)
- [Implementation Docs](docs/implementation/README.md)
- [Extension Docs](docs/extensions/README.md)
- [Agent Conventions](AGENTS.md)

## Workbench

Launch the optional Textual workbench with either:

```bash
python -m llm_tools.apps.textual_workbench
```

or:

```bash
llm-tools-workbench
```

## Chat App

Launch the optional Textual repository chat app with either:

```bash
python -m llm_tools.apps.textual_chat <directory> --config <path>
```

or:

```bash
llm-tools-chat <directory> --config <path>
```

## Harness Sessions

Launch the minimal persisted harness CLI with either:

```bash
python -m llm_tools.apps.harness_cli start --title "Task" --intent "Do work"
```

or:

```bash
llm-tools-harness start --title "Task" --intent "Do work"
```

Use the public Python session API from `llm_tools.harness_api` when you need
injectable session control, replay inspection, or a minimal built-in runner for
scripted and approval-aware harness tests.

## Streamlit Chat App

Launch the optional Streamlit repository chat app with either:

```bash
python -m llm_tools.apps.streamlit_chat <directory> --config <path>
```

or:

```bash
llm-tools-streamlit-chat <directory> --config <path>
```

The wrapper accepts only the chat app's own arguments. To pass regular
Streamlit server flags, run Streamlit directly:

```bash
streamlit run src/llm_tools/apps/streamlit_chat/app.py -- <directory> --config <path>
```

## Examples

- [Examples Overview](examples/README.md)
- `examples/minimal_tool.py`
- `examples/builtins_direct.py`
- `examples/openai_wiring.py`
- `examples/async_model_turn.py`
- `examples/openai_live.py`
- `examples/structured_response.py`
- `examples/prompt_schema.py`
