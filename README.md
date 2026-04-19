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
- an optional Streamlit assistant app layer for interactive use
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
- the optional Streamlit assistant app is the long-term interactive client

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

Native Microsoft Project (`.mpp`/`.mpt`) reads now use MPXJ. That dependency is
installed with the base package, and a working Java runtime must be available
wherever those file reads are expected to work.

To install the optional Streamlit apps:

```bash
~/.venvs/llm-tools/bin/python -m pip install -e .[streamlit]
```

## Security and Dependencies

The base package installs provider, remote-service, and document-conversion
components so those tool surfaces are available when enabled: `openai` and
`instructor` for OpenAI-compatible structured responses,
`atlassian-python-api` and `python-gitlab` for enterprise read tools,
`markitdown` for document conversion, and MPXJ plus a working Java runtime for
Microsoft Project reads. See [Security Hardening](docs/usage/security-hardening.md)
for the dependency surface, default storage locations, temp caches, and secret
handling guidance.

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
scripted and approval-aware harness tests. The CLI defaults to storing state in
`~/.llm-tools/harness`; pass `--store-dir` to isolate or relocate persisted
session data.

## Streamlit Assistant App

Launch the optional Streamlit assistant app with either:

```bash
python -m llm_tools.apps.streamlit_assistant <directory> --config <path>
```

or:

```bash
llm-tools-streamlit-assistant <directory> --config <path>
```

The assistant is the broader chat client: it can answer normal questions
without tools, optionally use the full built-in tool registry, and launch
harness-backed research sessions for durable investigation work. Selecting a
workspace root does not automatically enable filesystem or subprocess access;
those permissions stay opt-in in the sidebar.

This is the only long-term interactive client the repository plans to keep.

To pass regular Streamlit server flags, run Streamlit directly:

```bash
streamlit run src/llm_tools/apps/streamlit_assistant/app.py -- <directory> --config <path>
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
