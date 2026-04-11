# llm-tools

`llm-tools` is a low-level Python library for defining, validating, registering,
executing, and exposing typed tools for LLM and non-LLM applications.

The project is intentionally not an agent framework. It focuses on a strict,
class-based tool substrate and a thin one-turn integration layer that
higher-level systems can build on later.

## Status

The core v0.1 foundation is implemented:

- canonical tool and runtime models
- registry, runtime, policy, and observability
- built-in filesystem, git, Atlassian, and text tools
- OpenAI, structured-response, and prompt-schema adapters
- a thin `workflow_api` bridge for one parsed model turn

## Core Concepts

- tools are Python classes
- structured data uses Pydantic v2
- `ToolSpec` plus `input_model` and `output_model` are canonical
- `ToolRuntime` executes one invocation with validation and normalization
- adapters expose tools and parse model output into canonical turn results
- `WorkflowExecutor` bridges one parsed model turn into sequential tool
  execution when needed

## Package Layout

The library uses a `src` layout rooted at `src/llm_tools/`.

```text
src/llm_tools/
  tool_api/
  llm_adapters/
  tools/
  workflow_api/
```

## Quick Start

```bash
make setup-venv
make install-dev
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

## Examples

- [Examples Overview](examples/README.md)
- `examples/minimal_tool.py`
- `examples/builtins_direct.py`
- `examples/openai_wiring.py`
- `examples/openai_live.py`
- `examples/structured_response.py`
- `examples/prompt_schema.py`
