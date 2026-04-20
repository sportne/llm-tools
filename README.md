# llm-tools

`llm-tools` is a typed Python toolkit for defining, validating, exposing, and
executing tools, with higher-level workflow, harness, and assistant surfaces
built on top of that substrate.

The repository is broader than a minimal tool substrate. The current supported
floor includes:

- typed tool/runtime foundations in `tool_api`
- model-output normalization in `llm_adapters`
- OpenAI-compatible transport in `llm_providers`
- one-turn execution primitives in `workflow_api`
- durable orchestration in `harness_api`
- a Streamlit assistant and a persisted harness CLI in `apps`
- bundled built-in tools for local files, Git, text search, GitLab, and
  Atlassian products

When docs drift, current code and tests are the source of truth. The surviving
design and security docs are linked below.

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

## Setup

```bash
make setup-venv
make install-dev
```

The default development environment is shared across the main checkout and any
git worktrees at `~/.venvs/llm-tools`. Re-run `make install-dev` from the
checkout you are actively using so the shared environment's editable install
points at that tree.

The optional Streamlit runtime is included in `.[dev]`. If you want only the
assistant runtime extras:

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

## Main Entry Points

Supported library surfaces:

- `llm_tools.tool_api`
- `llm_tools.llm_adapters`
- `llm_tools.llm_providers`
- `llm_tools.tools`
- `llm_tools.workflow_api`
- `llm_tools.harness_api`

Supported product entrypoints:

- `llm_tools.apps.streamlit_assistant`
- `llm_tools.apps.harness_cli`

`apps/*` are supported product surfaces, but they are not the default extension
API for downstream consumers.

## Common Usage

### Define and run a tool

Use `Tool`, `ToolSpec`, `ToolRegistry`, and `ToolRuntime` for direct typed tool
work. Concrete tools define canonical metadata plus input and output models, and
`ToolRuntime` owns validation, policy checks, invocation, and normalized
results.

```python
from pydantic import BaseModel

from llm_tools.tool_api import Tool, ToolExecutionContext, ToolRegistry, ToolRuntime, ToolSpec


class EchoInput(BaseModel):
    value: str


class EchoOutput(BaseModel):
    echoed: str


class EchoTool(Tool[EchoInput, EchoOutput]):
    spec = ToolSpec(name="echo", description="Return the provided value.")
    input_model = EchoInput
    output_model = EchoOutput

    def _invoke_impl(
        self, context: ToolExecutionContext, args: EchoInput
    ) -> EchoOutput:
        return EchoOutput(echoed=f"{context.invocation_id}:{args.value}")


registry = ToolRegistry()
registry.register(EchoTool())
runtime = ToolRuntime(registry)
```

### Run one model turn

The canonical one-turn flow is:

1. Build a typed action envelope through `ActionEnvelopeAdapter`.
2. Ask a provider for one structured response.
3. Execute the parsed tool invocations through `WorkflowExecutor`.

```python
from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import ToolContext
from llm_tools.workflow_api import WorkflowExecutor

adapter = ActionEnvelopeAdapter()
executor = WorkflowExecutor(registry)
provider = OpenAICompatibleProvider.for_ollama(model="gemma4:26b")

context = ToolContext(invocation_id="turn-1")
prepared = executor.prepare_model_interaction(adapter, context=context)

parsed = provider.run(
    adapter=adapter,
    messages=[{"role": "user", "content": "Read README.md"}],
    response_model=prepared.response_model,
)
turn_result = executor.execute_parsed_response(parsed, context)
```

`workflow_api` remains one-turn scoped even though it also exports
assistant-facing chat and protection helpers.

### Run a durable harness session

`harness_api` builds on `workflow_api` for persisted multi-turn work:

```python
from llm_tools.harness_api import (
    BudgetPolicy,
    HarnessSessionCreateRequest,
    HarnessSessionRunRequest,
    HarnessSessionService,
    InMemoryHarnessStateStore,
    ScriptedParsedResponseProvider,
)
from llm_tools.llm_adapters import ParsedModelResponse
from llm_tools.tool_api import ToolPolicy, ToolRegistry
from llm_tools.workflow_api import WorkflowExecutor

registry = ToolRegistry()
workflow_executor = WorkflowExecutor(registry, policy=ToolPolicy())
service = HarnessSessionService(
    store=InMemoryHarnessStateStore(),
    workflow_executor=workflow_executor,
    provider=ScriptedParsedResponseProvider(
        [ParsedModelResponse(final_response="done")]
    ),
    workspace=".",
)

created = service.create_session(
    HarnessSessionCreateRequest(
        title="Root task",
        intent="Complete the request.",
        budget_policy=BudgetPolicy(max_turns=3),
    )
)
result = service.run_session(HarnessSessionRunRequest(session_id=created.session_id))
```

### Run the shipped apps

Launch the Streamlit assistant:

```bash
python -m llm_tools.apps.streamlit_assistant <directory> --config <path>
```

or:

```bash
llm-tools-streamlit-assistant <directory> --config <path>
```

Launch the persisted harness CLI:

```bash
python -m llm_tools.apps.harness_cli start --title "Task" --intent "Do work"
```

or:

```bash
llm-tools-harness start --title "Task" --intent "Do work"
```

The Streamlit assistant is the main interactive client. The harness CLI is a
minimal operational surface over the public `harness_api` session service.

## Dependency Surface

The base package currently includes:

- `openai` and `instructor` for provider transport and structured responses
- `atlassian-python-api` and `python-gitlab` for bundled enterprise read tools
- `markitdown` and `mpxj` for document-conversion support in read-oriented local
  tools

These integrations are not equally central. The local assistant core is mainly
filesystem, Git, and text tools; the remote integrations are bundled but
secondary.

## Documentation

- [Specification](docs/design/spec.md)
- [Architecture](docs/design/architecture.md)
- [Harness Architecture](docs/design/harness_api.md)
- [Security](docs/security.md)
- [Examples](examples/README.md)
- [Agent Conventions](AGENTS.md)
