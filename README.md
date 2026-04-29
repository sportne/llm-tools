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
- a LLM Tools Assistant and a persisted harness CLI in `apps`
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

The base package includes the current LLM Tools Assistant runtime and its
config-loading dependencies. `.[dev]` adds the toolchain used for local
development and CI.

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

- `llm_tools.apps.assistant_app`
- `llm_tools.apps.harness_cli`

`apps/*` are supported product surfaces, but they are not the default extension
API for downstream consumers. The LLM Tools Assistant is the primary interactive
product surface for internal deployments; the harness CLI is the lower-level
operator surface for durable session management.

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

This direct `provider.run(...)` path stays useful for small extraction-style
turns. For local Ollama agentic workflows, prefer the chat or harness surfaces
with `provider_mode_strategy: json` or `auto`; those paths use staged JSON
schema rounds for decision, one selected tool call, and final response instead
of asking the model for a full action envelope in one completion. If a local
model cannot satisfy native or markdown JSON reliably, `auto` can fall through
to the prompt-tool protocol.

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

Launch the LLM Tools Assistant:

```bash
python -m llm_tools.apps.assistant_app <directory>
```

or with an optional reusable YAML preset:

```bash
llm-tools-assistant <directory> --config <path>
```

The config YAML is optional. The NiceGUI UI can also generate and save a
non-secret config preset from the current session. Remote integration
credentials stay session-only and are never written into the exported YAML.

Launch the persisted harness CLI:

```bash
python -m llm_tools.apps.harness_cli start --title "Task" --intent "Do work"
```

or:

```bash
llm-tools-harness start --title "Task" --intent "Do work"
```

The LLM Tools Assistant is the main interactive client for private-network
deployments. It supports explicit provider-mode selection for OpenAI-compatible
endpoints, per-integration remote credentials entered in the UI, and optional
proprietary-protection corpus paths that can be saved as part of exported
config. The harness CLI remains an operational surface over the public
`harness_api` session service.

The assistant settings include a local, copyable list of common
OpenAI-compatible provider Base URLs. Deployment-specific builds can tailor that
list in `src/llm_tools/apps/assistant_app/provider_endpoints.py`; the app does
not fetch endpoint suggestions at runtime.

## Dependency Surface

The base package currently includes:

- `openai` and `instructor` for provider transport and structured responses
- `atlassian-python-api` and `python-gitlab` for bundled enterprise read tools
- `markitdown` and `mpxj` for document-conversion support in read-oriented local
  tools

These integrations are supported product scope. The LLM Tools Assistant treats
local workspace, GitLab, Jira, Confluence, and Bitbucket as first-class source
groups, with readiness and credential requirements surfaced explicitly in the
UI.

## Documentation

- [Specification](docs/design/spec.md)
- [Architecture](docs/design/architecture.md)
- [Harness Architecture](docs/design/harness_api.md)
- [Security](docs/security.md)
- [Examples](examples/README.md)
- [Agent Conventions](AGENTS.md)
