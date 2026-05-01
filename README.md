# llm-tools

`llm-tools` is a typed Python toolkit for defining, validating, exposing, and
executing tools, with workflow, durable harness, and browser assistant surfaces
built on top of that substrate.

The project is no longer just a minimal tool wrapper. It is the foundation for a
local/private-network assistant stack:

- strict tool definitions, validation, policy checks, approval gates, and
  normalized execution results
- provider adapters for native tools, native JSON schemas, and prompt-emitted
  tool calls when a model endpoint does not support tool APIs
- interactive chat turns with staged tool use, grounding, context compaction,
  proprietary-information checks, and structured final responses
- durable harness sessions for longer research/deep-task workflows
- the **LLM Tools Assistant**, a NiceGUI web app with encrypted SQLite
  persistence, local users, admin controls, tool selection, workbench inspection,
  and session-scoped credentials
- bundled tools for local files, Git, text search, GitLab, Jira, Confluence, and
  Bitbucket

The current code and tests are the source of truth when older design notes drift.

## Status

This package is pre-alpha. The lower-level typed tool and workflow APIs are
covered heavily by tests, but app-level UX and orchestration are still evolving.
Breaking changes are expected.

The supported interactive app is now `llm_tools.apps.assistant_app` /
`llm-tools-assistant`. The earlier Streamlit app has been removed.

## Package Layout

```text
src/llm_tools/
  tool_api/       typed tool substrate, runtime, policy, redaction
  llm_adapters/   model-output normalization and prompt-tool protocols
  llm_providers/  OpenAI-compatible provider transport
  tools/          bundled local, Git, text, GitLab, and Atlassian tools
  workflow_api/   one-turn execution, interactive chat, protection
  harness_api/    durable orchestration, replay, approvals, task state
  apps/           Assistant app, harness CLI, shared app runtime helpers
```

Primary public library surfaces:

- `llm_tools.tool_api`
- `llm_tools.llm_adapters`
- `llm_tools.llm_providers`
- `llm_tools.tools`
- `llm_tools.workflow_api`
- `llm_tools.harness_api`

Product entrypoints:

- `llm_tools.apps.assistant_app`
- `llm_tools.apps.harness_cli`

`apps/*` are supported product surfaces, but downstream extensions should prefer
the lower library layers unless they intentionally target app behavior.

## Install

Linux/macOS or WSL:

```bash
make setup-venv
make install-dev
```

Native Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -e ".[dev]"
```

The default development workflow uses the shared virtual environment
`~/.venvs/llm-tools`. Re-run `make install-dev` from the checkout you are
actively using so editable installs point at that tree.

Runtime dependencies include `openai`, `instructor`, `nicegui`, `SQLAlchemy`,
`sqlcipher3-wheels`, `cryptography`, `argon2-cffi`, `pathspec`,
`atlassian-python-api`, `python-gitlab`, `markitdown`, and `mpxj`. The app uses
SQLCipher-backed persistence, so `sqlcipher3-wheels` is required for normal
Assistant use.

## Run The Assistant

Launch the LLM Tools Assistant:

```bash
llm-tools-assistant .
```

or:

```bash
python -m llm_tools.apps.assistant_app .
```

Native Windows PowerShell:

```powershell
.\.venv\Scripts\llm-tools-assistant.exe .
```

Useful options:

```bash
llm-tools-assistant . \
  --config examples/assistant_configs/local-only-chat.yaml \
  --host 127.0.0.1 \
  --port 8080 \
  --provider ollama \
  --model gemma4:26b \
  --api-base-url http://127.0.0.1:11434/v1
```

On first launch the app shows a local admin creation screen. Normal local and
hosted use both use local username/password login. `--auth-mode none` exists
only as an explicit development/test escape hatch.

The Assistant UI supports:

- persistent and temporary chats
- provider/model/base-URL settings and model discovery
- provider modes `auto`, `tools`, `json`, and `prompt_tools`
- grouped tool selection with per-tool capability status
- session permissions and approval policies
- session-only provider API keys and tool credentials
- local workspace selection
- encrypted SQLite persistence
- per-user chat isolation
- automatic context compaction for long conversations
- structured final-answer details such as citations, confidence, uncertainty,
  missing information, and follow-up suggestions
- an inspector/workbench with provider messages, parsed responses, tool events,
  and timing metadata
- optional proprietary-information protection settings and challenge dialogs
- an admin screen for local user management and app branding
- an admin-gated Deep Task mode backed by the harness layer

The app stores durable state under `~/.llm-tools/assistant/nicegui/` by default.
You can choose another database path from Settings or with `--db-path`.

### Local And Hosted Use

For local use, run one app process and connect through the browser.

For hosted use, run one server process and put it behind a TLS-terminating
reverse proxy such as Caddy, nginx, or Traefik:

```bash
llm-tools-assistant \
  --host 0.0.0.0 \
  --auth-mode local \
  --public-base-url https://assistant.example.internal
```

Direct certificate files are available for bootstrap or self-signed testing:

```bash
llm-tools-assistant --tls-certfile cert.pem --tls-keyfile key.pem
```

If the app is reachable over non-loopback HTTP, secret entry is blocked unless
`--allow-insecure-hosted-secrets` is passed as an explicit risk acceptance.

Do not run multiple local Assistant app instances on different machines against
the same SQLite database file on a network drive. SQLCipher-backed SQLite with
WAL-style file coordination is not a supported multi-machine database model. For
multi-user use, centralize database access in one Assistant server process and
have users connect through the browser.

## Assistant Persistence And Secrets

The Assistant always uses encrypted persistence:

- the SQLite database is opened through SQLCipher
- the SQLCipher database key is stored outside SQLite
- user-owned chat fields are encrypted again with per-user data keys wrapped by
  a local server key
- passwords are stored as Argon2 hashes
- browser sessions use hashed server-side session tokens

Default key locations:

```text
~/.llm-tools/assistant/nicegui/hosted/db.key
~/.llm-tools/assistant/nicegui/hosted/user-kek.key
```

Back up these key files separately. Losing them makes encrypted Assistant data
unrecoverable.

Provider API keys and third-party tool credentials are session-only. They are
typed into the app, kept in server memory for the current browser/app session,
and are not written to SQLite or exported config. Non-secret service URLs, such
as OpenAI-compatible base URLs or Atlassian/GitLab base URLs, are normal runtime
configuration and may be persisted.

The provider Base URL fields include a local copy helper with common
OpenAI-compatible endpoint URLs. Deployment-specific builds can tailor the
baked-in list in
`src/llm_tools/apps/assistant_app/provider_endpoints.py`; the app does not fetch
endpoint suggestions from the internet at runtime.

## Provider Modes

`OpenAICompatibleProvider` supports these mode strategies:

- `tools`: native tool/function calling where the endpoint supports it
- `json`: native structured JSON/schema responses
- `prompt_tools`: raw text completions with a fenced prompt-tool protocol
- `auto`: try native structured paths first and fall back to prompt tools for
  parse/capability failures where appropriate

For agentic chat workflows, JSON mode uses staged one-action rounds: choose one
tool or finalize, validate the selected tool arguments, execute, repeat, then
produce a final response. This is more reliable for local models than asking for
a full multi-action envelope in one completion.

`prompt_tools` is the fallback for endpoints that can only return plain chat
text. It uses fenced blocks that parse into the same internal
`ParsedModelResponse` and `ToolInvocationRequest` path, so policy, approvals,
redaction, execution, and tool-result handling remain shared.

## Built-In Tools

The Assistant registry includes:

- `filesystem`: list, inspect, read, and write local workspace files
- `text`: literal text search
- `git`: non-interactive status, diff, and log helpers
- `gitlab`: project, file, merge-request, and code-search reads
- `atlassian`: Jira, Confluence, and Bitbucket read tools

Local discovery and search tools treat dot-hidden paths and `.gitignore`-ignored
paths as hidden by default. `include_hidden=True` reveals both hidden and ignored
paths for discovery/search. Direct `read_file` remains explicit access and is
not changed by that hidden-discovery policy.

Filesystem tool paths are workspace-relative and use POSIX-style separators for
model-visible paths on every OS. Native Windows paths are supported in the UI
for settings and workspace/database selection, but tool calls should use
workspace-relative paths such as `src/llm_tools/__init__.py`.

## Context Compaction

Interactive chat keeps the full transcript for the UI, but provider calls use a
bounded model-visible context. When the configured `max_context_tokens` limit is
exceeded, older completed turns are compacted into a durable summary and future
provider calls include that summary after the system prompt. If a provider
rejects a request as too large, the chat runner compacts more aggressively and
retries once.

Compaction does not delete transcript history. It updates
`ChatSessionState.context_summary` and emits a short transcript notice so the
user can see that older context was summarized.

## Deep Task And Harness

The normal chat mode is optimized for conversational turns with bounded tool
rounds. Deep Task mode is backed by `harness_api` and is intended for longer,
durable work with task state, budget policies, replay/inspection artifacts, and
approval resume behavior. It is hidden behind an admin feature flag by default.

The lower-level harness CLI remains available:

```bash
llm-tools-harness start --title "Task" --intent "Do work"
llm-tools-harness list
llm-tools-harness inspect <session-id> --json
```

The harness CLI stores state under `~/.llm-tools/harness` by default and exposes
the public `harness_api` session service directly.

## Library Examples

### Define and run a tool

```python
from pydantic import BaseModel

from llm_tools.tool_api import (
    Tool,
    ToolContext,
    ToolExecutionContext,
    ToolInvocationRequest,
    ToolRegistry,
    ToolRuntime,
    ToolSpec,
)


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
result = runtime.execute(
    ToolInvocationRequest(tool_name="echo", arguments={"value": "hello"}),
    ToolContext(invocation_id="example-1"),
)
```

### Run one model turn

The direct action-envelope path remains useful for small extraction-style turns:

```python
from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import ToolContext
from llm_tools.workflow_api import WorkflowExecutor

adapter = ActionEnvelopeAdapter()
executor = WorkflowExecutor(registry)
provider = OpenAICompatibleProvider.for_ollama(model="gemma4:26b")

context = ToolContext(invocation_id="turn-1", workspace=".")
prepared = executor.prepare_model_interaction(adapter, context=context)

parsed = provider.run(
    adapter=adapter,
    messages=[{"role": "user", "content": "Read README.md"}],
    response_model=prepared.response_model,
)
turn_result = executor.execute_parsed_response(parsed, context)
```

For agentic chat or local Ollama workflows, prefer the Assistant or
`ChatSessionTurnRunner` staged paths rather than direct full-envelope
completions.

### Run a durable harness session

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

More examples live under `examples/`.

## Configuration Presets

Reusable Assistant YAML presets live in `examples/assistant_configs/`:

- `local-only-chat.yaml`
- `enterprise-data-chat.yaml`
- `harness-research-chat.yaml`

Launch with:

```bash
llm-tools-assistant . --config examples/assistant_configs/local-only-chat.yaml
```

Config files should contain non-secret defaults only. The app can export a
non-secret preset from the current session; provider API keys and tool
credentials are intentionally omitted.

## Development

Linux/macOS or WSL:

```bash
make format
make lint
make typecheck
make test
make coverage
make package
```

Native Windows PowerShell equivalents:

```powershell
.\.venv\Scripts\python -m ruff format src tests
.\.venv\Scripts\python -m ruff check src tests
.\.venv\Scripts\python -m mypy src
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m build
```

Focused live backend probes are available under `scripts/e2e_assistant/`. For
example, with local Ollama:

```bash
python scripts/e2e_assistant/run_ollama_agent_eval.py \
  --workspace . \
  --models qwen3.5:9b \
  --provider-modes json,prompt_tools
```

## Documentation

- [Specification](docs/design/spec.md)
- [Architecture](docs/design/architecture.md)
- [Assistant App Design](docs/design/assistant_app.md)
- [SQLite Persistence Design](docs/design/sqlite_persistence.md)
- [Harness Architecture](docs/design/harness_api.md)
- [Security](docs/security.md)
- [Examples](examples/README.md)
- [Agent Conventions](AGENTS.md)
