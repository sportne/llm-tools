# Harness Sessions

`llm_tools.harness_api` exposes a durable multi-turn execution surface above the
one-turn workflow layer.

Public entry points include:

- `HarnessExecutor` for lower-level durable orchestration
- `HarnessSessionService` for typed create/run/resume/inspect/list/stop flows
- `DefaultHarnessTurnDriver` and `MinimalHarnessTurnApplier` as built-in
  defaults
- replay and summary helpers through stored artifacts

## Python API

```python
from llm_tools.harness_api import (
    BudgetPolicy,
    HarnessSessionCreateRequest,
    HarnessSessionInspectRequest,
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
result = service.run_session(
    HarnessSessionRunRequest(session_id=created.session_id)
)
inspection = service.inspect_session(
    HarnessSessionInspectRequest(
        session_id=result.snapshot.session_id,
        include_replay=True,
    )
)
```

Async entry points are also public:

- `run_session_async(...)`
- `resume_session_async(...)`
- `HarnessExecutor.run_async(...)`
- `HarnessExecutor.resume_async(...)`

## Stored Artifacts

Stored snapshots carry derived artifacts alongside canonical state:

- `snapshot.artifacts.trace`
- `snapshot.artifacts.summary`
- `snapshot.saved_at`

## CLI

The minimal CLI is backed by `HarnessSessionService` and supports:

- `start`
- `run`
- `resume`
- `inspect`
- `list`
- `stop`

Examples:

```bash
llm-tools-harness start --title "Task" --intent "Do work"
llm-tools-harness run <session-id> --script scripted-turns.json
llm-tools-harness inspect <session-id> --replay --json
llm-tools-harness list --json
```

By default, CLI state is stored under `~/.llm-tools/harness`.

## Pending Approval Snapshots

Newly written pending-approval records preserve only `invocation_id`,
`workspace`, and `metadata`. Process env data, logs, artifacts, and source
provenance are cleared before persistence. Approval resume rebuilds execution
context from the current process environment at resume time.
