# Harness Sessions

`llm_tools.harness_api` exposes a durable multi-turn execution surface above the
one-turn `workflow_api` layer.

Public entry points include:

- `HarnessExecutor` for lower-level durable orchestration
- `HarnessSessionService` for typed create/run/resume/inspect/list/stop flows
- `DefaultHarnessTurnDriver` and `MinimalHarnessTurnApplier` as built-in
  defaults for simple sessions, tests, and minimal CLI usage
- replay and summary helpers built from canonical state, with stored artifacts
  treated as derived caches rather than the source of truth

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

- `HarnessSessionService.run_session_async(...)`
- `HarnessSessionService.resume_session_async(...)`
- `HarnessExecutor.run_async(...)`
- `HarnessExecutor.resume_async(...)`

## Stored Artifacts

Stored snapshots carry derived artifacts alongside canonical state:

- `snapshot.artifacts.trace`: aggregated per-turn observability trace cache
- `snapshot.artifacts.summary`: operator-facing session summary cache
- `snapshot.saved_at`: store save timestamp for recent-session views

Those persisted `trace` and `summary` artifacts are inspection aids, not the
source of truth. Canonical `HarnessState` remains authoritative, and inspect/list
flows rebuild trusted artifacts from canonical state rather than trusting stale
or tampered stored summaries.

Replay is likewise driven from canonical turns. Stored trace data may be
rebuilt or ignored when it is missing, stale, inconsistent, or intentionally
dropped.

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

Newly written pending-approval records preserve only scrubbed base-context
fields:

- preserved: `invocation_id`, `workspace`, and `metadata`
- cleared before persistence: process env, logs, artifacts, and source
  provenance
- rebuilt on resume: execution context derived from the stored base context plus
  the current process environment

Pending approval turns also persist a minimal approval-audit record on the turn
itself so replay and inspection can show the blocked invocation without storing
raw arguments or full request payloads.

Non-approved approval outcomes are fail-closed: denial, expiration, or operator
cancel records the blocked invocation, but later invocations from that same
paused model turn do not continue running.

Older snapshots written before this hardening change may still contain raw
environment data. Delete those persisted files if they may have held secrets.

## Corruption Handling

Malformed file-backed session records are treated as corruption rather than
partially trusted state. File-store list and inspect flows skip or surface those
records cleanly without trusting cached summary or trace data from the damaged
file.
