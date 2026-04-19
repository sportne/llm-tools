# Harness Sessions

`llm_tools.harness_api` now exposes a public session surface for durable,
multi-turn harness work. The public surface is intentionally mixed:

- low-level composition around `HarnessExecutor` remains injectable
- `HarnessSessionService` adds a typed session API for persisted create/run/
  resume/inspect/list/stop flows
- `DefaultHarnessTurnDriver` and `MinimalHarnessTurnApplier` provide a narrow
  built-in runner for simple root-task sessions, tests, and the minimal CLI
- `replay_session(...)` reconstructs deterministic replay steps from canonical
  turns, with stored trace artifacts treated as cache-only derived views

## Python API

```python
from llm_tools.apps.chat_runtime import build_chat_executor
from llm_tools.harness_api import (
    BudgetPolicy,
    HarnessSessionCreateRequest,
    HarnessSessionRunRequest,
    HarnessSessionService,
    InMemoryHarnessStateStore,
    ScriptedParsedResponseProvider,
)
from llm_tools.llm_adapters import ParsedModelResponse

_, workflow_executor = build_chat_executor()
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
inspection = service.inspect_session(
    {"session_id": result.snapshot.session_id, "include_replay": True}
)
```

Stored snapshots now carry derived artifacts alongside canonical state:

- `snapshot.artifacts.trace`: aggregated per-turn observability trace cache
- `snapshot.artifacts.summary`: operator-facing session summary cache
- `snapshot.saved_at`: store save timestamp for recent-session views

Those persisted `trace` and `summary` artifacts are inspection aids, not the
source of truth. Canonical `HarnessState` remains authoritative, and replay or
inspection may rebuild or ignore stored artifacts when they are missing,
stale, inconsistent, or intentionally dropped. Persisted traces should stay
minimal and omit raw request arguments, environment state, and other
unredacted payloads.

## CLI

The minimal CLI is backed entirely by `HarnessSessionService` and supports:

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

By default, CLI state is stored under `~/.llm-tools/harness`. Pass `--store-dir`
when you want an isolated test directory or intentionally repo-local storage.

Pending approval snapshots now persist only a scrubbed base context. Newly
written records keep `invocation_id`, `workspace`, and `metadata`, but clear
process env data, logs, artifacts, and source provenance before they are saved.
Resumed approvals rebuild execution context from the current process environment
at resume time. Non-approved approval outcomes are fail-closed: the denied or
expired invocation is recorded, but later invocations from that same paused
model turn do not run. Older snapshots written before this hardening change may
still contain environment data and should be deleted if they may have held
secrets.

Malformed file-backed session records should be treated as corruption rather
than partially trusted state. CLI or app list and inspect flows should skip or
surface those records cleanly without trusting cached summary or trace data
from the damaged file.

Script files passed to `--script` contain a JSON list of
`ParsedModelResponse` payloads. This keeps the CLI deterministic and suitable
for replay-focused testing and manual harness inspection.
