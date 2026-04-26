# SPEC.md

## Purpose

`llm-tools` is a typed Python library for defining, validating, exposing, and
executing tools, with higher-level workflow, harness, and assistant surfaces
built on top of that substrate.

The repository is not just a minimal tool core. It currently supports:

- typed tool definition and execution
- model-output normalization and OpenAI-compatible provider transport
- one-turn workflow execution
- assistant-facing chat and protection flows
- durable harness sessions for longer-running research and workflow tasks
- bundled built-in integrations for local and remote read surfaces

## Truth resolution

When documentation and implementation disagree:

1. Tested, implemented behavior wins.
2. Product entrypoints and public exports break ties when docs lag.
3. Documented intent wins only when the code is clearly transitional,
   duplicated, or inconsistent.

Every disagreement should be called out as one of:

- docs stale
- code drift
- unresolved product decision

## Supported surfaces

The current supported library surfaces are:

- `llm_tools.tool_api`
- `llm_tools.llm_adapters`
- `llm_tools.llm_providers`
- `llm_tools.tools` registration entrypoints
- `llm_tools.workflow_api`
- `llm_tools.harness_api`

The current supported product entrypoints are:

- `llm_tools.apps.nicegui_chat`
- `llm_tools.apps.harness_cli`

`apps/*` are supported product surfaces, but they are not the default extension
API for downstream consumers.

## Current product floor

### Tool substrate

The canonical substrate is still the typed tool system:

- `ToolSpec`, `ToolContext`, `ToolInvocationRequest`, `ToolResult`, and related
  models live in `tool_api`
- `ToolRegistry` owns registration and lookup
- `ToolRuntime` owns validation, policy enforcement, execution mediation,
  gateway wiring, and result normalization
- built-in tools are implemented outside `tool_api`

### Adapters and providers

- `llm_adapters` normalizes model output into canonical parsed responses
- `ActionEnvelopeAdapter` is the current structured action adapter
- `llm_providers` owns typed model transport through OpenAI-compatible endpoints

### Workflow layer

`workflow_api` is the public layer above the raw tool substrate.

Today it contains two kinds of functionality:

- one-turn workflow primitives centered on `WorkflowExecutor`,
  `PreparedModelInteraction`, and `WorkflowTurnResult`
- assistant-oriented interactive chat and protection helpers used by the
  NiceGUI assistant

Some public workflow modules are now intentionally thin facades:

- `workflow_api/chat_session.py` re-exports split chat internals from
  `chat_runner.py`, `chat_state.py`, and `chat_inspector.py`
- `workflow_api/protection.py` re-exports split protection internals from
  `protection_models.py`, `protection_store.py`, and
  `protection_controller.py`

The repo should preserve the clean one-turn workflow primitive even while
assistant-oriented session helpers remain in the same package.

### Harness layer

`harness_api` is the durable orchestration layer above the one-turn workflow
primitive.

It owns:

- persisted session state
- resume and replay
- task lifecycle state
- deterministic planning and replanning triggers
- verification contracts and no-progress signals
- a public session service and store abstractions

Some public harness modules are also intentionally thin facades:

- `harness_api/executor.py` fronts split execution internals such as
  `executor_loop.py`, `executor_approvals.py`, and `executor_persistence.py`
- `harness_api/session.py` fronts `session_service.py` and default driver or
  applier helpers

This layer is in scope for prescribed workflows and future definable-agent
capabilities.

### Assistant product layer

The NiceGUI assistant is the main interactive client currently supported by
the repository. It supports:

- direct assistant chat
- policy-aware tool exposure and approvals
- prompt and response protection hooks
- durable research sessions backed by `harness_api`
- optional bundled remote integrations alongside the local tool core

## Built-in integrations

The bundled built-in tool families are:

- filesystem
- git
- text
- GitLab
- Atlassian

The assistant's local working set is primarily filesystem, git, and text.
GitLab and Atlassian are intentionally bundled but secondary integrations.
Document-conversion backends such as MarkItDown and MPXJ are part of the
filesystem and text read pipeline rather than first-class tool families.

## Current architectural constraints

- `tool_api` must remain independent of `workflow_api`, `harness_api`, `apps`,
  and the tool implementation packages.
- `workflow_api` must not import `harness_api`.
- `harness_api` may depend on `workflow_api`, `tool_api`, `llm_adapters`, and
  `llm_providers`, but not on `apps` or `tools`.
- `tools` must not depend on `workflow_api`, `harness_api`, or `apps`.
- `apps` may compose any lower layer, but app-local convenience code should not
  become the default public extension surface by accident.

These constraints are enforced in part by the architecture tests.

## What this spec does not claim

This spec does not claim that the current package boundaries are already ideal.
The codebase still contains concentration in assistant, workflow, harness, and
bundled integration modules. The near-term priority is to keep splitting the
largest internal modules without removing supported behavior prematurely.

## Acceptance criteria for the current repository state

The repository should continue to satisfy these baseline expectations:

- the public package surfaces listed above remain importable
- the NiceGUI assistant and harness CLI entrypoints remain supported
- the assistant continues to support both normal chat and durable research
  sessions
- workflow one-turn execution remains reusable independently of durable harness
  state
- harness sessions remain durable, resumable, and inspectable
- bundled integrations remain available, even if some are less central than the
  local assistant core
- docs and backlog files should describe the current codebase rather than an
  earlier scaffold state
