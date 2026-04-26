# ARCHITECTURE.md

## Purpose

This document describes the current architecture of `llm-tools` as implemented,
not an earlier scaffold or deferred design.

The repository has seven practical layers:

1. `tool_api`
2. `llm_adapters`
3. `llm_providers`
4. `tools`
5. `workflow_api`
6. `harness_api`
7. `apps`

## Truth resolution

When design docs and code disagree, treat tested implementation as the primary
source of truth. Update the docs rather than preserving an aspirational diagram
that no longer matches the repository.

## Layer model

```text
apps
  -> harness_api
  -> workflow_api
  -> llm_providers
  -> tool_api
  -> tools

harness_api
  -> workflow_api
  -> llm_adapters
  -> llm_providers
  -> tool_api

workflow_api
  -> llm_adapters
  -> tool_api

llm_providers
  -> llm_adapters

tools
  -> tool_api

llm_adapters
  -> (no internal higher layers)

tool_api
  -> (no internal higher layers)
```

These dependency rules are enforced by `tests/architecture/test_layering.py`.

## Layer responsibilities

### `tool_api`

`tool_api` is the reusable typed substrate.

It owns:

- canonical models for tool metadata, invocation, results, policy, and errors
- registry and runtime behavior
- runtime mediation through `ToolExecutionContext` and execution services
- secret, filesystem, subprocess, and remote-service gateway wiring used by the
  runtime

This layer should stay independent of tool implementations, workflow layers,
assistant app code, and harness orchestration.

### `llm_adapters`

`llm_adapters` translates provider or model output into canonical internal turn
results.

Today the main adapter is `ActionEnvelopeAdapter`, backed by shared parsed
response contracts in `base.py`.

### `llm_providers`

`llm_providers` owns typed model transport through OpenAI-compatible endpoints.

It does not execute tools directly. It prepares and validates structured model
responses against adapter or workflow-provided response models.

### `tools`

`tools` contains bundled built-in tool implementations that are registered
against `tool_api`.

The current families are:

- filesystem
- git
- text
- GitLab
- Atlassian

The local assistant core uses filesystem, git, and text most directly. GitLab
and Atlassian are intentionally bundled remote integrations.

### `workflow_api`

`workflow_api` currently contains two related but distinct surfaces:

- one-turn execution primitives centered on `WorkflowExecutor`
- assistant-oriented interactive chat and protection helpers used by the
  NiceGUI assistant

The cleanest one-turn boundary is:

- prepare the model-facing contract
- parse model output through the adapter
- execute resulting tool invocations through `WorkflowExecutor`
- return `WorkflowTurnResult`

That boundary is what `harness_api` builds on.

The package also exports interactive session state and protection helpers, but
the public modules are now split differently than earlier audit docs implied:

- `workflow_api/chat_session.py` is a thin facade over `chat_runner.py`,
  `chat_state.py`, and `chat_inspector.py`
- `workflow_api/protection.py` is a thin facade over `protection_models.py`,
  `protection_store.py`, `protection_provenance.py`, and
  `protection_controller.py`

Those helpers are still real supported surfaces today, but they no longer all
live in one implementation file.

### `harness_api`

`harness_api` is the durable orchestration layer above one-turn workflow.

It owns:

- persisted session state
- turn history and replay artifacts
- resume classification and approval durability
- task lifecycle and deterministic planning
- verification contracts and no-progress signals
- a public session service plus state-store abstractions

The key seam between workflow and harness is the handoff from
`WorkflowTurnResult` into persisted `HarnessTurn` and `HarnessState`.

Public harness modules are also intentionally split:

- `harness_api/executor.py` is a thin facade over execution internals such as
  `executor_loop.py`, `executor_approvals.py`, and `executor_persistence.py`
- `harness_api/session.py` is a thin facade over `session_service.py` and the
  default driver or applier helpers

### `apps`

`apps` contains supported product entrypoints and app-local glue.

Current app surfaces are:

- `nicegui_chat`
- `harness_cli`
- supporting config, runtime, prompt, presentation, and persistence helpers

These modules may compose all lower layers, but they should not accidentally
become the main public extension surface for library consumers.

## Important boundaries

### Workflow vs harness

The intended durable split is:

- `workflow_api` owns one parsed model turn and related in-memory assistant chat
  helpers
- `harness_api` owns durable session orchestration and persisted state

This boundary is mostly clean in the import graph and in the durable models, but
it is still semantically blurred in implementation because approval suspension,
resume behavior, and assistant-facing orchestration concerns span both layers.
That is a continuing refactor target, not evidence that either layer is
unnecessary.

### Core substrate vs bundled integrations

The typed substrate remains reusable, but the runtime still knows how to
instantiate remote-service gateways and document-conversion helpers directly.
That keeps behavior centralized, but it means some bundled integrations are not
fully modular from the runtime's perspective.

### Library surface vs app glue

Some app-layer modules are clearly product-specific and should stay that way.
Others, such as assistant bootstrap helpers, still mix reusable runtime assembly
with product presentation concerns. The cleanup goal is to separate those
concerns without pretending the assistant surface is outside repo scope.

## Known concentration points

Current hotspots are implementation modules, not the thin public facades:

- `src/llm_tools/apps/nicegui_chat/app.py`
- `src/llm_tools/apps/assistant_runtime.py`
- `src/llm_tools/harness_api/executor_loop.py`
- `src/llm_tools/workflow_api/chat_runner.py`
- `src/llm_tools/tool_api/runtime.py`
- `src/llm_tools/tools/atlassian/tools.py`
- `src/llm_tools/tools/filesystem/_content.py`

These are concentration and modularization problems. They are not, by
themselves, proof that the surrounding surfaces should be removed.

## Public API guidance

The primary public library surfaces are:

- `tool_api`
- `llm_adapters`
- `llm_providers`
- `tools` registration helpers
- `workflow_api`
- `harness_api`

`apps/*` are supported product entrypoints, but downstream extensions should
prefer the lower library layers unless a product-specific app behavior is
explicitly what they want to consume.

## Change guidance

When changing architecture:

- preserve tested import-layer rules
- preserve the one-turn workflow primitive even if assistant session helpers move
- preserve durable harness state and public session contracts
- prefer splitting concentrated implementation modules over deleting supported
  behavior
- keep docs aligned with the actual codebase and test suite
