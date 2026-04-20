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

`llm_adapters` translates provider/model output into canonical internal turn
results.

Today the main adapter is `ActionEnvelopeAdapter`, backed by shared parsed
response contracts in `base.py`.

### `llm_providers`

`llm_providers` owns typed model transport through OpenAI-compatible endpoints.

It does not execute tools directly. It prepares and validates structured model
responses against adapter/workflow-provided response models.

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
- assistant-oriented interactive chat/protection helpers used by the Streamlit
  assistant

The cleanest one-turn boundary is:

- prepare model-facing contract
- parse model output through the adapter
- execute resulting tool invocations through `WorkflowExecutor`
- return `WorkflowTurnResult`

That boundary is what `harness_api` builds on.

The package also currently includes interactive session state, approval waiting,
and protection helpers. Those are real supported surfaces today, but they make
`workflow_api` broader than a strict one-turn package.

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
`WorkflowTurnResult` into persisted `HarnessTurn` / `HarnessState`.

### `apps`

`apps` contains supported product entrypoints and app-local glue.

Current app surfaces are:

- `streamlit_assistant`
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
it is semantically blurred in implementation because:

- `workflow_api/chat_session.py` is itself an in-memory orchestrator
- approval suspension/resume exists in both workflow and harness layers
- `workflow_api/protection.py` manages filesystem-backed feedback and pending
  prompt state

This is a refactor target, not evidence that either layer is unnecessary.

### Core substrate vs bundled integrations

The typed substrate remains reusable, but the runtime currently knows how to
instantiate remote-service gateways and document-conversion helpers directly.
That keeps behavior centralized, but it means some bundled integrations are not
fully modular from the runtime's perspective.

### Library surface vs app glue

Some app-layer modules are clearly product-specific and should stay that way.
Others, such as assistant bootstrap helpers, currently mix reusable runtime
assembly with product presentation concerns. The cleanup goal is to separate
those concerns without pretending the assistant surface is outside repo scope.

## Known concentration points

Current hotspots called out by the scope audit are:

- `src/llm_tools/apps/streamlit_assistant/app.py`
- `src/llm_tools/apps/assistant_runtime.py`
- `src/llm_tools/workflow_api/chat_session.py`
- `src/llm_tools/workflow_api/protection.py`
- `src/llm_tools/harness_api/executor.py`
- `src/llm_tools/tools/atlassian/tools.py`
- `src/llm_tools/tools/filesystem/_content.py`

These are concentration and modularization problems. They are not, by themselves, proof that the surrounding surfaces should be removed.

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
- prefer splitting concentrated modules over deleting supported behavior
- keep docs aligned with the actual codebase and test suite
