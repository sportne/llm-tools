# Scope Audit

## Purpose

This document is the code-first audit for the current `llm-tools` repository.
It classifies every module under `src/llm_tools`, records which supported
surfaces depend on it, identifies the main concentration points, and captures
cleanup work that is justified by the current code rather than by stale design
docs.

This pass is intentionally non-destructive. It does not remove supported
surfaces. It establishes what the project actually is today.

## Audit rules

### Truth resolution

When docs, code, and tests disagree, resolve truth in this order:

1. Implemented behavior that is exercised by tests and clearly intentional
2. Implemented behavior that is exercised by product entrypoints, even if docs
   lag
3. Documented intent, only when the code appears transitional or duplicated

Every disagreement should be labeled as one of:

- docs stale
- code drift
- unresolved product decision

### Supported product floor

This audit assumes the repository must continue to support:

- the Streamlit assistant client, including normal chat and durable research
  sessions
- the public workflow and harness layers needed for prescribed workflows and
  future definable-agent capabilities
- bundled built-in integrations, even when they are not part of the assistant's
  local default working set

## Current footprint

Code size from `cloc src`:

| Area | Python LOC |
|---|---:|
| `apps` | 4,105 |
| `harness_api` | 3,910 |
| `tools` | 3,645 |
| `workflow_api` | 2,676 |
| `tool_api` | 1,972 |
| `llm_providers` | 363 |
| `llm_adapters` | 207 |
| Total `src` | 16,880 |

Test size from `cloc tests`:

| Area | Python LOC |
|---|---:|
| Total `tests` | 17,554 |

Largest modules by file size:

| Module | Approx. file size |
|---|---:|
| `src/llm_tools/apps/streamlit_assistant/app.py` | 93 KB |
| `src/llm_tools/harness_api/executor.py` | 43 KB |
| `src/llm_tools/workflow_api/chat_session.py` | 39 KB |
| `src/llm_tools/tools/atlassian/tools.py` | 35 KB |
| `src/llm_tools/tool_api/runtime.py` | 32 KB |
| `src/llm_tools/workflow_api/protection.py` | 29 KB |
| `src/llm_tools/apps/assistant_runtime.py` | 27 KB |
| `src/llm_tools/tools/filesystem/_content.py` | 26 KB |

The repository is not large because the typed core exploded. Most growth sits
in assistant-facing app code, orchestration layers, and bundled integrations.

## Supported surfaces

Current supported surfaces, as evidenced by exports, entrypoints, and tests:

- `llm_tools.tool_api`
- `llm_tools.llm_adapters`
- `llm_tools.llm_providers`
- `llm_tools.tools` registration entrypoints
- `llm_tools.workflow_api`
- `llm_tools.harness_api`
- `llm_tools.apps.streamlit_assistant`
- `llm_tools.apps.harness_cli`

`apps/*` are supported product entrypoints and support code. They are not the
default extension API for downstream consumers.

## Module classification inventory

Legend:

- `reusable core substrate`
- `assistant-critical`
- `research/workflow/agent-critical`
- `bundled integration surface`
- `app-only glue`
- `refactor candidate`
- `duplicate/obsolete candidate`

### Top-level package

| Module | Primary class | Supported surfaces | Notes |
|---|---|---|---|
| `src/llm_tools/__init__.py` | reusable core substrate | package import | Package metadata only. |

### `apps`

| Module | Primary class | Supported surfaces | Notes |
|---|---|---|---|
| `src/llm_tools/apps/__init__.py` | app-only glue | package export | Names the remaining app package exports. |
| `src/llm_tools/apps/assistant_config.py` | assistant-critical | Streamlit assistant | Assistant product config models and loaders. |
| `src/llm_tools/apps/assistant_prompts.py` | app-only glue | Streamlit assistant | Assistant prompt text assembly tied to product UX. |
| `src/llm_tools/apps/assistant_runtime.py` | refactor candidate | Streamlit assistant, research sessions | Mixes registry assembly, capability modeling, policy/context helpers, and harness turn provider logic. |
| `src/llm_tools/apps/chat_config.py` | app-only glue | assistant app helpers | Shared app-level config models. |
| `src/llm_tools/apps/chat_presentation.py` | app-only glue | Streamlit assistant | Formatting helpers for chat UI output. |
| `src/llm_tools/apps/chat_runtime.py` | duplicate/obsolete candidate | harness CLI, legacy shared bootstrap | Small shared builder that overlaps provider/registry assembly in `assistant_runtime.py`. |
| `src/llm_tools/apps/harness_cli.py` | app-only glue | harness CLI | Thin CLI over `HarnessSessionService` plus scripted provider setup. |
| `src/llm_tools/apps/protection_runtime.py` | app-only glue | Streamlit assistant | App-layer adapter around workflow protection plus provider-backed classification. |
| `src/llm_tools/apps/streamlit_models.py` | app-only glue | Streamlit assistant | Persisted UI/session models only. |
| `src/llm_tools/apps/streamlit_assistant/__init__.py` | app-only glue | Streamlit assistant | Public app entrypoint wrapper. |
| `src/llm_tools/apps/streamlit_assistant/__main__.py` | app-only glue | Streamlit assistant | Module launcher. |
| `src/llm_tools/apps/streamlit_assistant/app.py` | refactor candidate | Streamlit assistant | Main assistant product surface; currently combines persistence, thread runner orchestration, chat UX, research controls, and inspection UI. |

### `harness_api`

| Module | Primary class | Supported surfaces | Notes |
|---|---|---|---|
| `src/llm_tools/harness_api/__init__.py` | research/workflow/agent-critical | public harness API | Public facade for durable orchestration contracts. |
| `src/llm_tools/harness_api/context.py` | research/workflow/agent-critical | harness runtime | Projects durable harness state into `ToolContext`. |
| `src/llm_tools/harness_api/executor.py` | refactor candidate | harness runtime | Durable execution core; combines run-loop control, retries, approval resume, persistence, and replay artifact handling. |
| `src/llm_tools/harness_api/models.py` | research/workflow/agent-critical | public harness API | Canonical durable harness state/contracts. |
| `src/llm_tools/harness_api/planning.py` | research/workflow/agent-critical | harness runtime | Deterministic planning and replanning triggers. |
| `src/llm_tools/harness_api/protection.py` | research/workflow/agent-critical | harness runtime | Protection-oriented scrubbing over durable state. |
| `src/llm_tools/harness_api/replay.py` | research/workflow/agent-critical | harness runtime, inspection | Replay, summary, and trace artifacts. |
| `src/llm_tools/harness_api/resume.py` | research/workflow/agent-critical | harness runtime | Resume classification for stored sessions. |
| `src/llm_tools/harness_api/session.py` | research/workflow/agent-critical | public harness API, harness CLI, assistant research | Public session service plus default driver/applier/provider abstractions. |
| `src/llm_tools/harness_api/store.py` | research/workflow/agent-critical | harness runtime | Durable state store and serialization contracts. |
| `src/llm_tools/harness_api/tasks.py` | research/workflow/agent-critical | harness runtime | Task lifecycle state machine. |
| `src/llm_tools/harness_api/verification.py` | research/workflow/agent-critical | harness runtime | Verification contracts and no-progress signals. |

### `llm_adapters`

| Module | Primary class | Supported surfaces | Notes |
|---|---|---|---|
| `src/llm_tools/llm_adapters/__init__.py` | reusable core substrate | public adapter API | Re-export surface. |
| `src/llm_tools/llm_adapters/action_envelope.py` | reusable core substrate | workflow, harness, providers | Canonical structured action envelope. |
| `src/llm_tools/llm_adapters/base.py` | reusable core substrate | workflow, harness, providers | Parsed model response contracts. |

### `llm_providers`

| Module | Primary class | Supported surfaces | Notes |
|---|---|---|---|
| `src/llm_tools/llm_providers/__init__.py` | reusable core substrate | public provider API | Re-export surface. |
| `src/llm_tools/llm_providers/openai_compatible.py` | reusable core substrate | assistant, workflow, harness | Typed provider transport for OpenAI-compatible endpoints. |

### `tool_api`

| Module | Primary class | Supported surfaces | Notes |
|---|---|---|---|
| `src/llm_tools/tool_api/__init__.py` | reusable core substrate | public tool API | Re-export surface. |
| `src/llm_tools/tool_api/errors.py` | reusable core substrate | public tool API | Registry/runtime error contracts. |
| `src/llm_tools/tool_api/execution.py` | reusable core substrate | runtime, built-in tools | Execution services and service gateways. |
| `src/llm_tools/tool_api/models.py` | reusable core substrate | public tool API | Canonical typed contracts. |
| `src/llm_tools/tool_api/policy.py` | reusable core substrate | public tool API | Policy model. |
| `src/llm_tools/tool_api/redaction.py` | reusable core substrate | tool runtime, workflow chat | Redaction models and helpers. |
| `src/llm_tools/tool_api/registry.py` | reusable core substrate | public tool API | Registration/lookup. |
| `src/llm_tools/tool_api/runtime.py` | reusable core substrate | runtime, assistant, workflow, harness | Central runtime execution and gateway wiring. Dense, but still part of the core substrate. |
| `src/llm_tools/tool_api/tool.py` | reusable core substrate | public tool API | Base tool protocol and generics. |

### `tools`

| Module | Primary class | Supported surfaces | Notes |
|---|---|---|---|
| `src/llm_tools/tools/__init__.py` | bundled integration surface | public built-in registrars | Top-level built-in tool registration entrypoints. |
| `src/llm_tools/tools/_path_utils.py` | assistant-critical | filesystem, git tools | Internal path guard/helper shared by local tools. |
| `src/llm_tools/tools/atlassian/__init__.py` | bundled integration surface | public built-in registrars | Atlassian tool exports. |
| `src/llm_tools/tools/atlassian/tools.py` | bundled integration surface | assistant optional integrations | Jira, Bitbucket, and Confluence readers in one large module. |
| `src/llm_tools/tools/filesystem/__init__.py` | assistant-critical | public built-in registrars | Filesystem tool exports. |
| `src/llm_tools/tools/filesystem/_content.py` | refactor candidate | filesystem/text tools | Shared readable-content pipeline with conversion/cache logic, including MarkItDown and MPXJ backends. |
| `src/llm_tools/tools/filesystem/_ops.py` | assistant-critical | filesystem tools | Local file operation helpers. |
| `src/llm_tools/tools/filesystem/_paths.py` | assistant-critical | filesystem tools | Path normalization and workspace checks. |
| `src/llm_tools/tools/filesystem/models.py` | assistant-critical | filesystem/text tools | Shared file-read models and limits. |
| `src/llm_tools/tools/filesystem/tools.py` | assistant-critical | assistant default tool set | Core local file tools, including the only built-in write tool. |
| `src/llm_tools/tools/git/__init__.py` | assistant-critical | public built-in registrars | Git tool exports. |
| `src/llm_tools/tools/git/tools.py` | assistant-critical | assistant default tool set | Narrow read-oriented git wrappers. |
| `src/llm_tools/tools/gitlab/__init__.py` | bundled integration surface | public built-in registrars | GitLab tool exports. |
| `src/llm_tools/tools/gitlab/tools.py` | bundled integration surface | assistant optional integrations | Remote read-only GitLab tools. |
| `src/llm_tools/tools/text/__init__.py` | assistant-critical | public built-in registrars | Text tool exports. |
| `src/llm_tools/tools/text/_ops.py` | assistant-critical | text tools | Shared text-search helpers. |
| `src/llm_tools/tools/text/models.py` | assistant-critical | text tools | Search request/result models. |
| `src/llm_tools/tools/text/tools.py` | assistant-critical | assistant default tool set | Search surface layered over filesystem-readable content. |

### `workflow_api`

| Module | Primary class | Supported surfaces | Notes |
|---|---|---|---|
| `src/llm_tools/workflow_api/__init__.py` | reusable core substrate | public workflow API | Public surface is broader than a strict one-turn API because it also exports chat and protection types. |
| `src/llm_tools/workflow_api/chat_models.py` | assistant-critical | Streamlit assistant | Interactive chat session models and UI events. |
| `src/llm_tools/workflow_api/chat_session.py` | refactor candidate | Streamlit assistant | Interactive multi-round chat orchestration, approvals, event emission, and session state transitions in one module. |
| `src/llm_tools/workflow_api/executor.py` | reusable core substrate | workflow, harness | Cleanest one-turn execution boundary. |
| `src/llm_tools/workflow_api/models.py` | reusable core substrate | workflow, harness | One-turn workflow result contracts. |
| `src/llm_tools/workflow_api/protection.py` | refactor candidate | assistant workflow protection | Protection config, corpus loading, feedback store, controller, and assessment logic are all co-located here. |

## Dependency-backed findings

### 1. The typed core is not the main source of growth

`tool_api`, `llm_adapters`, and `llm_providers` together account for a minority
of total source size. Most growth sits in:

- the Streamlit assistant app
- workflow chat/protection orchestration
- harness execution/session surfaces
- bundled integration modules

This means the repo is not obviously bloated because the tool substrate itself
became over-engineered.

### 2. The assistant product surface is real, but highly concentrated

The assistant client is not speculative. It is exercised by package exports,
console scripts, docs, and tests. The main problem is concentration:

- `apps/streamlit_assistant/app.py` is the dominant hotspot
- `apps/assistant_runtime.py` still mixes several layers of responsibility
- policy/provider/protection bootstrap is repeated between normal chat turns and
  research-session launches

These are refactor and boundary problems, not evidence that the assistant
surface is unnecessary.

### 3. The workflow-to-harness boundary is mostly clean, but semantically blurred

The cleanest seam in current code is `WorkflowTurnResult`: one-turn parsing and
execution ends there, and durable orchestration begins when harness persists
that result into `HarnessState`.

The dependency graph is good and tested, but `workflow_api` already owns more
interactive session behavior than its name suggests:

- `workflow_api/chat_session.py` is an in-memory orchestrator, not just a
  one-turn helper
- approval suspension/resume exists in both workflow and harness layers
- workflow protection owns filesystem-backed state and pending prompts

This is a documentation and boundary-clarity problem first, and a cleanup
candidate second.

### 4. The built-in tool surface splits into a local core and bundled integrations

The assistant's local working set is:

- filesystem
- git
- text

Bundled but less central integrations are:

- GitLab
- Atlassian
- MarkItDown and MPXJ conversion support underneath filesystem reads

These integrations are intentionally bundled today, but they should still be
documented as secondary to the local assistant core.

### 5. No safe deletion targets are obvious from this pass alone

This audit did not identify a large body of clearly dead code. The strongest
cleanup candidates are:

- oversized modules
- duplicated bootstrap logic
- broad exports that make boundaries look blurrier than they are
- stale docs and stale backlog artifacts

Removal should follow only after a replacement or consolidation path is explicit
and tested.

## Cleanup backlog

### High-priority modularization

1. Split `apps/streamlit_assistant/app.py` into narrower modules for persisted
   workspace state, turn execution, research-session controls, and rendering.
2. Split `apps/assistant_runtime.py` into registry assembly, capability summary,
   policy/context construction, and harness-provider bridging.
3. Split `workflow_api/chat_session.py` into session-state management, event
   streaming, approval handling, and provider-loop logic.
4. Split `workflow_api/protection.py` into config/models, corpus/feedback
   storage, and controller/classifier coordination.
5. Split `harness_api/executor.py` so run-loop control, persistence conflict
   handling, approval resumption, and replay artifact building are not all in
   one file.
6. Split `tools/atlassian/tools.py` by Jira, Bitbucket, and Confluence product
   area.

### Boundary cleanup

1. Decide whether `apps/chat_runtime.py` remains a supported bootstrap helper or
   is folded into a clearer assistant/harness bootstrap path.
2. Reduce app-layer calls into lower-layer private APIs.
3. Revisit whether interactive chat session orchestration belongs in
   `workflow_api` or a more explicit session/orchestration package.
4. Revisit whether the public `workflow_api` export surface should present chat
   and protection as peer APIs to `WorkflowExecutor`.

### Packaging and dependency follow-ups

1. Keep bundled integrations for now, but make the local assistant core versus
   optional bundled integrations explicit in docs.
2. Review whether `jpype` should be documented or declared more explicitly for
   the MPXJ path.
3. Review runtime wiring in `tool_api` that still knows about remote-service
   gateways directly.

### Secondary documentation cleanup

1. Refresh usage/security docs whose cache-path or dependency descriptions no
   longer match the code.
2. Keep design docs aligned with the import-layer tests in
   `tests/architecture/test_layering.py`.

## Audit conclusion

The current repository is larger than a strict tool substrate, but the extra
size is not obviously accidental. Most of the growth is carrying one of three
real responsibilities:

- the assistant product surface
- durable workflow/harness orchestration
- bundled integrations

The correct cleanup target is concentration, overlap, and stale documentation,
not wholesale removal of these surfaces.
