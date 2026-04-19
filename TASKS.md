# TASKS.md

## Purpose

This document defines the next productization backlog for `llm-tools`.

The repository now has enough completed tool, workflow, and harness foundation
work to support a broader assistant product surface. The next direction is not
more harness-core scaffolding. It is the delivery of a separate assistant-facing
application that can answer normal questions, optionally use the full built-in
registry, and help users work across large amounts of local and remote
information.

The root `TASKS.md` remains the canonical backlog for this repository.

## Status conventions

- `[ ]` Not started
- `[~]` In progress
- `[x]` Done
- `[-]` Deferred

## Current foundation

The following foundations are already available and should be treated as inputs
for the next milestone rather than re-opened as new backlog items:

- `tool_api` typed models, runtime, registry, policy, and observability
- `llm_adapters` with the canonical `ActionEnvelopeAdapter`
- `llm_providers` for OpenAI-compatible model access
- built-in filesystem, git, text, Atlassian, and GitLab tools
- `workflow_api` one-turn execution and interactive chat-session orchestration
- `harness_api` durable sessions, persisted traces, summaries, replay, and the
  public Python session surface
- the minimal persisted harness CLI
- existing Textual and Streamlit repository chat apps
- the Textual workbench

## Definition of Done

The repository qualifies for this next milestone when all of the following are
true:

- A separate `streamlit_assistant` app exists alongside the current
  repository-focused `streamlit_chat` app.
- The assistant can answer ordinary chat questions without requiring any tool
  use.
- The assistant can expose the full built-in tool registry through safe,
  session-scoped controls.
- The assistant clearly surfaces tool capability state, including missing
  workspace, missing credentials, permission-blocked tools, disabled tools, and
  approval-gated tools.
- Local, Atlassian, and GitLab access flows are practical without turning the
  repository chat app into a general assistant.
- Harness-backed research sessions can be launched, listed, inspected, resumed,
  stopped, and summarized back into the interactive assistant.
- The current repository chat app remains intact as the repository-focused lane.
- Documentation, packaging, and tests reflect the new assistant direction.

## Phased backlog

### [x] Phase 1: Assistant product contract and config

Outcome: define the assistant lane as a separate product surface rather than a
mutation of the repository chat app.

#### [x] 1.1 Add a separate assistant app package

Description: introduce `src/llm_tools/apps/streamlit_assistant/` as a sibling
Streamlit app with its own package entrypoints.

#### [x] 1.2 Add a separate assistant config model

Description: define and load an assistant-specific config model that reuses the
existing low-level LLM, session, UI, policy, and tool-limit shapes where they
still fit, while dropping repository-chat assumptions such as `source_filters`
as the top-level identity of the app.

#### [x] 1.3 Stabilize assistant-facing configuration examples

Description: add and curate configuration examples for common assistant usage
modes such as local-only chat, enterprise-data chat, and harness-backed
research.

### [x] Phase 2: Shared assistant runtime and safe full-registry policy model

Outcome: expose the full built-in registry behind safe defaults and explicit
session-scoped controls.

#### [x] 2.1 Add a shared assistant runtime builder

Description: provide assistant-specific helpers for building the full registry,
policy, tool context, and capability summaries without widening the current
repository-chat runtime helpers in place.

#### [x] 2.2 Default to tool-free chat with explicit opt-in tool use

Description: ensure the assistant starts from ordinary chat with zero tools
required, while keeping the full registry available for enablement.

#### [x] 2.3 Keep high-risk capabilities safely gated

Description: keep `write_file` present but disabled by default, require approval
for local and external writes, and leave network and subprocess access under
explicit session control.

#### [x] 2.4 Add richer capability summaries where useful

Description: extend capability reporting and helper APIs where needed so future
frontends can consume the same availability model cleanly.

### [x] Phase 3: Streamlit assistant UI and persisted chat sessions

Outcome: ship a usable assistant chat client that is distinct in purpose from
repository chat.

#### [x] 3.1 Add the assistant Streamlit app shell

Description: add the main Streamlit assistant UI, packaging entrypoint, and
basic persisted session behavior.

#### [x] 3.2 Support grouped tool and permission controls

Description: group tools by source category and surface their current
availability state in the UI.

#### [x] 3.3 Reuse the existing one-turn chat execution loop for v1

Description: keep `workflow_api` chat-turn execution as the primary v1 chat
engine so the assistant can ship without turning `harness_api` into the main
transcript surface.

#### [x] 3.4 Refine assistant UX copy and session ergonomics

Description: continue tightening labels, empty-state guidance, source hints,
and session affordances so the assistant feels purpose-built rather than like a
renamed repository chat app.

### [x] Phase 4: Harness-backed research sessions, replay, approvals, and summaries

Outcome: let the assistant launch durable research work without making the
harness the primary chat engine.

#### [x] 4.1 Add an app-facing research session controller

Description: wrap the public `HarnessSessionService` in a thin assistant-facing
controller for launch, list, inspect, resume, stop, and summary-copy flows.

#### [x] 4.2 Add a live harness model-turn provider for research mode

Description: add a provider wrapper that turns harness task context into model
messages for real research-session execution.

#### [x] 4.3 Surface research summaries back into chat

Description: let the assistant insert durable research summaries back into the
interactive transcript.

#### [x] 4.4 Improve research-session UI depth

Description: expose richer replay, trace, approval-resolution, and inspection
views beyond the current minimal assistant session panel.

### [x] Phase 5: Documentation, examples, and architecture/test coverage

Outcome: make the new assistant direction clear and maintainable.

#### [x] 5.1 Update packaging and top-level docs

Description: add the assistant console script and update README and usage docs
for the new product split.

#### [x] 5.2 Replace the root backlog with the assistant direction

Description: rewrite this `TASKS.md` so it reflects productization work rather
than repeating already-completed harness-foundation phases.

#### [x] 5.3 Add targeted assistant tests

Description: cover assistant packaging, config loading, full-registry runtime
helpers, direct chat turns, and harness-backed research-session control flows.

#### [x] 5.4 Expand regression coverage around the assistant and repository-chat split

Description: add broader integration and UI tests where needed to keep the two
Streamlit lanes clearly separated over time.

## Milestones

### [x] Milestone A: Assistant UI exists

Goal: the repository ships a separate Streamlit assistant that handles ordinary
chat plus tool-enabled chat.

Included work:
- Phase 1
- Phase 2
- Phase 3

Exit criteria:
- separate assistant package and CLI exist
- assistant config is separate from repository-chat config
- ordinary assistant answers work without tool use
- full built-in registry is available behind safe session controls

### [x] Milestone B: Assistant safely accesses local and network knowledge sources

Goal: the assistant is a practical entry point for large local and proprietary
information sets.

Included work:
- remaining Phase 2 and Phase 3 work
- assistant documentation and source-mode examples

Exit criteria:
- local, Atlassian, and GitLab tool flows are documented and usable
- missing workspace, credentials, and permissions are surfaced clearly
- write and higher-risk capabilities remain approval-aware and opt-in

### [x] Milestone C: Harness-backed research workflows are usable from the assistant

Goal: the assistant can launch and manage durable research sessions without
making the harness the primary transcript engine.

Included work:
- Phase 4
- remaining Phase 5 work

Exit criteria:
- research sessions can be launched from the assistant
- recent research sessions can be listed and inspected
- pending or paused research can be resumed or stopped
- assistant users can pull durable research summaries back into normal chat
