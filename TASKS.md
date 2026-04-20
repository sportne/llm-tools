# TASKS.md

## Purpose

This file is the repository's active backlog and status tracker.

The previous `streamlit_chat` removal work is complete. The current baseline is
the code-first scope audit and documentation re-baselining captured in
[docs/implementation/scope-audit.md](docs/implementation/scope-audit.md).

## Status conventions

- `[ ]` Not started
- `[~]` In progress
- `[x]` Done

## Current backlog

### [x] Establish current scope truth

- [x] Audit every module under `src/llm_tools`.
- [x] Re-baseline `README.md`, `TASKS.md`, and the core design docs.
- [x] Record supported surfaces, dependency paths, and cleanup candidates
  without removing supported behavior.

### [ ] Decompose oversized assistant app surfaces

- [ ] Split `src/llm_tools/apps/streamlit_assistant/app.py` into narrower
  modules for state persistence, turn execution, research-session controls, and
  rendering.
- [ ] Split `src/llm_tools/apps/assistant_runtime.py` so registry assembly,
  policy/context helpers, and research-provider wiring are no longer co-located
  in one file.
- [ ] Remove repeated assistant setup logic shared between normal chat turns and
  research-session launches.

### [ ] Simplify overlapping app/runtime assembly

- [ ] Decide whether `src/llm_tools/apps/chat_runtime.py` remains a supported
  helper or should be folded into a more explicit assistant/harness bootstrap
  layer.
- [ ] Eliminate reliance on private APIs in app helpers where feasible.
- [ ] Narrow app-layer code that currently reaches through lower-layer private
  internals.

### [ ] Reduce concentration in workflow and harness execution code

- [ ] Split `src/llm_tools/workflow_api/chat_session.py` into smaller session,
  event, and approval-flow components.
- [ ] Revisit whether interactive chat session orchestration belongs inside
  `workflow_api` or should move to a more explicit session/orchestration layer.
- [ ] Split `src/llm_tools/workflow_api/protection.py` into narrower controller,
  persistence, and model/config modules.
- [ ] Split `src/llm_tools/harness_api/executor.py` so run-loop control,
  persistence/retry handling, and approval resumption are easier to follow and
  test.
- [ ] Separate the public service API in `src/llm_tools/harness_api/session.py`
  from default driver/applier implementations.

### [ ] Modularize bundled integration surfaces

- [ ] Split `src/llm_tools/tools/atlassian/tools.py` by product area instead of
  keeping Jira, Bitbucket, and Confluence in one module.
- [ ] Revisit the `filesystem` and `text` tool boundary, which currently shares
  the same readable-content pipeline and cache path.
- [ ] Review remote-integration wiring that is still embedded directly in
  `tool_api` runtime/execution helpers.

### [ ] Finish secondary documentation cleanup

- [ ] Refresh usage docs that still describe outdated cache paths or stale
  dependency details.
- [ ] Ensure dependency documentation matches the actual packaged and lazily
  imported runtime surface.
- [ ] Keep architecture docs aligned with the tested import-layer rules.

### [ ] Validate safe dead-code removals

- [ ] Confirm whether any current duplicate or transitional helpers can be
  deleted without affecting supported assistant, workflow, harness, or bundled
  integration behavior.
- [ ] Do not remove supported surfaces until their replacement path is explicit
  and documented.
