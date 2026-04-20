# TASKS.md

## Purpose

This file is the repository's active backlog and status tracker.

The current baseline is the rebuilt documentation canon plus the supported code
surfaces described in `README.md` and the design docs.

## Status conventions

- `[ ]` Not started
- `[~]` In progress
- `[x]` Done

## Current backlog

### [x] Re-baseline repository documentation

- [x] Collapse the Markdown set into a smaller canonical doc tree.
- [x] Remove stale references to deleted implementation, usage, and security
  index docs.
- [x] Align `README.md`, `AGENTS.md`, and the design docs with the current
  package layout and split facade/internal module structure.

### [ ] Decompose oversized assistant app surfaces

- [ ] Split `src/llm_tools/apps/streamlit_assistant/app.py` further so state
  persistence, turn execution, research-session controls, and rendering stop
  concentrating in one module.
- [ ] Reduce repeated assistant setup logic shared between direct chat turns and
  harness-backed research launches.
- [ ] Decide how much of `assistant_runtime.py` should remain a convenience
  export surface versus moving into narrower assistant-specific modules.

### [ ] Reduce concentration in workflow and harness internals

- [ ] Keep the thin public facades in `workflow_api/chat_session.py`,
  `workflow_api/protection.py`, `harness_api/executor.py`, and
  `harness_api/session.py`, but continue splitting heavy internal logic where it
  is still concentrated.
- [ ] Revisit whether interactive chat orchestration should continue to live in
  `workflow_api` or move to a more explicit session/orchestration package while
  preserving the one-turn workflow surface.
- [ ] Continue narrowing large internal modules such as
  `workflow_api/chat_runner.py`, `workflow_api/protection_controller.py`, and
  `harness_api/executor_loop.py`.

### [ ] Modularize bundled integration surfaces

- [ ] Finish the Atlassian split so product-specific implementation stops
  concentrating in `src/llm_tools/tools/atlassian/tools.py`.
- [ ] Revisit the `filesystem` and `text` tool boundary, which still shares the
  same readable-content pipeline and cache path.
- [ ] Review remote-integration wiring that is still embedded directly in
  `tool_api` runtime and execution helpers.

### [ ] Validate safe dead-code removals

- [ ] Confirm whether duplicate or transitional helpers in `apps`,
  `workflow_api`, and `harness_api` can now be deleted without affecting
  supported behavior.
- [ ] Remove only surfaces whose replacement path is explicit, documented, and
  covered by tests.
