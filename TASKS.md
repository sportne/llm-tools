# TASKS.md

## Purpose

This document defines the current decommissioning backlog for the interactive
application surfaces in `llm-tools`.

The repository is consolidating around `streamlit_assistant` as the only
long-term interactive client. The older `streamlit_chat` app remains only as
temporary migration and setup guidance, and the Textual app surfaces are being
removed.

The root `TASKS.md` remains the canonical backlog for this repository.

## Status conventions

- `[ ]` Not started
- `[~]` In progress
- `[x]` Done
- `[-]` Deferred

## Current state

- `streamlit_assistant` is the target interactive client.
- `streamlit_chat` is deprecated immediately and should no longer be treated as
  a supported destination product.
- `streamlit_chat` remains only until any useful setup and migration guidance
  has been moved into assistant-facing docs and examples.
- Shared repository-chat helpers such as `chat_config.py` and `chat_runtime.py`
  remain in place for now because `streamlit_chat` still depends on them.
- Textual app surfaces are removed and should not return as supported products.

## Phased backlog

### [x] Phase 1: Remove Textual app surfaces

Outcome: remove the Textual repository chat and workbench packages from the
active product surface.

- [x] Delete `src/llm_tools/apps/textual_chat/`.
- [x] Delete `src/llm_tools/apps/textual_workbench/`.
- [x] Remove Textual console scripts and dependency hooks from packaging.
- [x] Remove Textual usage docs and active-product references from top-level
  docs.
- [x] Remove Textual-focused tests and import helpers.

### [~] Phase 2: Deprecate `streamlit_chat` down to migration guidance

Outcome: keep the old Streamlit repository chat client only as short-term
reference material while directing users to the assistant.

- [x] Mark `streamlit_chat` as deprecated in the backlog and top-level docs.
- [ ] Narrow `streamlit_chat` docs to migration and setup guidance only.
- [ ] Stop presenting `streamlit_chat` as a peer product lane to the assistant.
- [ ] Identify the shared config and runtime helpers that exist only because
  `streamlit_chat` still ships.

### [-] Phase 3: Remove `streamlit_chat`

Outcome: remove the obsolete repository-chat app once its remaining guidance is
absorbed elsewhere.

- [-] Move any remaining setup guidance into assistant docs or assistant config
  examples.
- [-] Delete `src/llm_tools/apps/streamlit_chat/` and its console entrypoint.
- [-] Remove Streamlit-chat-specific docs, examples, and tests.
- [-] Collapse compatibility code that only exists to keep the old app running.

### [-] Phase 4: Clean up post-`streamlit_chat` compatibility names

Outcome: remove or normalize Textual-era naming once the old Streamlit app no
longer needs it.

- [-] Rename `TextualChatConfig` and `load_textual_chat_config` to neutral
  repository-chat names if they still remain public.
- [-] Delete those compatibility names instead if no remaining public surface
  needs them.
- [-] Remove any remaining Textual-era wording from docs, examples, or tests.
