# TASKS.md

## Purpose

This document tracks the direct removal of the obsolete `streamlit_chat`
surface from `llm-tools`.

The root `TASKS.md` remains the canonical backlog for this repository.

## Status conventions

- `[ ]` Not started
- `[~]` In progress
- `[x]` Done

## Removal checklist

### [x] Delete the deprecated client

- [x] Delete `src/llm_tools/apps/streamlit_chat/`.
- [x] Remove the `llm-tools-streamlit-chat` console script.
- [x] Remove top-level app exports for `streamlit_chat`.

### [x] Clean up shared code that only existed for the deleted client

- [x] Move persisted Streamlit session models into
  `src/llm_tools/apps/streamlit_models.py`.
- [x] Repoint `streamlit_assistant` to the neutral shared models module.
- [x] Delete `src/llm_tools/apps/chat_controls.py`.
- [x] Delete `src/llm_tools/apps/chat_prompts.py`.
- [x] Delete `TextualChatConfig` and `load_textual_chat_config`.
- [x] Remove `chat_runtime` helpers that existed only for `streamlit_chat`.

### [x] Remove obsolete docs and tests

- [x] Remove `docs/usage/streamlit-chat.md`.
- [x] Remove README and usage-index references to `streamlit_chat`.
- [x] Remove launch/import test expectations for `streamlit_chat`.
- [x] Delete `tests/apps/test_streamlit_chat.py`.
- [x] Replace shared-app helper tests with coverage for the surviving helpers.

### [x] Final wording and architecture cleanup

- [x] Update architecture and implementation docs so
  `streamlit_assistant` is the only interactive client.
- [x] Remove stale `streamlit_chat` and Textual-era public-surface references
  from active code and docs.
