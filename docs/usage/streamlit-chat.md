# Streamlit Chat

`llm_tools.apps.streamlit_chat` is the interactive Streamlit repository chat
client built on top of the existing `llm-tools` runtime, adapter, provider, and
workflow layers. It is deprecated and retained only as temporary migration and
setup guidance while repository-chat usage is absorbed into the assistant
documentation and examples.

It reuses the current repository-chat config shape and the same fixed read-only
tool set, while exposing Streamlit-native controls for the same session
features:

- transcript-style user and assistant turns
- in-memory chat session history
- live-ish active-turn polling with stop and approval actions
- model inspection and switching
- session-scoped tool enable/disable and approval toggles
- transcript export and inspector/debug panels in the sidebar settings section
- slash-command aliases for `/help`, `/model`, `/tools`, `/approvals`,
  `/inspect`, `/copy`, `quit`, and `exit`
- grounded final answers with citations, confidence, uncertainty, missing
  information, and follow-up suggestions
- OpenAI-compatible provider support for OpenAI, Ollama, and custom endpoints

## Install

```bash
~/.venvs/llm-tools/bin/python -m pip install -e .[streamlit]
```

The default development environment is shared at `~/.venvs/llm-tools` across
the main checkout and any git worktrees. Re-run `make install-dev` from the
checkout you want the shared environment to point at before launching the app.

## Launch

Use the console script:

```bash
llm-tools-streamlit-chat <directory> --config <path>
```

The app boots the standard Streamlit server and passes the same chat arguments
through to the underlying app script. The wrapper only accepts the chat app's
own arguments. If you need normal Streamlit server flags such as a custom port,
run Streamlit directly:

```bash
streamlit run src/llm_tools/apps/streamlit_chat/app.py -- <directory> --config <path>
```

## Config Shape

The Streamlit app reuses the repository-chat YAML shape exposed by
`llm_tools.apps.chat_config`:

- `llm`
- `source_filters`
- `session`
- `tool_limits`
- `policy`
- `ui`

The Streamlit lane honors the shared `ui` controls it supports:

- `show_token_usage`
- `show_footer_help`
- `inspector_open_by_default`

Minimal Ollama example:

```yaml
llm:
  provider: ollama
  model_name: gemma4:26b
  api_base_url: http://127.0.0.1:11434/v1
  temperature: 0.1

source_filters:
  include: []
  exclude: []
  include_hidden: false

session:
  max_context_tokens: 24000
  max_tool_round_trips: 8
  max_tool_calls_per_round: 4
  max_total_tool_calls_per_turn: 12

tool_limits:
  max_entries_per_call: 200
  max_recursive_depth: 12
  max_search_matches: 50
  max_read_lines: 200
  max_file_size_characters: 262144
  max_tool_result_chars: 24000

policy:
  enabled_tools: null
```

## Behavior

The Streamlit lane uses the existing grounded final-answer flow and the same
fixed read-only repository tool set:

- `list_directory`
- `find_files`
- `search_text`
- `get_file_info`
- `read_file`

The Streamlit lane keeps session history in memory only. It uses the native left
sidebar for both the session rail and a collapsible settings section,
alongside the same slash-command aliases:

- `Session Rail`: chat switching, theme, search, and session management at
  the top of the left sidebar
- `Settings`: root, provider, model, tool, approval, export, and inspector
  controls in a collapsible sidebar section below the session list

`quit` and `exit` do not stop the Streamlit server. They add a system notice
telling you to clear the chat or close the browser tab instead.

This client should not be treated as a long-term peer product to
`streamlit_assistant`; it remains only until its useful setup guidance has been
moved into the assistant-facing surfaces.
