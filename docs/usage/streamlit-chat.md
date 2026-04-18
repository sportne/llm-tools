# Streamlit Chat

`llm_tools.apps.streamlit_chat` is an interactive Streamlit repository chat
client built on top of the existing `llm-tools` runtime, adapter, provider, and
workflow layers.

It reuses the current repository-chat config shape and the same fixed read-only
tool set as the Textual lane:

- transcript-style user and assistant turns
- in-memory chat session history
- grounded final answers with citations, confidence, uncertainty, missing
  information, and follow-up suggestions
- OpenAI-compatible provider support for OpenAI, Ollama, and custom endpoints

## Install

```bash
.venv/bin/python -m pip install -e .[streamlit]
```

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

The Streamlit app reuses the same YAML shape as `llm_tools.apps.textual_chat`:

- `llm`
- `source_filters`
- `session`
- `tool_limits`
- `policy`
- `ui`

The `ui` section is accepted for schema compatibility with the Textual lane.
The current Streamlit app ignores those `ui` values.

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

The current Streamlit lane keeps session history in memory only and does not
implement an interactive approval queue. If local-read approvals are enabled in
config, the app will deny those approval requests and report that in the
transcript.
