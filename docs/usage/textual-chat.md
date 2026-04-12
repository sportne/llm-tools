# Textual Chat

`llm_tools.apps.textual_chat` is an interactive Textual repository chat client
built on top of the `llm-tools` runtime, adapter, provider, and workflow
layers.

It preserves the `engllm-chat` style user experience:

- transcript-first terminal chat UI
- Enter to send and Shift+Enter for newlines
- `/help`, `/model`, `/copy`, `quit`, and `exit`
- startup credential prompting when an API key is needed and missing
- interrupt/stop support for an active turn
- grounded final answers with citations, confidence, and follow-up metadata

The first port targets OpenAI-compatible providers only:

- OpenAI
- Ollama
- custom OpenAI-compatible endpoints

## Install

```bash
.venv/bin/python -m pip install -e .[apps]
```

## Launch

You can launch the chat app with either:

```bash
python -m llm_tools.apps.textual_chat <directory> --config <path>
```

or:

```bash
llm-tools-chat <directory> --config <path>
```

## Config Shape

The YAML config keeps the same five top-level sections used by the original
client:

- `llm`
- `source_filters`
- `session`
- `tool_limits`
- `ui`

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

ui:
  show_token_usage: true
  show_footer_help: true
```

## Behavior

The app exposes a fixed read-only repository tool set:

- `list_directory`
- `find_files`
- `search_text`
- `get_file_info`
- `read_file`

Those tools run through the standard `ToolRuntime` and `WorkflowExecutor`
layers, while model output is parsed through `ActionEnvelopeAdapter`.
