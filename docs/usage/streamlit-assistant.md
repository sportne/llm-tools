# Streamlit Assistant

`llm_tools.apps.streamlit_assistant` is an interactive Streamlit assistant built
on top of the existing `llm-tools` runtime, adapter, provider, workflow, and
harness layers.

It is intentionally different from the repository-focused chat clients:

- normal chat answers can be returned without any tool use
- the full built-in tool registry can be enabled per session
- local files, Git, text helpers, Atlassian, and GitLab tools can all be used
  when the relevant permissions and credentials are available
- durable harness-backed research sessions can be launched, inspected, resumed,
  stopped, and summarized back into the interactive chat

## Install

```bash
.venv/bin/python -m pip install -e .[streamlit]
```

## Launch

Use the console script:

```bash
llm-tools-streamlit-assistant <directory> --config <path>
```

Or run the Streamlit app directly:

```bash
streamlit run src/llm_tools/apps/streamlit_assistant/app.py -- <directory> --config <path>
```

The optional directory becomes the initial workspace root for local file and
Git tools. The assistant can still answer normal questions when no root is
selected.

## Config Shape

The assistant uses a separate config model from the repository chat apps:

- `llm`
- `session`
- `tool_limits`
- `policy`
- `ui`
- `workspace`
- `research`

Minimal Ollama example:

```yaml
llm:
  provider: ollama
  model_name: gemma4:26b
  api_base_url: http://127.0.0.1:11434/v1
  temperature: 0.1

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

workspace:
  default_root: .

research:
  default_max_turns: 6
  max_recent_sessions: 8
```

## Behavior

The assistant starts with no tools enabled by default unless the config
explicitly enables them.

Session controls let you opt into:

- tool enablement by source category
- network access
- filesystem access
- subprocess access
- approval requirements for reads and writes

The UI surfaces capability state for each enabled tool, including:

- available
- missing workspace
- missing credentials
- permission blocked
- approval gated

## Citations

The assistant keeps the existing `ChatFinalResponse` schema. In assistant mode,
`citations[].source_path` should be read as a generic source identifier rather
than only a local file path. It may refer to a local path, repository artifact,
or remote resource identifier.
