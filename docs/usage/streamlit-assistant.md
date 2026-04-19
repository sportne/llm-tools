# Streamlit Assistant

`llm_tools.apps.streamlit_assistant` is the main interactive client in the
repository. It sits on top of the existing tool, provider, workflow, and
harness layers without changing their public contracts.

## Install

```bash
~/.venvs/llm-tools/bin/python -m pip install -e .[streamlit]
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

The optional directory becomes the initial workspace root. That selection does
not grant filesystem or subprocess access by itself.

## Config Shape

`StreamlitAssistantConfig` currently includes:

- `llm`
- `session`
- `tool_limits`
- `policy`
- `protection`
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

policy:
  enabled_tools: []

protection:
  enabled: false

workspace:
  default_root: .

research:
  default_max_turns: 6
  max_recent_sessions: 8
```

## Session Behavior

Fresh sessions start with:

- no network permission
- no filesystem permission
- no subprocess permission
- write approvals enabled by default for `local_write` and `external_write`

The assistant starts with no enabled tools unless `policy.enabled_tools` names a
default subset. Tool readiness copy distinguishes states such as ready, needs
workspace, needs credentials, blocked by permissions, and approval on use.

"New session from current setup" clones the current runtime choices into a new
session. If you submit another prompt while a turn is still running, the UI
keeps a single queued follow-up prompt for that session and sends it when the
active turn finishes or is stopped.

## Research Sessions

The assistant can launch harness-backed research sessions from the sidebar and
inspect them in the main pane.

Current research session state copy distinguishes:

- running
- awaiting approval
- resumable
- stopped
- summarized

A session is treated as summarized for the active assistant chat only after its
summary has been inserted back into that chat transcript.

When `research.store_dir` is omitted, durable research state is stored under
`~/.llm-tools/assistant/streamlit/research`.
