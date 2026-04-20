# Examples

These examples are small, runnable entry points that show the implemented
public APIs working together.

For the current repository shape, treat `README.md` as the practical overview,
the design docs as the architecture source of truth, and this directory as the
place for runnable examples and assistant config samples.

## Assistant Config Examples

These YAML files configure `llm_tools.apps.streamlit_assistant` for common
assistant usage modes.

The YAML file is optional. The Streamlit assistant can start without one, and
it can export the current non-secret session setup back into YAML when you want
a reusable preset. Config examples preselect tools, research defaults,
protection corpus paths, and sometimes a workspace root, but they do not grant
filesystem, subprocess, or network access. Those permissions still have to be
enabled per session in the Streamlit sidebar.

- `assistant_configs/local-only-chat.yaml`
  Use this for local workspace Q&A when you want file browsing and text search
  without remote data or durable research.
  Launch: `llm-tools-streamlit-assistant . --config examples/assistant_configs/local-only-chat.yaml`
- `assistant_configs/enterprise-data-chat.yaml`
  Use this when the assistant should start with remote enterprise read tools for
  Jira, Confluence, Bitbucket, and GitLab, including private-network
  OpenAI-compatible endpoints that work best with explicit JSON mode.
  Launch: `llm-tools-streamlit-assistant --config examples/assistant_configs/enterprise-data-chat.yaml`
- `assistant_configs/harness-research-chat.yaml`
  Use this for assistant chat that also launches durable harness-backed research
  sessions over a local workspace.
  Launch: `llm-tools-streamlit-assistant . --config examples/assistant_configs/harness-research-chat.yaml`

## Offline Examples

These examples do not make network calls:

- `minimal_tool.py`
  Defines a custom tool, registers it, and executes it through `ToolRuntime`.
- `builtins_direct.py`
  Registers built-in tools and executes them directly.
- `openai_wiring.py`
  Shows one-turn wiring with `ActionEnvelopeAdapter` and
  `OpenAICompatibleProvider` without a live API call.
- `async_model_turn.py`
  Uses async provider and workflow APIs with the same one-turn model/tool flow.
- `structured_response.py`
  Demonstrates the canonical action-envelope flow with a structured-output style
  payload.
- `prompt_schema.py`
  Demonstrates the same flow with a prompt-emitted JSON payload shape.

## Live Ollama Example

- `openai_live.py`
  Uses the OpenAI-compatible provider layer against Ollama's local
  OpenAI-compatible endpoint.
  It defaults to `http://localhost:11434/v1` and model `gemma4:26b`.

This example is intentionally not run in CI. It should fail with a clear
message if Ollama is not reachable.
