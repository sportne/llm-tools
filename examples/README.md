# Examples

These examples are small, runnable entry points that show the implemented
public APIs working together.

For an interactive terminal UI over the same core surfaces, see the optional
Textual workbench in `llm_tools.apps.textual_workbench`.

## Offline Examples

These examples do not make network calls:

- `minimal_tool.py`
  Defines a custom tool, registers it, and executes it through `ToolRuntime`.
- `builtins_direct.py`
  Registers built-in tools and executes them directly.
- `openai_wiring.py`
  Uses the native-tool-calling adapter plus the provider layer without making a
  live API call.
- `structured_response.py`
  Uses the structured-output adapter through the provider layer and executes the
  parsed result through `WorkflowExecutor`.
- `prompt_schema.py`
  Uses the prompt-schema adapter through the provider layer and executes the
  parsed result through `WorkflowExecutor`.

## Live Ollama Example

- `openai_live.py`
  Uses the OpenAI-compatible provider layer against Ollama's local
  OpenAI-compatible endpoint.
  It defaults to `http://localhost:11434/v1` and model `gemma4`.

This example is intentionally not run in CI. It should fail with a clear
message if Ollama is not reachable.
