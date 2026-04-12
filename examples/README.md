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
  Shows one-turn wiring with `ActionEnvelopeAdapter` and
  `OpenAICompatibleProvider` without a live API call.
- `async_model_turn.py`
  Uses async provider and workflow APIs with the same one-turn model/tool flow.
- `structured_response.py`
  Demonstrates the same canonical action-envelope flow with a structured-output
  style payload.
- `prompt_schema.py`
  Demonstrates the same canonical action-envelope flow with a prompt-emitted
  JSON payload shape.

## Live Ollama Example

- `openai_live.py`
  Uses the OpenAI-compatible provider layer against Ollama's local
  OpenAI-compatible endpoint.
  It defaults to `http://localhost:11434/v1` and model `gemma4:26b`.

This example is intentionally not run in CI. It should fail with a clear
message if Ollama is not reachable.
