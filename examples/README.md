# Examples

These examples are small, runnable entry points that show the implemented
public APIs working together.

## Offline Examples

These examples do not make network calls:

- `minimal_tool.py`
  Defines a custom tool, registers it, and executes it through `ToolRuntime`.
- `builtins_direct.py`
  Registers built-in tools and executes them directly.
- `openai_wiring.py`
  Uses the OpenAI adapter shape without making an API call.
- `structured_response.py`
  Parses structured JSON output and executes it through `WorkflowExecutor`.
- `prompt_schema.py`
  Renders prompt instructions, parses prompt-style JSON, and executes it through
  `WorkflowExecutor`.

## Live OpenAI Example

- `openai_live.py`
  Makes a real OpenAI API call using `OPENAI_API_KEY`.

This example is intentionally not run in CI. It should fail with a clear
message if `OPENAI_API_KEY` is not set.

