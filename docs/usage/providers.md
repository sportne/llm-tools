# Providers

`llm_tools.llm_providers` contains model-call clients that use the OpenAI Python
SDK against OpenAI-compatible endpoints.

The provider layer is separate from `llm_tools.llm_adapters`:

- adapters describe how to expose tools and parse model output
- providers handle transport, Instructor mode selection, and structured-call
  retries across compatible modes

## OpenAI-Compatible Provider

`OpenAICompatibleProvider` is the canonical provider client.

Constructor inputs include:

- `model`
- `api_key`
- `base_url`
- `mode_strategy`
- optional `default_request_params`

Convenience constructors:

- `OpenAICompatibleProvider.for_openai(...)`
- `OpenAICompatibleProvider.for_ollama(...)`

Treat `base_url` as a trust boundary. Prompts, structured responses, and
credentials are sent to that endpoint.

## Public Methods

The provider currently exposes:

- `run(...) -> ParsedModelResponse`
- `run_async(...) -> ParsedModelResponse`
- `run_structured(...) -> object`
- `run_structured_async(...) -> object`
- `list_available_models() -> list[str]`

`run(...)` and `run_async(...)` normalize model output through an
`ActionEnvelopeAdapter`.

`run_structured(...)` and `run_structured_async(...)` return the structured
payload directly without adapter-level normalization. They are the supported
surface for callers that need one structured call but not a full workflow turn.

## Strategy Modes

`mode_strategy` controls Instructor mode selection:

- `auto` (default): try `TOOLS`, then `JSON`, then `MD_JSON` for structured
  compatibility fallbacks
- `tools`
- `json`
- `md_json`

`auto` is a compatibility fallback, not a general retry policy. Unexpected
transport or authentication failures are surfaced immediately.

## Example

```python
from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import ToolContext
from llm_tools.workflow_api import WorkflowExecutor

adapter = ActionEnvelopeAdapter()
provider = OpenAICompatibleProvider.for_ollama(model="gemma4:26b")
executor = WorkflowExecutor(registry)

context = ToolContext(invocation_id="turn-1")
prepared = executor.prepare_model_interaction(adapter, context=context)

parsed = provider.run(
    adapter=adapter,
    messages=[{"role": "user", "content": "Read README.md"}],
    response_model=prepared.response_model,
)
turn_result = executor.execute_parsed_response(parsed, context)
```
