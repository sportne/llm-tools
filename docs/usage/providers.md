# Providers

`llm_tools.llm_providers` contains model-call clients that use the OpenAI
Python SDK against OpenAI-compatible endpoints.

The provider layer is separate from `llm_tools.llm_adapters`:

- adapters describe how to expose tools and parse model output
- providers handle request construction and response extraction

## OpenAI-Compatible Provider

`OpenAICompatibleProvider` is the canonical provider client.

It accepts:

- `model`
- `api_key`
- `base_url`
- `mode_strategy`
- optional `default_request_params`

It uses the OpenAI Python SDK for transport and Instructor for structured
response parsing. Treat `base_url` as a trust boundary: prompts, structured
responses, and credentials are sent to that endpoint. Prefer HTTPS for remote
providers; plaintext HTTP is only appropriate for local loopback-style setups
such as Ollama on `localhost`.

## Public Methods

The provider now exposes one sync and one async call surface:

- `run(...) -> ParsedModelResponse`
- `run_async(...) -> ParsedModelResponse`

Both methods accept:

- `adapter: ActionEnvelopeAdapter`
- `messages: Sequence[dict[str, Any]]`
- `response_model: type[BaseModel]`
- optional `request_params`

`response_model` should come from
`WorkflowExecutor.prepare_model_interaction(...)`.

## Strategy Modes

`mode_strategy` controls Instructor mode selection:

- `auto` (default): tries `TOOLS`, then `JSON`, then `MD_JSON` only for
  structured-output compatibility failures
- `tools`
- `json`
- `md_json`

`auto` is a compatibility fallback, not a general retry policy. Authentication,
transport, timeout, and other unexpected provider failures are surfaced
immediately instead of being replayed across modes.

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

Async usage:

```python
parsed = await provider.run_async(
    adapter=adapter,
    messages=[{"role": "user", "content": "Read README.md"}],
    response_model=prepared.response_model,
)
turn_result = await executor.execute_parsed_response_async(parsed, context)
```
