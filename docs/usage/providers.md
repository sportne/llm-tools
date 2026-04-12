# Providers

`llm_tools.llm_providers` contains model-call clients that use the OpenAI
Python SDK against OpenAI-compatible endpoints.

The provider layer is separate from `llm_tools.llm_adapters`:

- adapters describe how to expose tools and parse model output
- providers handle request construction and response extraction

## OpenAI-Compatible Provider

`OpenAICompatibleProvider` is the first concrete provider client.

It accepts:

- `model`
- `api_key`
- `base_url`
- optional default request parameters

It uses only the OpenAI Python SDK. It does not use vendor-native SDKs.

That means this layer can work with any actual provider that offers an
OpenAI-compatible endpoint, such as local Ollama setups or other compatible
gateways.

## Interaction Modes

`OpenAICompatibleProvider` exposes one entry point per interaction mode:

- `run_native_tool_calling(...)`
- `run_structured_output(...)`
- `run_prompt_schema(...)`
- `run_native_tool_calling_async(...)`
- `run_structured_output_async(...)`
- `run_prompt_schema_async(...)`

Each method:

1. exports tool descriptions through the supplied adapter
2. calls the model through the OpenAI SDK
3. extracts the raw response payload
4. parses it through the adapter into `ParsedModelResponse`

Providers do not execute tools directly.

The async methods use the OpenAI SDK's `AsyncOpenAI` client while preserving
the same request/response semantics as the sync methods.

## Example

```python
from llm_tools.llm_adapters import NativeToolCallingAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.workflow_api import WorkflowExecutor

adapter = NativeToolCallingAdapter()
provider = OpenAICompatibleProvider.for_ollama(model="gemma4:26b")
executor = WorkflowExecutor(registry)

parsed = provider.run_native_tool_calling(
    adapter=adapter,
    messages=[{"role": "user", "content": "Read README.md"}],
    registry=registry,
)
turn_result = executor.execute_parsed_response(parsed, context)
```

For runnable samples, see the [examples directory](../../examples/README.md).

Async usage follows the same shape with `await`:

```python
parsed = await provider.run_native_tool_calling_async(
    adapter=adapter,
    messages=[{"role": "user", "content": "Read README.md"}],
    registry=registry,
)
turn_result = await executor.execute_parsed_response_async(parsed, context)
```
