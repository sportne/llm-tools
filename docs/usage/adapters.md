# Adapters

`llm_tools.llm_adapters` contains pure interaction-mode translators.

Adapters do two things:

- expose available tools in a model-facing format
- parse model output into a canonical `ParsedModelResponse`

They do not make network calls, execute tools, or enforce policy.

## Parsed Model Response

Adapters normalize one model turn into exactly one of these shapes:

- `invocations`: one or more `ToolInvocationRequest` objects
- `final_response`: a plain assistant response string

Exactly one mode is allowed at a time.

## Available Adapters

### Native tool calling

`NativeToolCallingAdapter`:

- exports canonical native tool definitions
- parses native tool-call payloads
- parses plain final assistant text when no tool call is present

### Structured output

`StructuredOutputAdapter`:

- exports a canonical JSON schema
- expects an envelope with `actions` or `final_response`
- parses JSON strings or decoded payloads

### Prompt schema

`PromptSchemaAdapter`:

- renders prompt instructions for the canonical JSON envelope
- parses prompt-returned JSON post hoc
- supports one deterministic repair pass for fenced JSON

## Providers vs Adapters

Adapters describe interaction modes. Providers handle transport.

`llm_tools.llm_providers` contains provider clients built on the OpenAI Python
SDK for OpenAI-compatible endpoints such as OpenAI-compatible gateways,
Ollama, and similar compatibility surfaces.

The usual flow is:

1. choose an adapter for the interaction mode
2. call a provider client
3. receive a `ParsedModelResponse`
4. execute it with `WorkflowExecutor` when tool invocations are present

## Executing Adapter Output

`WorkflowExecutor` is the thin one-turn bridge above adapters and runtime:

```python
from llm_tools.llm_adapters import StructuredOutputAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import ToolContext, ToolRegistry
from llm_tools.workflow_api import WorkflowExecutor

adapter = StructuredOutputAdapter()
provider = OpenAICompatibleProvider(model="demo-model")
executor = WorkflowExecutor(registry)

parsed = provider.run_structured_output(
    adapter=adapter,
    messages=[{"role": "user", "content": "Read README.md"}],
    registry=registry,
)
turn_result = executor.execute_parsed_response(
    parsed,
    ToolContext(invocation_id="turn-1"),
)
```

The workflow layer:

- parses one model output
- executes parsed invocations sequentially when present
- returns a final-response-only result when no tool is needed

It does not perform multi-turn loops, replanning, or follow-up model calls.
