# Adapters

Adapters translate between LLM-facing interaction styles and the canonical
internal tool invocation model.

They are translation layers only. They do not own:

- runtime validation
- policy enforcement
- tool execution semantics

## Two Separate Concerns

Every adapter handles two independent tasks:

- tool exposure
- model output parsing

Tool exposure describes available tools to an external model. Parsing converts
model output into a canonical `ParsedModelResponse`.

## Parsed Model Response

Adapters normalize one model turn into either:

- `invocations`: one-or-more `ToolInvocationRequest` objects
- `final_response`: a plain assistant response string

Exactly one mode is allowed at a time.

## Available Adapters

### OpenAI tool calling

`OpenAIToolCallingAdapter`:

- exports OpenAI SDK-compatible tool definitions
- parses OpenAI tool-call payloads
- parses plain final assistant text when no tool call is present

### Structured response

`StructuredResponseAdapter`:

- exports a canonical JSON schema
- expects an envelope with `actions` or `final_response`
- parses JSON strings or decoded payloads

### Prompt schema

`PromptSchemaAdapter`:

- renders prompt instructions describing the expected JSON envelope
- parses prompt-returned JSON post hoc
- supports one deterministic repair pass for fenced JSON

## Executing Adapter Output

`WorkflowExecutor` is the thin one-turn bridge above adapters and runtime:

```python
from llm_tools.llm_adapters import StructuredResponseAdapter
from llm_tools.tool_api import ToolContext, ToolRegistry
from llm_tools.workflow_api import WorkflowExecutor

adapter = StructuredResponseAdapter()
executor = WorkflowExecutor(registry)

turn_result = executor.execute_model_output(
    adapter,
    {"actions": [{"tool_name": "read_file", "arguments": {"path": "README.md"}}]},
    ToolContext(invocation_id="turn-1"),
)
```

The workflow layer:

- parses one model output
- executes parsed invocations sequentially when present
- returns a final-response-only result when no tool is needed

It does not perform multi-turn loops, replanning, or follow-up model calls.

