# Adapters

`llm_tools.llm_adapters` now exposes one canonical adapter:
`ActionEnvelopeAdapter`.

The adapter is intentionally transport-agnostic. It handles only two concerns:

- build the canonical structured action envelope for visible tools
- parse model payloads into canonical `ParsedModelResponse`

It does not call models, execute tools, or enforce policy.

## Canonical Turn Shape

`ParsedModelResponse` normalizes one model turn into exactly one of:

- `invocations`: one or more `ToolInvocationRequest` objects
- `final_response`: a plain assistant response string

Exactly one mode is valid at a time.

## Action Envelope

`ActionEnvelopeAdapter` uses one canonical envelope:

- tool path: `{"actions": [...], "final_response": null}`
- final-response path: `{"actions": [], "final_response": "..."}`

When tools are visible, `actions` are typed per tool:

- `tool_name` is constrained to registered/visible tool names
- `arguments` are validated against each tool’s `input_model`

## Typical Flow

Use `WorkflowExecutor.prepare_model_interaction(...)` to derive the typed
response model and JSON schema once per turn, then pass that response model to
the provider:

```python
from llm_tools.llm_adapters import ActionEnvelopeAdapter
from llm_tools.llm_providers import OpenAICompatibleProvider
from llm_tools.tool_api import ToolContext
from llm_tools.workflow_api import WorkflowExecutor

adapter = ActionEnvelopeAdapter()
executor = WorkflowExecutor(registry)
provider = OpenAICompatibleProvider.for_ollama(model="gemma4:26b")

context = ToolContext(invocation_id="turn-1")
prepared = executor.prepare_model_interaction(adapter, context=context)

parsed = provider.run(
    adapter=adapter,
    messages=[{"role": "user", "content": "Read README.md"}],
    response_model=prepared.response_model,
)

turn_result = executor.execute_parsed_response(parsed, context)
```

The workflow layer remains a one-turn bridge. It does not add replanning,
multi-turn loops, or follow-up model calls.
