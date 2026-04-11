# Defining Tools

`llm-tools` uses a strict class-based tool contract built on Pydantic v2.

Every concrete tool must define:

- `spec`: canonical metadata as a `ToolSpec`
- `input_model`: a Pydantic model for validated input
- `output_model`: a Pydantic model for validated output
- `invoke(context, args) -> output_model`

`invoke()` should return the declared output model, not a raw dict. The runtime
will validate both the input payload and the returned output model again at the
execution boundary.

## Minimal Example

```python
from pydantic import BaseModel

from llm_tools.tool_api import Tool, ToolContext, ToolSpec


class EchoInput(BaseModel):
    value: str


class EchoOutput(BaseModel):
    echoed: str


class EchoTool(Tool[EchoInput, EchoOutput]):
    spec = ToolSpec(
        name="echo",
        description="Return the provided value.",
        tags=["example"],
    )
    input_model = EchoInput
    output_model = EchoOutput

    def invoke(self, context: ToolContext, args: EchoInput) -> EchoOutput:
        return EchoOutput(echoed=f"{context.invocation_id}:{args.value}")
```

## Tool Metadata

`ToolSpec` is the canonical metadata source. It is where you declare:

- tool name and description
- tags
- side effects
- risk level
- capability requirements such as filesystem, network, or subprocess access
- required secrets

Provider-facing schemas are generated from this canonical metadata plus the
input model. They are not the source of truth.

## Best Practices

- keep tools small and focused
- make `spec.name` stable and explicit
- use input and output models that are easy to validate and serialize
- append meaningful runtime logs/artifacts to `context.logs` and
  `context.artifacts` when the tool performs interesting work
- raise normal Python exceptions from tool code and let `ToolRuntime` normalize
  them into `ToolResult`

## Next Steps

After defining a tool, you usually:

1. register it in a `ToolRegistry`
2. execute it through `ToolRuntime`
3. optionally expose it through an adapter or `WorkflowExecutor`

See:

- [Registry and Runtime](registry-and-runtime.md)
- [Adapters](adapters.md)

