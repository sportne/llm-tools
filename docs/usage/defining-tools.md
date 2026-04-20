# Defining Tools

`llm-tools` uses a strict class-based tool contract built on Pydantic v2.

Every concrete tool must define:

- `spec`: canonical metadata as a `ToolSpec`
- `input_model`: a Pydantic model for validated input
- `output_model`: a Pydantic model for validated output
- at least one implementation method:
  - `_invoke_impl(context, args) -> output_model`
  - `_ainvoke_impl(context, args) -> output_model`

The public `invoke()` / `ainvoke()` wrappers are runtime-owned entry points.
Calling them directly without a runtime-issued execution permit raises.

## Minimal Example

```python
from pydantic import BaseModel

from llm_tools.tool_api import Tool, ToolExecutionContext, ToolSpec


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

    def _invoke_impl(
        self, context: ToolExecutionContext, args: EchoInput
    ) -> EchoOutput:
        return EchoOutput(echoed=f"{context.invocation_id}:{args.value}")
```

## Tool Metadata

`ToolSpec` is the canonical metadata source. It is where you declare:

- stable tool name and description
- tags
- side effects and risk level
- capability requirements such as filesystem, network, or subprocess access
- required secrets
- whether the tool writes internal workspace cache data

Provider-facing schemas are generated from `ToolSpec` plus the input model.

## Best Practices

- keep tools small and focused
- keep `spec.name` stable
- return the declared output model, not a raw dict
- use `ToolExecutionContext` for runtime logs, artifacts, and source provenance
- raise normal Python exceptions from tool code and let `ToolRuntime`
  normalize them into `ToolResult`
