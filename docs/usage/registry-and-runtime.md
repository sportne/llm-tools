# Registry and Runtime

`ToolRegistry` stores concrete tool instances by canonical name. `ToolRuntime`
executes one `ToolInvocationRequest` at a time with validation, policy checks,
error normalization, and execution recording.

## Registering Tools

```python
from llm_tools.tool_api import ToolRegistry

registry = ToolRegistry()
registry.register(EchoTool())
```

The registry:

- preserves registration order
- rejects duplicate tool names
- exposes canonical `ToolSpec` metadata for listing and filtering

Built-in tool groups also provide registration helpers:

```python
from llm_tools.tools import register_filesystem_tools, register_text_tools

register_filesystem_tools(registry)
register_text_tools(registry)
```

## Executing a Tool

```python
from llm_tools.tool_api import ToolContext, ToolInvocationRequest, ToolRuntime

runtime = ToolRuntime(registry)

result = runtime.execute(
    ToolInvocationRequest(tool_name="echo", arguments={"value": "hello"}),
    ToolContext(invocation_id="inv-1"),
)
```

`ToolRuntime` handles:

- tool lookup
- policy evaluation
- input validation
- tool invocation
- output validation
- normalized success/failure envelopes
- execution record capture

## Result Shape

Runtime execution always returns a `ToolResult`.

Success shape:

- `ok=True`
- `output` contains serialized output model data
- `error=None`

Failure shape:

- `ok=False`
- `output=None`
- `error` contains a normalized `ToolError`

`ToolResult.metadata["execution_record"]` includes a serialized
`ExecutionRecord` with timing, validated input, policy decision, logs,
artifacts, and error information.

## One-Turn Workflow Bridge

If you want to go from adapter output to tool execution without building your
own glue code, use `WorkflowExecutor`:

```python
from llm_tools.workflow_api import WorkflowExecutor

executor = WorkflowExecutor(registry)
```

`WorkflowExecutor` consumes a parsed model turn and either:

- returns a final response with no tool execution
- or executes one-or-more parsed tool invocations sequentially

It still stays one-turn scoped. It does not replan, loop, or make a second
model call.

