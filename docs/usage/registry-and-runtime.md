# Registry and Runtime

`ToolRegistry` stores concrete tool instances by canonical name. `ToolRuntime`
executes one `ToolInvocationRequest` at a time with validation, policy checks,
error normalization, execution recording, and provenance capture.

## Registering Tools

```python
from llm_tools.tool_api import ToolRegistry

registry = ToolRegistry()
registry.register(EchoTool())
```

The registry preserves registration order, rejects duplicate tool names, and
exposes public read-only bindings through `list_bindings()`.

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

The host passes a `ToolContext` into `ToolRuntime`. The runtime then issues a
`ToolExecutionContext` to the concrete tool implementation.

`ToolRuntime` handles:

- tool lookup
- policy evaluation
- input validation
- tool invocation
- output validation
- normalized success/failure envelopes
- execution record capture

## Result Shape

Runtime execution returns a `ToolResult`.

Important fields include:

- `ok`, `output`, and normalized `error`
- `logs` and `artifacts`
- `source_provenance`
- `metadata["execution_record"]` with a serialized `ExecutionRecord`

`ExecutionRecord` and `ToolResult` both carry `source_provenance`, which is now
used by higher-level protection and research flows, not just observability.

Some built-in read-oriented tools also write internal workspace cache data such
as converted-document cache entries under `.llm_tools/cache/read_file`. That is
runtime-mediated behavior, not a separate public runtime contract.

## One-Turn Workflow Bridge

`WorkflowExecutor` is the bridge from parsed model output to sequential tool
execution:

```python
from llm_tools.workflow_api import WorkflowExecutor

executor = WorkflowExecutor(registry)
```

`WorkflowExecutor` is still one-turn scoped. It can execute a parsed batch of
invocations or return a final response, but it does not own assistant session
state, research orchestration, or durable harness resume logic.
