# ARCHITECTURE.md

## 1. Purpose

This document defines the architecture for `llm-tools`, a low-level Python library for defining, registering, executing, and exposing typed tools for LLM and non-LLM applications.

The project is intentionally **not** an agent framework. It does not treat planning, memory, prompt orchestration, or workflows as foundational concerns. Instead, it provides a strict, typed substrate for tools that higher-level systems may build on top of later.

The first implemented subsystem is `tool_api`. Additional layers such as
`llm_adapters`, `llm_providers`, built-in `tools`, and later `workflow_api` are
built on top of that substrate.

---

## 2. Architectural overview

The system is organized into five primary layers:

1. `tool_api`
2. `llm_adapters`
3. `llm_providers`
4. `tools`
5. `workflow_api`

Relationship:

```text
External callers
  ├─ Python application code
  ├─ CLI / service layers
  └─ LLM integrations
          │
          ▼
    llm_providers
          │
          ▼
    llm_adapters
          │
          ▼
       tool_api
          │
          ▼
         tools
```

### 2.1 Layer responsibilities

#### `tool_api`

Defines the canonical internal model for:

* tool metadata
* tool input and output contracts
* invocation requests
* results
* errors
* registry
* runtime
* policy
* observability

#### `llm_adapters`

Translate between model-facing interaction modes and the canonical internal
invocation model.

Supported modes:

* native tool calling
* structured output action selection
* prompt-schema action selection

#### `llm_providers`

Own model-call setup and response extraction using the OpenAI Python SDK
against OpenAI-compatible endpoints.

#### `tools`

Concrete tool implementations built against `tool_api`.

Examples:

* filesystem tools
* git tools
* Atlassian tools
* text tools

#### `workflow_api`

A composition layer for explicit multi-step tool execution. It is intentionally
not part of the base tool abstraction. In v0.1 it provides a thin one-turn
bridge that takes a parsed adapter result and executes the returned invocations
sequentially without any replanning or follow-up model calls.

---

## 3. Architectural constraints

The architecture is governed by the following decisions.

### 3.1 Tools are class-based

A tool is a Python class. Tools are not prompt files, manifests, or static function schemas.

### 3.2 Pydantic is the structured data backbone

All structured runtime entities use Pydantic v2.

This includes:

* tool metadata
* invocation requests
* contexts
* results
* errors
* policy decisions
* execution records
* tool input models
* tool output models

### 3.3 Registration is pure Python in v0.1

Tools are instantiated and explicitly registered in Python code.

No manifest-based discovery or plugin loader is included in v0.1.

### 3.4 `ToolSpec` is canonical metadata

Each tool exposes a `spec: ToolSpec` as a class attribute.

`ToolSpec` is the source of truth for tool metadata used by:

* registry
* runtime
* policy
* adapters

### 3.5 `invoke()` returns the declared output model

A tool’s `invoke(context, args)` method must return an instance of the declared `output_model`.

Returning raw dicts is not supported in v0.1.

### 3.6 The runtime is strict early

The runtime validates:

* raw request arguments into the declared input model before invocation
* the returned object against the declared output model after invocation

Strictness is preferred early because loosening later is easier than tightening later.

### 3.7 Provider-facing schemas are derived artifacts

Native tool definitions, structured output schemas, and prompt-rendered schemas
are generated from canonical internal models. They are not the source of truth.

---

## 4. High-level component model

The main architectural components are:

* `ToolSpec`
* `ToolContext`
* `ToolInvocationRequest`
* `ToolResult`
* `ToolError`
* `Tool`
* `ToolRegistry`
* `ToolRuntime`
* `ToolPolicy`
* `PolicyDecision`
* `ExecutionRecord`
* `LLMAdapter`

These form the stable system boundary for v0.1.

---

## 5. Canonical execution flow

```text
Caller
  ↓
ToolInvocationRequest
  ↓
ToolRuntime.execute(...)
  ↓
Registry lookup
  ↓
Policy evaluation
  ↓
Input validation
  ↓
Tool.invoke(...)
  ↓
Output validation
  ↓
ToolResult
  ↓
ExecutionRecord
```

### 5.1 Step-by-step flow

#### 1. Caller creates or receives a `ToolInvocationRequest`

The request may originate from:

* Python application code
* a CLI or service wrapper
* an LLM adapter
* a future workflow step

#### 2. Runtime resolves the tool

`ToolRuntime` queries `ToolRegistry` by tool name.

#### 3. Runtime evaluates policy

The runtime checks whether the invocation is allowed under the active policy and context.

#### 4. Runtime validates the input payload

The raw argument payload is validated against the tool’s declared `input_model`.

#### 5. Runtime invokes the tool

The tool receives:

* `ToolContext`
* an already-validated input model instance

#### 6. Runtime validates the output

The runtime checks that the returned value is an instance of, or can be validated as, the declared `output_model`.

#### 7. Runtime normalizes the outcome

The runtime returns a `ToolResult`, not raw tool output.

#### 8. Runtime captures observability data

Execution metadata is captured in an `ExecutionRecord` regardless of success or failure.

---

## 6. Package structure

```text
project/
  src/
    llm_tools/
      __init__.py
      tool_api/
        __init__.py
        models.py
        tool.py
        registry.py
        runtime.py
        policy.py
        observability.py
        errors.py
        decorators.py
      llm_adapters/
        __init__.py
        base.py
        native_tool_calling.py
        structured_output.py
        prompt_schema.py
      llm_providers/
        __init__.py
        openai_compatible.py
      tools/
        __init__.py
        filesystem/
          __init__.py
          read_file.py
          write_file.py
          list_directory.py
          register.py
        git/
          __init__.py
          run_git_status.py
          run_git_diff.py
          run_git_log.py
          register.py
        atlassian/
          __init__.py
          search_jira.py
          read_jira_issue.py
          register.py
        text/
          __init__.py
          search_text.py
          register.py
      workflow_api/
        __init__.py
        models.py
        executor.py

  tests/
```

Step 0 scaffolds these packages and repository tooling only. The concrete
interfaces and behavior described below begin in later implementation steps.

### 6.1 Naming rationale

* `tool_api` is the foundational subsystem, not a generic “core.”
* `llm_adapters` is explicitly layered above `tool_api`.
* `llm_providers` is layered above `llm_adapters`.
* `tools` contains concrete implementations, not framework internals.
* `workflow_api` is separated to avoid contaminating the base abstraction with orchestration concerns.

---

## 7. Data model architecture

## 7.1 Enumerations and controlled values

The architecture should use stable enums where appropriate.

Initial enums:

* `SideEffectClass`
* `RiskLevel`
* `ErrorCode`
* `PolicyVerdict`

Example:

```python
from enum import Enum

class SideEffectClass(str, Enum):
    NONE = "none"
    LOCAL_READ = "local_read"
    LOCAL_WRITE = "local_write"
    EXTERNAL_READ = "external_read"
    EXTERNAL_WRITE = "external_write"
```

---

## 7.2 ToolSpec

`ToolSpec` is the canonical metadata model for a tool.

Responsibilities:

* identify the tool
* describe the tool
* expose execution metadata
* expose risk and environment requirements

Suggested shape:

```python
from pydantic import BaseModel, Field

class ToolSpec(BaseModel):
    name: str = Field(min_length=1)
    version: str = Field(default="0.1.0")
    description: str
    tags: list[str] = Field(default_factory=list)

    side_effects: SideEffectClass = SideEffectClass.NONE
    idempotent: bool = True
    deterministic: bool = True
    timeout_seconds: int | None = None

    risk_level: str = "low"
    requires_network: bool = False
    requires_filesystem: bool = False
    requires_subprocess: bool = False
    required_secrets: list[str] = Field(default_factory=list)

    cost_hint: str | None = None
```

### 7.2.1 Placement

Each concrete tool must expose `spec: ToolSpec` as a class attribut eand the contract is that the runtime and registry can read `tool.spec`.

---

## 7.3 ToolContext

`ToolContext` carries runtime information for one invocation.

Suggested shape:

```python
from pydantic import BaseModel, Field
from typing import Any

class ToolContext(BaseModel):
    invocation_id: str
    workspace: str | None = None
    env: dict[str, str] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### Notes

* `ToolContext` is intentionally generic.
* It should not hardcode planner state, memory state, or UI concerns.

---

## 7.4 ToolInvocationRequest

This is the canonical invocation request model.

Suggested shape:

```python
from pydantic import BaseModel, Field
from typing import Any

class ToolInvocationRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)
```

This is the normalization point across:

* native tool-calling mode
* structured output mode
* prompt-schema mode
* direct Python callers

---

## 7.5 ToolError

All failures crossing the runtime boundary normalize into a `ToolError`.

Suggested shape:

```python
from pydantic import BaseModel, Field
from typing import Any

class ToolError(BaseModel):
    code: str
    message: str
    retryable: bool = False
    details: dict[str, Any] = Field(default_factory=dict)
```

Initial error categories:

* `tool_not_found`
* `input_validation_error`
* `output_validation_error`
* `policy_denied`
* `timeout`
* `dependency_missing`
* `execution_failed`
* `runtime_error`

---

## 7.6 ToolResult

`ToolResult` is the canonical result envelope returned by the runtime.

Suggested shape:

```python
from pydantic import BaseModel, Field
from typing import Any

class ToolResult(BaseModel):
    ok: bool
    tool_name: str
    tool_version: str

    output: dict[str, Any] | None = None
    error: ToolError | None = None

    logs: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

### Notes

* `ToolResult` is the only outward-facing execution result.
* Tool implementations do not return this directly.
* `output` contains serialized data from the declared `output_model`.

---

## 7.7 PolicyDecision

Policy evaluation always produces an explicit result.

Suggested shape:

```python
from pydantic import BaseModel, Field
from typing import Any

class PolicyDecision(BaseModel):
    allowed: bool
    reason: str
    requires_approval: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)
```

Even if approval mechanics are minimal in v0.1, the decision object should exist from the start.

---

## 7.8 ExecutionRecord

Observability is represented as a structured model.

Suggested shape:

```python
from pydantic import BaseModel, Field
from typing import Any

class ExecutionRecord(BaseModel):
    invocation_id: str
    tool_name: str
    tool_version: str

    started_at: str
    ended_at: str | None = None
    duration_ms: int | None = None

    request: ToolInvocationRequest
    validated_input: dict[str, Any] | None = None
    redacted_input: dict[str, Any] | None = None

    ok: bool | None = None
    error_code: str | None = None

    policy_decision: PolicyDecision | None = None
    logs: list[str] = Field(default_factory=list)
    artifacts: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
```

---

## 8. Tool class architecture

## 8.1 Base Tool contract

Tools are class-based and generic over input and output model types.

Suggested conceptual shape:

```python
from abc import ABC, abstractmethod
from pydantic import BaseModel
from typing import Generic, TypeVar

InputT = TypeVar("InputT", bound=BaseModel)
OutputT = TypeVar("OutputT", bound=BaseModel)

class Tool(ABC, Generic[InputT, OutputT]):
    spec: ClassVar[ToolSpec]
    input_model: ClassVar[type[InputT]]
    output_model: ClassVar[type[OutputT]]

    @abstractmethod
    def invoke(self, context: ToolContext, args: InputT) -> OutputT:
        raise NotImplementedError
```

### Required contract

Every concrete tool must provide:

* `spec`
* `input_model`
* `output_model`
* `invoke(context, args)`

### Strict output requirement

`invoke()` must return the declared `output_model`, not a dict.

This is a deliberate v0.1 constraint.

---

## 8.2 Why the runtime owns validation

Validation is centralized in the runtime rather than delegated to individual tools.

This gives:

* consistent behavior
* consistent error normalization
* consistent observability
* cleaner tool implementations

Tool implementations should focus on business logic, not parsing or envelope management.

---

## 8.3 Optional decorator layer

A decorator API may be added later for convenience, but it must compile down to the same class-based internal model.

For v0.1:

* class-based definitions are primary
* decorators are optional convenience only

---

## 9. Registry architecture

## 9.1 Responsibilities

`ToolRegistry` is responsible for:

* registering tool instances
* looking up tools by name
* listing tool specs
* filtering tools by metadata

It is not responsible for:

* execution
* policy
* planning
* workflow resolution

---

## 9.2 Suggested shape

```python
class ToolRegistry:
    def register(self, tool: Tool) -> None:
        ...

    def get(self, name: str) -> Tool:
        ...

    def list_tools(self) -> list[ToolSpec]:
        ...

    def filter_tools(
        self,
        *,
        tags: list[str] | None = None,
        side_effects: list[SideEffectClass] | None = None,
    ) -> list[ToolSpec]:
        ...
```

### Duplicate handling

In v0.1:

* duplicate tool names raise a registration error
* version-aware multi-registration is deferred

This keeps lookup semantics simple.

---

## 10. Runtime architecture

## 10.1 Responsibilities

`ToolRuntime` orchestrates one invocation.

Responsibilities:

* resolve tool from registry
* evaluate policy
* validate input arguments
* invoke the tool
* validate output
* normalize errors
* build `ToolResult`
* capture `ExecutionRecord`

---

## 10.2 Suggested interface

```python
class ToolRuntime:
    def __init__(
        self,
        registry: ToolRegistry,
        policy: ToolPolicy | None = None,
    ) -> None:
        ...

    def execute(
        self,
        request: ToolInvocationRequest,
        context: ToolContext,
    ) -> ToolResult:
        ...
```

---

## 10.3 Internal execution phases

The runtime should break work into explicit phases:

1. `resolve_tool`
2. `evaluate_policy`
3. `validate_input`
4. `invoke_tool`
5. `validate_output`
6. `normalize_result`
7. `record_execution`

This separation improves testability and clarity.

---

## 10.4 Input validation

The runtime validates the raw `arguments` payload against the tool’s `input_model` before invoking the tool.

Failure becomes:

* `input_validation_error`

Tools should never receive unvalidated raw dicts.

---

## 10.5 Output validation

The runtime validates the returned object against the tool’s `output_model`.

Expected behavior in v0.1:

* correct output model instance → accepted
* wrong type or invalid output → normalized as `output_validation_error`

This preserves a strict tool contract early.

---

## 10.6 Timeout handling

Timeout support should be designed into the runtime even if the first implementation is modest.

For v0.1:

* `ToolSpec.timeout_seconds` is the declared hint
* enforcement may initially be partial depending on execution mode

The architecture should not assume every Python tool can be safely interrupted.

---

## 10.7 Error normalization

All exceptions crossing the runtime boundary are normalized.

Examples:

* registry miss → `tool_not_found`
* Pydantic input validation failure → `input_validation_error`
* policy denial → `policy_denied`
* tool-thrown exception → `execution_failed`
* invalid returned output → `output_validation_error`

This is essential for predictable caller behavior.

---

## 11. Policy architecture

## 11.1 Responsibilities

Policy decides whether a tool invocation is permitted under the current constraints.

Policy may consider:

* tool name
* tags
* side-effect class
* network/filesystem/subprocess requirements
* secret requirements
* invocation context

---

## 11.2 Suggested model

```python
from pydantic import BaseModel, Field

class ToolPolicy(BaseModel):
    allowed_tools: set[str] | None = None
    denied_tools: set[str] = Field(default_factory=set)

    allowed_tags: set[str] | None = None
    denied_tags: set[str] = Field(default_factory=set)

    allowed_side_effects: set[SideEffectClass] = Field(
        default_factory=lambda: {
            SideEffectClass.NONE,
            SideEffectClass.LOCAL_READ,
        }
    )

    require_approval_for: set[SideEffectClass] = Field(default_factory=set)
    allow_network: bool = True
    allow_filesystem: bool = True
    allow_subprocess: bool = True
```

---

## 11.3 Policy evaluation

Policy evaluation produces a `PolicyDecision`, not just a boolean.

For v0.1:

* if approval would be required and no approval mechanism exists, runtime may deny the invocation conservatively

---

## 12. Observability architecture

## 12.1 Requirements

Each invocation should capture:

* tool identity
* timing
* request
* validated input
* redacted input
* policy result
* result status
* error classification
* logs
* artifacts

---

## 12.2 Logging strategy

The runtime should collect structured execution data and attach it to:

* `ToolResult`
* `ExecutionRecord`

Tool-specific logging hooks may be added later, but v0.1 can begin with runtime-managed collection.

---

## 12.3 Redaction

The architecture should reserve space for sensitive-field redaction.

A simple early implementation may:

* redact by field name
* support later extension for tool-specific redaction rules

---

## 13. LLM adapter and provider architecture

## 13.1 Design goal

Adapters translate between model-facing interaction styles and the canonical internal invocation model.

Adapters must not:

* redefine tool semantics
* own validation logic
* own policy logic
* bypass the runtime

They are translation layers only.

---

## 13.2 Two separate concerns

The architecture distinguishes:

### Tool exposure

How available tools are described to a model.

### Invocation parsing

How model output is interpreted as `ToolInvocationRequest` objects.

These are conceptually separate even if some implementations combine them.

---

## 13.3 Base conceptual interface

```python
class LLMAdapter(ABC):
    @abstractmethod
    def export_tool_descriptions(
        self,
        specs: list[ToolSpec],
        input_models: dict[str, type[BaseModel]],
    ) -> object:
        raise NotImplementedError

    @abstractmethod
    def parse_model_output(self, payload: object) -> ParsedModelResponse:
        raise NotImplementedError
```

---

## 13.4 Native tool-calling adapter

Responsibilities:

* export canonical native tool schemas
* parse returned model payloads
* normalize them into either tool invocations or a final assistant response

This adapter uses canonical `ToolSpec` and `input_model` information as source material.

---

## 13.5 Structured output adapter

Responsibilities:

* define a structured action schema for model output
* parse structured JSON into either tool invocations or a final assistant response

Suggested action envelope:

```python
from pydantic import BaseModel, Field
from typing import Any

class ToolAction(BaseModel):
    tool_name: str
    arguments: dict[str, Any] = Field(default_factory=dict)

class ToolActionEnvelope(BaseModel):
    actions: list[ToolAction]
    final_response: str | None = None
```

Even if v0.1 typically executes one action at a time, using an envelope creates a clean future path.

---

## 13.6 Prompt-schema adapter

Responsibilities:

* render prompt instructions for the expected JSON action shape
* parse and validate model output post hoc
* support repair/retry behavior where useful

This is the least reliable mode and is treated as fallback compatibility mode.

---

## 13.7 Provider layer

Provider clients sit above adapters and own transport concerns.

Responsibilities:

* call models through the OpenAI Python SDK
* target OpenAI-compatible endpoints via configurable `base_url`
* construct requests from adapter-exported artifacts
* extract raw response payloads and hand them back to adapters for parsing

Provider clients must not:

* execute tools directly
* reimplement runtime validation or policy
* depend on vendor-native SDKs in v0.1

---

## 14. Concrete tools architecture

## 14.1 Tool implementation modules

Concrete tools live under `src/llm_tools/tools/`.

Example domains:

* `src/llm_tools/tools/filesystem`
* `src/llm_tools/tools/git`
* `src/llm_tools/tools/atlassian`
* `src/llm_tools/tools/text`

Each tool is a normal subclass of `Tool`.

---

## 14.2 Registration helpers

Because registration is pure Python, each domain package should provide a helper.

Example:

```python
def register_filesystem_tools(registry: ToolRegistry) -> None:
    registry.register(ReadFileTool())
    registry.register(WriteFileTool())
    registry.register(ListDirectoryTool())
```

This gives clean explicit loading without manifests.

---

## 14.3 Built-in tool philosophy

Built-in tools serve as:

* useful starter capabilities
* reference implementations
* integration-test fixtures

They should not become tightly coupled to runtime internals.

---

## 15. Workflow architecture (deferred)

`workflow_api` is intentionally separate from the base abstraction.

A workflow should be modeled as an explicit composition of tool invocations layered on top of `tool_api`.

Constraint:

* workflows consume tools
* tools do not depend on workflows

For v0.1:

* workflow architecture may be described
* implementation may be deferred

---

## 16. Dependency direction

Required dependency direction:

```text
tool_api        ← foundational
llm_adapters    ← depends on tool_api
llm_providers   ← depends on llm_adapters
tools           ← depends on tool_api
workflow_api    ← depends on tool_api
applications    ← compose everything
```

Prohibited directions:

* `tool_api` must not depend on `llm_adapters`
* `tool_api` must not depend on `llm_providers`
* `tool_api` must not depend on `tools`
* `tools` should not depend on `llm_adapters`
* `tools` should not depend on `llm_providers`
* `tools` should not depend on `workflow_api`

---

## 17. Testing architecture

## 17.1 Testing layers

Testing should happen at several levels.

### Unit tests

* individual tools
* registry behavior
* policy behavior
* schema export behavior

### Runtime tests

* input validation
* output validation
* normalized failure behavior
* observability capture

### Adapter tests

* native tool schema generation
* structured output parsing
* prompt-schema parsing and repair behavior

### Integration tests

* register real tools
* execute through runtime
* exercise adapters end-to-end

---

## 17.2 Testability principles

To keep the system testable:

* runtime phases should be explicit
* adapters should be mostly pure transformations
* tools should stay small where possible
* I/O-heavy tools should isolate side effects cleanly

---

## 18. Initial implementation order

### Phase 1: Canonical models

Implement:

* enums
* `ToolSpec`
* `ToolContext`
* `ToolInvocationRequest`
* `ToolError`
* `ToolResult`
* `PolicyDecision`
* `ExecutionRecord`

### Phase 2: Base tool and registry

Implement:

* `Tool`
* `ToolRegistry`
* registration error handling

### Phase 3: Runtime

Implement:

* input validation
* policy evaluation
* execution
* output validation
* normalized results
* execution records

### Phase 4: Built-in tools

Implement a small starter set:

* read file
* write file
* list directory
* run process
* fetch URL

### Phase 5: Adapters

Implement:

* native tool schema export
* structured output adapter
* prompt-schema adapter

### Phase 5.5: Providers

Implement:

* OpenAI-compatible provider client
* provider-managed request construction
* response extraction before adapter parsing

### Phase 6: Polish

Implement:

* redaction improvements
* better logs
* examples
* docs
* helper utilities

---

## 19. Open design questions

These remain intentionally bounded.

### 19.1 Sync vs async

The library now exposes dual sync/async execution surfaces:

* tools can implement `invoke`, `ainvoke`, or both
* runtime exposes `execute` and `execute_async`
* workflow exposes sync and async one-turn execution APIs
* provider clients expose sync and async mode entrypoints

Sync compatibility is preserved while allowing non-blocking async integration.

### 19.2 Constructor dependencies

Some tools may need configuration or service objects. Pure Python registration permits this, but conventions may need refinement later.

### 19.3 Approval workflows

The policy model supports approval concepts, but full approval mechanics are likely post-v0.1.

---

## 20. Summary

This architecture is built around a small number of strict, typed abstractions:

* tools are class-based executable capabilities
* `ToolSpec` is canonical metadata on the tool
* `invoke()` returns the declared output model
* the runtime owns validation, policy, normalization, and observability
* LLM integrations are adapters layered above the canonical model
* concrete tools are separate from the API layer
* workflows are a later composition layer

The result should be a clean, explicit, testable foundation for tool-driven systems without prematurely becoming an agent framework.
