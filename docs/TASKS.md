# TASKS.md

## Status conventions

- [ ] Not started
- [~] In progress
- [x] Done
- [-] Deferred

---

## 0. Repository foundation

### 0.1 Project scaffolding
- [x] Review and update initial package layout favoring the existing one
- [x] Review and update `pyproject.toml` as necessary
- [x] Review and update `.gitignore`
- [x] Review and update `README.md`
- [x] Review and update `SPEC.md`
- [x] Review and update `ARCHITECTURE.md`
- [x] Review and update `TASKS.md`
- [x] Add `AGENTS.md`

### 0.2 Tooling and quality setup
- [x] Review and update formatter configuration
- [x] Review and update linter configuration
- [x] Review and update test runner configuration
- [x] Review and update type checking configuration
- [x] Review and update CI workflow
- [x] Review and update coverage reporting

Step 0 note:

- coverage is wired into local tooling and CI as report-only for now
- the repository uses `src/llm_tools/...` as the canonical package layout
- subsystem packages are scaffolded without implementing Step 1+ runtime types

---

## 1. Canonical models

### 1.1 Enumerations
- [x] Implement `SideEffectClass`
- [x] Implement `RiskLevel`
- [x] Implement `ErrorCode`
- [x] Implement any policy-related enums actually needed in v0.1

### 1.2 Tool metadata and runtime models
- [x] Implement `ToolSpec`
- [x] Implement `ToolContext`
- [x] Implement `ToolInvocationRequest`
- [x] Implement `ToolError`
- [x] Implement `ToolResult`
- [x] Implement `PolicyDecision`
- [x] Implement `ExecutionRecord`

### 1.3 Model tests
- [x] Add validation tests for canonical models
- [x] Add serialization tests
- [x] Add schema snapshot tests where useful

---

## 2. Base tool contract

### 2.1 Tool base class
- [x] Implement generic `Tool[InputT, OutputT]` base class
- [x] Enforce `spec` as a class attribute
- [x] Enforce `input_model` as a class attribute
- [x] Enforce `output_model` as a class attribute
- [x] Define strict `invoke(context, args) -> OutputT` contract

### 2.2 Tool contract tests
- [x] Add tests for valid tool subclass definitions
- [x] Add tests for invalid/misconfigured tool subclasses

---

## 3. Registry

### 3.1 Registry implementation
- [x] Implement `ToolRegistry.register()`
- [x] Implement `ToolRegistry.get()`
- [x] Implement `ToolRegistry.list_tools()`
- [x] Implement `ToolRegistry.filter_tools()`

### 3.2 Registry behavior
- [x] Reject duplicate tool names
- [x] Define and implement registry-specific errors
- [x] Ensure registry returns canonical class-level metadata cleanly

### 3.3 Registry tests
- [x] Test successful registration
- [x] Test duplicate-name rejection
- [x] Test lookup behavior
- [x] Test filtering behavior

---

## 4. Policy

### 4.1 Policy model
- [x] Implement `ToolPolicy`
- [x] Define default policy behavior for v0.1

### 4.2 Policy evaluation
- [x] Implement policy evaluation against tool metadata
- [x] Support allow/deny by tool name
- [x] Support allow/deny by tag
- [x] Support allow/deny by side-effect class
- [x] Support network/filesystem/subprocess restrictions
- [x] Produce `PolicyDecision`

### 4.3 Policy tests
- [x] Test allowed execution cases
- [x] Test denied execution cases
- [x] Test approval-required cases
- [x] Test resource restriction cases

---

## 5. Runtime

### 5.1 Runtime structure
- [x] Implement `ToolRuntime`
- [x] Implement explicit execution phases:
  - [x] resolve tool
  - [x] evaluate policy
  - [x] validate input
  - [x] invoke tool
  - [x] validate output
  - [x] normalize result
  - [x] record execution

### 5.2 Input and output validation
- [x] Validate raw arguments into the declared input model before invocation
- [x] Enforce strict output validation against the declared output model
- [x] Reject wrong output types with normalized errors

### 5.3 Error normalization
- [x] Normalize tool-not-found errors
- [x] Normalize input validation failures
- [x] Normalize policy denials
- [x] Normalize tool execution exceptions
- [x] Normalize output validation failures
- [x] Normalize unexpected runtime failures

### 5.4 Runtime tests
- [x] Test successful execution path
- [x] Test invalid input path
- [x] Test denied policy path
- [x] Test tool exception path
- [x] Test invalid output path
- [x] Test tool-not-found path

---

## 6. Observability

### 6.1 Execution records
- [ ] Implement `ExecutionRecord` generation in runtime
- [ ] Capture start/end time
- [ ] Capture duration
- [ ] Capture validated input
- [ ] Capture policy decision
- [ ] Capture result status
- [ ] Capture normalized error code
- [ ] Capture logs and artifacts

### 6.2 Redaction
- [ ] Add a simple initial redaction mechanism
- [ ] Define extension points for future richer redaction rules

### 6.3 Observability tests
- [ ] Test execution record creation
- [ ] Test success record contents
- [ ] Test failure record contents
- [ ] Test redaction behavior

---

## 7. Built-in tools

### 7.1 Filesystem tools
- [ ] Implement `ReadFileTool`
- [ ] Implement `WriteFileTool`
- [ ] Implement `ListDirectoryTool`
- [ ] Add `register_filesystem_tools(...)`

### 7.2 Process tools
- [ ] Implement `RunProcessTool`
- [ ] Add `register_process_tools(...)`

### 7.3 HTTP tools
- [ ] Implement `FetchUrlTool`
- [ ] Add `register_http_tools(...)`

### 7.4 Optional early text tools
- [ ] Implement one simple text utility tool
- [ ] Add `register_text_tools(...)`

### 7.5 Built-in tool tests
- [ ] Add unit tests for each built-in tool
- [ ] Add runtime integration tests using built-in tools

---

## 8. LLM adapters

### 8.1 Shared adapter abstractions
- [ ] Implement adapter base interfaces
- [ ] Separate tool exposure from invocation parsing cleanly

### 8.2 OpenAI tool-calling adapter
- [ ] Export OpenAI-compatible tool schemas from canonical tool definitions
- [ ] Parse OpenAI tool call payloads into `ToolInvocationRequest`
- [ ] Add adapter tests

### 8.3 Structured response adapter
- [ ] Define canonical structured action schema
- [ ] Export structured response schema
- [ ] Parse structured action payloads into `ToolInvocationRequest`
- [ ] Add adapter tests

### 8.4 Prompt-schema adapter
- [ ] Render prompt instructions for expected JSON action shape
- [ ] Parse prompt-returned JSON into `ToolInvocationRequest`
- [ ] Add basic repair/retry handling
- [ ] Add adapter tests

---

## 9. End-to-end integration

### 9.1 Integration flows
- [ ] Register built-in tools and execute through runtime
- [ ] Execute through OpenAI adapter path
- [ ] Execute through structured response path
- [ ] Execute through prompt-schema path

### 9.2 Integration tests
- [ ] Add end-to-end tests for each adapter mode
- [ ] Add tests for common failure modes across adapters

---

## 10. Documentation and examples

### 10.1 Developer documentation
- [ ] Document how to define a new tool
- [ ] Document how to register tools
- [ ] Document runtime behavior
- [ ] Document policy behavior
- [ ] Document adapter behavior

### 10.2 Examples
- [ ] Add minimal example project
- [ ] Add example using built-in tools directly
- [ ] Add example using OpenAI tool calling
- [ ] Add example using structured response fallback
- [ ] Add example using prompt-schema fallback

---

## 11. Deferred / post-v0.1

- [-] Async execution support
- [-] Manifest-based tool discovery
- [-] Plugin ecosystem support
- [-] Remote execution
- [-] Approval workflow UX
- [-] Rich redaction framework
- [-] Workflow engine implementation
- [-] Version-aware multi-registration
