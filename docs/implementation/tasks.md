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
- [x] Implement `ExecutionRecord` generation in runtime
- [x] Capture start/end time
- [x] Capture duration
- [x] Capture validated input
- [x] Capture policy decision
- [x] Capture result status
- [x] Capture normalized error code
- [x] Capture logs and artifacts

### 6.2 Redaction
- [x] Add a simple initial redaction mechanism
- [x] Define extension points for future richer redaction rules

### 6.3 Observability tests
- [x] Test execution record creation
- [x] Test success record contents
- [x] Test failure record contents
- [x] Test redaction behavior

---

## 7. Built-in tools

### 7.1 Filesystem tools
- [x] Implement `ReadFileTool` (text files or select files using markitdown for conversion)
- [x] Implement `WriteFileTool` (text files only)
- [x] Implement `ListDirectoryTool` (recursive is optional)
- [x] Add `register_filesystem_tools(...)`

### 7.2 Git tools
- [x] Implement `RunGitStatusTool`
- [x] Implement `RunGitDiffTool`
- [x] Implement `RunGitLogTool`
- [x] Add `register_git_tools(...)`

### 7.3 Atlassian tools
- [x] Implement `SearchJiraTool`
- [x] Implement `ReadJiraIssueTool`
- [x] Add `register_atlassian_tools(...)`

### 7.4 Text tools
- [x] Implement `FileTextSearchTool`
- [x] Implement `DirectoryTextSearchTool`
- [x] Add `register_text_tools(...)`

### 7.5 Built-in tool tests
- [x] Add unit tests for each built-in tool
- [x] Add runtime integration tests using built-in tools

---

## 8. LLM adapters

### 8.1 Shared adapter abstractions
- [x] Implement adapter base interfaces
- [x] Separate tool exposure from invocation parsing cleanly
- [x] Support parsing model turns into tool invocations or a final response

### 8.2 OpenAI tool-calling adapter
- [x] Export OpenAI-compatible tool schemas from canonical tool definitions
- [x] Parse OpenAI tool call payloads into canonical turn outcomes
- [x] Parse plain final assistant responses when no tool call is present
- [x] Add adapter tests

### 8.3 Structured response adapter
- [x] Define canonical structured action/final-response schema
- [x] Export structured response schema
- [x] Parse structured model outputs into canonical turn outcomes
- [x] Add adapter tests

### 8.4 Prompt-schema adapter
- [x] Render prompt instructions for expected JSON action/final-response shape
- [x] Parse prompt-returned JSON into canonical turn outcomes
- [x] Add basic repair/retry handling
- [x] Add adapter tests

---

## 9. End-to-end integration

### 9.1 Integration flows
- [x] Register built-in tools and execute through runtime
- [x] Execute through OpenAI adapter path
- [x] Execute through structured response path
- [x] Execute through prompt-schema path

### 9.2 Integration tests
- [x] Add end-to-end tests for each adapter mode
- [x] Add tests for common failure modes across adapters

---

## 10. Documentation and examples

### 10.1 Developer documentation
- [x] Document how to define a new tool
- [x] Document how to register tools
- [x] Document runtime behavior
- [x] Document policy behavior
- [x] Document adapter behavior

### 10.2 Examples
- [x] Add minimal example project
- [x] Add example using built-in tools directly
- [x] Add example using OpenAI tool calling
- [x] Add example using structured response fallback
- [x] Add example using prompt-schema fallback

---

## 11. Apps and UX

### 11.1 Apps package foundation
- [x] Add `llm_tools.apps`
- [x] Add `llm_tools.apps.textual_workbench`
- [x] Add module and console entrypoints
- [x] Keep Textual behind an optional dependency

### 11.2 Textual workbench shell
- [x] Add a developer-facing one-turn Textual workbench
- [x] Add configuration, execution, and inspector panes
- [x] Add safe built-in tool defaults
- [x] Add provider preset and mode controls

### 11.3 Workbench execution flows
- [x] Add tool export flow
- [x] Add direct tool execution flow
- [x] Add provider-backed model-turn flow
- [x] Add result and execution-record inspection

### 11.4 Workbench tests and docs
- [x] Add launcher and controller tests
- [x] Add Textual startup and interaction tests
- [x] Add workbench usage documentation
- [x] Update repository docs for the optional apps layer

---

## 12. Approval workflow outcomes

### 12.1 Workflow outcomes
- [x] Add canonical approval-requested workflow outcomes
- [x] Add pending approval metadata model
- [x] Add approval resolve and timeout-finalization hooks

### 12.2 Workbench approval UX
- [x] Add approval queue and selected-request inspector
- [x] Add approve/deny/finalize-expired actions
- [x] Preserve in-memory approval state per workbench session

---

## 13. End-to-end async operation

### 13.1 Tool and runtime contract
- [x] Support tool implementations with `invoke` and/or `ainvoke`
- [x] Add `ToolRuntime.execute_async(...)`
- [x] Add sync/async bridging rules for sync-only and async-only tools

### 13.2 Provider and workflow async APIs
- [x] Add async provider entrypoints for all interaction modes
- [x] Add async workflow execution/resume/finalize entrypoints
- [x] Preserve existing synchronous public APIs

### 13.3 Workbench async execution
- [x] Add async controller execution methods
- [x] Use async workers for model turn and execution actions
- [x] Preserve current UI behavior and approval queue semantics

---

## 14. Deferred / post-v0.1

- [-] Manifest-based tool discovery
- [-] Plugin ecosystem support
- [-] Remote execution
- [-] Rich redaction framework
- [-] Workflow engine implementation
- [-] Version-aware multi-registration
