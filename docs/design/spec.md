# SPEC.md

## 1. Overview

This project provides a low-level Python library for defining, validating, registering, executing, and exposing LLM-adjacent tools.

The system began as a foundational tool substrate and is now evolving toward a
broader agent framework. The current implementation remains strongest in the
typed tool/runtime layers, but the design should no longer treat planning,
memory, orchestration, or prompt-management as inherently out of scope for the
project.

Instead, it provides a strongly-typed, Pydantic-based tool system that can be used by:

- LLM tool-calling systems
- structured output pipelines
- prompt-driven integrations
- CLI or application logic
- workflow engines
- testing and automation systems

The core principle is:

> A tool is a typed, executable capability with explicit metadata, validation, and controlled side effects.

---

## 2. Goals

### 2.1 Primary goals

- Provide a clean, class-based abstraction for tools
- Use Pydantic for all structured data (inputs, outputs, metadata, results)
- Support consistent execution via a runtime layer
- Normalize tool invocation across multiple LLM interaction modes
- Enable safe and observable execution
- Keep the system usable without any LLM dependency
- Allow higher-level systems and first-party agent capabilities to be built on
  top

---

### 2.2 Secondary goals

- Provide built-in tools for common operations
- Support multiple exposure mechanisms (LLM, CLI, etc.)
- Make tool behavior easy to test and debug
- Maintain a small, stable core abstraction

---

## 3. Current scope limits (v0.1)

The following are not yet implemented in the current release:

- Autonomous multi-turn agent loops
- Planning systems or hierarchical execution graphs
- Long-term memory systems
- Retrieval systems or vector databases
- Prompt-as-skill abstractions
- Plugin/manifest-based tool discovery
- Distributed execution infrastructure
- UI-driven skill management systems

These are future-facing areas rather than permanent non-goals. The main design
constraint is that new agent features should extend the existing typed core
cleanly instead of bypassing it.

---

## 4. Core concepts

### 4.1 Tool

A Tool is the atomic unit of capability.

A Tool:
- has a unique name and version
- defines typed input and output models (Pydantic)
- exposes execution behavior
- declares metadata and side effects

Tools are implemented as Python classes.

---

### 4.2 ToolSpec

Structured metadata describing a tool, including:
- name
- description
- version
- side effects
- tags
- risk level
- execution hints (timeout, idempotency, determinism)
- environment requirements

---

### 4.3 ToolContext

Carries runtime information for a tool invocation:
- invocation id
- environment variables
- workspace context
- metadata
- tracing information

---

### 4.4 ToolInvocationRequest

Canonical representation of a request to invoke a tool:

- tool name
- argument payload

This is the normalized input to the runtime, regardless of source.

---

### 4.5 ToolResult

Normalized result envelope for all tool executions, including:

- success/failure
- output payload
- structured error (if any)
- logs
- artifacts
- metadata
- execution timing

---

### 4.6 ToolRegistry

Stores and provides access to registered tools.

Responsibilities:
- register tools
- lookup tools by name
- list available tools
- filter tools by metadata

The registry is not responsible for planning or selection.

---

### 4.7 ToolRuntime

Responsible for executing tools safely and consistently.

Responsibilities:
- validate inputs
- enforce policy
- invoke tool logic
- capture logs and metadata
- normalize results
- handle errors and timeouts

---

### 4.8 Policy

Defines constraints on tool execution, including:

- allow/deny tool names
- allow/deny tags
- allow/deny side-effect classes
- require approval for certain operations
- restrict network, filesystem, subprocess, or secret usage

---

### 4.9 LLM Adapters

Adapters translate between model output payloads and the internal tool
invocation model.

v0.1 uses one canonical adapter: `ActionEnvelopeAdapter`.

The adapter defines a structured envelope that contains either:

- one or more ToolInvocationRequest objects
- or a final assistant response with no tool use

`workflow_api` may consume that model-turn outcome and execute any returned
tool invocations sequentially. In the current implementation it remains a thin
one-turn bridge, but it is expected to grow toward richer agent execution
capabilities over time.

### 4.9.4 LLM Providers

Provider clients are a separate layer above adapters.

- They use the OpenAI Python SDK plus Instructor with OpenAI-compatible request
  semantics.
- They can target multiple actual providers only when those providers expose an
  OpenAI-compatible endpoint.
- They do not execute tools directly.
- They call a model using a typed response model prepared by workflow/adapters,
  then hand the parsed payload to the adapter for normalization.

---

### 4.10 Tool Exposure vs Invocation

The system distinguishes between:

- Tool Exposure:
  - How tools are described to external systems (schemas, prompts)

- Invocation Parsing:
  - How external outputs are interpreted as tool calls

These are separate concerns.

---

### 4.11 Tools (Implementations)

Concrete tool implementations live outside `tool_api`.

They are consumers of the API, not part of it.

Tools are grouped under:

```text
src/llm_tools/tools/
  filesystem/
  git/
  atlassian/
  text/
```

---

## 5. Design principles

### 5.1 Class-based foundation
Tools are defined as Python classes, not manifests or prompt templates.

### 5.2 Typed everywhere
All structured data must use Pydantic models.

### 5.3 Single source of truth
Tool definitions live in Python code. Adapters derive schemas from these definitions.

### 5.4 Separation of concerns
Clearly separate:
- tool definition
- execution runtime
- policy
- registry
- LLM adapters
- workflows

### 5.5 Explicit execution model
No hidden behavior. Validation, policy, and execution must be visible and inspectable.

### 5.6 LLM-agnostic core
The tool system must work without any LLM.

### 5.7 Observability first
All executions must produce structured, inspectable metadata.

---

## 6. Functional requirements

### 6.1 Tool definition

Tools must support:
- Pydantic input model
- Pydantic output model
- ToolSpec metadata
- execution method

---

### 6.2 Execution

The runtime must:
- validate input
- enforce policy
- execute tool
- capture logs
- handle errors
- return ToolResult

---

### 6.3 Registration

Tools must be registered via Python:

```python
registry.register(MyTool())
```

No manifest-based loading in v0.1.

---

### 6.4 LLM interoperability

The system must:

* export tool schemas for LLMs
* support structured output fallback
* support prompt-based fallback

---

### 6.5 Observability

Each execution must record:

* invocation id
* tool name/version
* start/end time
* duration
* validated inputs
* result status
* errors
* logs
* artifacts
* policy decisions

---

### 6.6 Built-in tools

The project must include example tools:

* filesystem operations
* git repository inspection
* Atlassian Jira reads
* text processing

---

### 6.7 Testing support

The system must support:

* direct tool unit testing
* runtime testing
* schema snapshot testing
* mock contexts

---

## 7. Non-functional requirements

### 7.1 Reliability

Always return structured results, even on failure.

### 7.2 Extensibility

Allow new tools, adapters, and policies without modifying core interfaces.

### 7.3 Maintainability

Keep the core small and understandable.

### 7.4 Security

Explicitly model side effects and resource access.

### 7.5 Performance

Avoid unnecessary overhead in execution paths.

---

## 8. Data model requirements

### 8.1 Side effect classification

Tools must declare side effects:

* none
* local_read
* local_write
* external_read
* external_write

---

### 8.2 Tool metadata fields

ToolSpec should include:

* name
* version
* description
* tags
* side_effects
* idempotent
* deterministic
* timeout_seconds
* risk_level
* requires_network
* requires_filesystem
* requires_subprocess
* required_secrets
* cost_hint

---

### 8.3 Result envelope

ToolResult must support:

* success flag
* output payload
* structured error
* logs
* artifacts
* metadata

---

## 9. API expectations

Typical usage:

1. Define Pydantic input/output models
2. Implement Tool class
3. Register tool
4. Invoke via runtime
5. Optionally expose via LLM adapter

---

## 10. Package structure

```text
project/
  src/
    llm_tools/
      __init__.py
      tool_api/
        models.py
        tool.py
        registry.py
        runtime.py
        policy.py
        redaction.py
        errors.py
      llm_adapters/
        base.py
        action_envelope.py
      llm_providers/
        __init__.py
        openai_compatible.py
      tools/
        filesystem/
        git/
        atlassian/
        text/
      workflow_api/
        models.py
        executor.py

  tests/
```

Step 0 establishes this package layout and tooling only. The concrete models and
runtime behavior described in this spec begin in later implementation steps.

---

## 11. Milestones

### Milestone 1: Tool API

* Tool base class
* Pydantic models
* registry
* result envelope

### Milestone 2: Runtime

* validation
* execution
* policy enforcement
* observability

### Milestone 3: LLM adapters

* canonical action-envelope adapter
* `ParsedModelResponse` normalization

### Milestone 3.5: LLM providers

* OpenAI-compatible provider layer
* Instructor-backed typed parsing
* provider mode strategy fallback (`TOOLS -> JSON -> MD_JSON`)

### Milestone 4: Built-in tools

* filesystem
* git
* Atlassian
* text

### Milestone 5: Workflows (optional early)

* simple step composition

---

## 12. Acceptance criteria (v0.1)

* Define and register tools via Python
* Execute tools through runtime with validation
* Enforce basic policy
* Return structured results
* Export canonical action-envelope schemas
* Support provider strategy fallback for OpenAI-compatible endpoints
* Provide example tools
* Include tests for core functionality

---

## 13. Future directions (not in v0.1)

* manifest-based discovery
* plugin ecosystems
* remote execution
* distributed runtimes
* advanced workflow orchestration
* approval UIs
* persistent execution logs
