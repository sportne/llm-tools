# llm-tools Context

This context captures the domain language for `llm-tools`: typed tools, model
turns, workflow execution, durable harness sessions, and assistant product
entrypoints.

## Language

**Tool**:
A typed callable capability exposed to a model and executed through the runtime.
_Avoid_: function, plugin

**Tool Runtime**:
The policy-aware execution substrate that validates, mediates, invokes, and normalizes **Tool** calls.
_Avoid_: executor, runner

**Model Turn**:
One model response cycle that may produce a final response or one or more **Tool** invocations.
_Avoid_: request, completion

**Model-Turn Protocol**:
The model-facing contract used to obtain and parse a **Model Turn**, including native structured output, staged schemas, and prompt-emitted tool calls.
_Avoid_: provider mode, strategy

**Model-Turn Event**:
A user- or operator-visible redacted progress record emitted while a **Model-Turn Protocol** is producing a **Model Turn**.
_Avoid_: UI event, chat event

**Workflow Turn**:
One parsed **Model Turn** executed through the workflow layer against the **Tool Runtime**.
_Avoid_: chat turn

**Harness Session**:
A durable, resumable record of multi-turn work built from persisted **Workflow Turn** results.
_Avoid_: job, run

**Assistant Chat**:
The interactive product flow that keeps conversational state while executing **Workflow Turn** results.
_Avoid_: chat UI, nicegui chat

**Deep Task**:
The assistant product flow that runs a user request through a durable **Harness Session**.
_Avoid_: research session

**Assistant Runtime Assembly**:
The app-layer construction of provider, **Tool** exposure, policy, prompts, protection, and execution objects for **Assistant Chat** or **Deep Task**.
_Avoid_: bootstrap, setup

## Relationships

- A **Model-Turn Protocol** produces one parsed **Model Turn**.
- A **Model-Turn Protocol** may emit zero or more **Model-Turn Events** before it produces a parsed **Model Turn**.
- A **Workflow Turn** executes one parsed **Model Turn** through the **Tool Runtime**.
- An **Assistant Chat** contains one or more **Workflow Turn** results.
- A **Deep Task** owns exactly one active **Harness Session** at a time.
- A **Harness Session** persists one or more **Workflow Turn** results.
- **Assistant Runtime Assembly** prepares the app-layer objects used by **Assistant Chat** and **Deep Task**.

## Example Dialogue

> **Dev:** "Should prompt-tool fallback live in the **Assistant Chat**?"
> **Domain expert:** "No. Prompt-tool fallback is part of the **Model-Turn Protocol**, because **Deep Task** needs the same contract before a **Workflow Turn** can execute."

## Flagged Ambiguities

- "research session" and **Deep Task** have both been used for the assistant's durable harness-backed flow; prefer **Deep Task** for product behavior and **Harness Session** for durable state.
- "provider mode" has been used for both transport configuration and the **Model-Turn Protocol**; use **Model-Turn Protocol** when discussing the model-facing parse contract.
- **Model-Turn Event** payloads are redacted at the **Model-Turn Protocol** seam; raw provider messages and responses should not be emitted to callers.
